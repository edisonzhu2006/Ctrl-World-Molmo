"""
Compute FID (Fréchet Inception Distance) and FVD (Fréchet Video Distance) for predicted vs GT videos.

FID uses torch-fidelity (Inception v3 features; widely used PyTorch implementation):
  https://github.com/toshas/torch-fidelity

FVD uses the I3D TorchScript checkpoint from the StyleGAN-V / TF-FVD reference line:
  https://github.com/universome/fvd-comparison/blob/master/compare_models.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_METRICS_DIR = Path(__file__).resolve().parent
_EFW = _METRICS_DIR.parents[1]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import (
    aggregated_metrics_dir,
    checkpoint_step_int,
    gt_dir,
    list_prediction_targets,
    predictions_dir,
    raw_results_dir,
    raw_results_root,
    resolve_relpath,
)

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

_METRICS_DIR = Path(__file__).resolve().parent


def _load_fvd_helpers():
    spec = importlib.util.spec_from_file_location(
        "fvd_i3d_torchscript",
        _METRICS_DIR / "fvd_i3d_torchscript.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load fvd_i3d_torchscript")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FID / FVD for videos (paired pred vs GT).")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per video (same as PSNR script).")
    p.add_argument("--category", type=str, default="prism")
    p.add_argument("--trial", type=str, default="trial_0")
    p.add_argument("--no-fid", action="store_true", help="Skip FID.")
    p.add_argument("--no-fvd", action="store_true", help="Skip FVD.")
    p.add_argument("--fid-frame-stride", type=int, default=1, help="Use every k-th frame for FID (>=1).")
    p.add_argument("--fid-batch-size", type=int, default=64, help="Batch size for torch-fidelity Inception.")
    p.add_argument("--fvd-batch-size", type=int, default=4, help="Video batch size for I3D.")
    p.add_argument(
        "--i3d-weights",
        type=Path,
        default=None,
        help="Path to i3d_torchscript.pt (downloaded once if missing).",
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Only evaluate this training run folder under predictions/.",
    )
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps (e.g. 20000,40000).",
    )
    return p.parse_args()


def _parse_steps_filter(steps: Optional[str]) -> Optional[List[int]]:
    if not steps:
        return None
    return [int(s.strip()) for s in steps.split(",") if s.strip()]


def read_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_samples_from_gt(root: Path, category: str) -> List[Dict]:
    gt_category_dir = gt_dir(root) / category
    if not gt_category_dir.exists():
        raise FileNotFoundError(f"GT category directory not found: {gt_category_dir}")
    rows: List[Dict] = []
    for sample_dir in sorted([p for p in gt_category_dir.iterdir() if p.is_dir()]):
        sample_id = sample_dir.name
        for view_file in sorted(sample_dir.glob("*.mp4")):
            rows.append(
                {
                    "category": category,
                    "sample_id": sample_id,
                    "view_id": view_file.stem,
                    "gt_relpath": str(view_file.relative_to(root)),
                }
            )
    if len(rows) == 0:
        raise RuntimeError(f"No GT videos found under {gt_category_dir}")
    return rows


def load_video_uint8(video_path: Path) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return vr.get_batch(range(len(vr))).asnumpy()


def video_frame_count(video_path: Path) -> int:
    """Frame count without decoding pixels (cheap vs full load_video_uint8)."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return int(len(vr))


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def _crop(x: np.ndarray, hh: int, ww: int) -> np.ndarray:
        y0 = (x.shape[0] - hh) // 2
        x0 = (x.shape[1] - ww) // 2
        return x[y0 : y0 + hh, x0 : x0 + ww]

    return _crop(a, h, w), _crop(b, h, w)


def resolve_paths(
    row: Dict,
    ckpt_dir: Path,
    root: Path,
    category_default: str,
    trial: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    category = row.get("category", category_default)
    sample_id = row["sample_id"]
    view_id = row["view_id"]
    gt_relpath = row["gt_relpath"]
    if trial:
        pred_path = ckpt_dir / category / trial / sample_id / f"{view_id}.mp4"
    else:
        pred_path = ckpt_dir / category / sample_id / f"{view_id}.mp4"
    gt_path = resolve_relpath(root, gt_relpath)
    if not pred_path.exists() or not gt_path.exists():
        return None, None
    return pred_path, gt_path


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def main() -> None:
    args = parse_args()
    do_fid = not args.no_fid
    do_fvd = not args.no_fvd
    if not do_fid and not do_fvd:
        raise ValueError("Nothing to compute: enable FID and/or FVD.")

    if args.fid_frame_stride < 1:
        raise ValueError("--fid-frame-stride must be >= 1")

    root = args.root
    pred_root = args.predictions_dir or predictions_dir(root)
    raw_root = raw_results_root(root)
    agg_root = aggregated_metrics_dir(root)
    raw_root.mkdir(parents=True, exist_ok=True)
    agg_root.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        samples = read_manifest(args.manifest)
    else:
        samples = discover_samples_from_gt(root=root, category=args.category)

    steps_filter = _parse_steps_filter(args.steps)
    targets = list_prediction_targets(
        pred_root, run_filter=args.run, steps_filter=steps_filter
    )
    if len(targets) == 0:
        raise FileNotFoundError(f"No checkpoint folders found under {pred_root}")

    device = torch.device(args.device)

    calculate_metrics = None
    key_metric_fid = "frechet_inception_distance"
    if do_fid:
        try:
            from torch_fidelity import calculate_metrics as _calculate_metrics  # type: ignore[import-not-found]
            from torch_fidelity.metric_fid import KEY_METRIC_FID as _KEY_METRIC_FID  # type: ignore[import-not-found]

            calculate_metrics = _calculate_metrics
            key_metric_fid = _KEY_METRIC_FID
        except ImportError as exc:
            raise ImportError(
                "FID requires `torch-fidelity` and `torchvision`. "
                "Install with: pip install -r eval_framework/pipelines/metrics/requirements_fid_fvd.txt"
            ) from exc

    fvd_mod = None
    if do_fvd:
        fvd_mod = _load_fvd_helpers()

    checkpoint_rows: List[Dict] = []
    for run_id, checkpoint_dir, ckpt_dir in targets:
        ckpt_key = f"{run_id}/{checkpoint_dir}"
        step = checkpoint_step_int(checkpoint_dir)
        per_sample_rows: List[Dict] = []
        min_t_for_fvd: Optional[int] = None
        if do_fvd:
            lengths: List[int] = []
            for row in samples:
                pred_path, gt_path = resolve_paths(row, ckpt_dir, root, args.category, args.trial)
                if pred_path is None:
                    continue
                n = min(video_frame_count(pred_path), video_frame_count(gt_path))  # type: ignore[arg-type]
                if args.max_frames is not None:
                    n = min(n, args.max_frames)
                if n >= 10:
                    lengths.append(n)
            if len(lengths) >= 2:
                min_t_for_fvd = min(lengths)
            else:
                min_t_for_fvd = None

        i3d = None
        if do_fvd and fvd_mod is not None and min_t_for_fvd is not None and min_t_for_fvd >= 10:
            i3d_path = args.i3d_weights or fvd_mod.default_i3d_cache_path()
            print(f"[{ckpt_key}] Loading I3D for FVD (one-time per checkpoint)...", flush=True)
            i3d = fvd_mod.load_i3d(i3d_path, device)

        pred_clips: List[torch.Tensor] = []
        gt_clips: List[torch.Tensor] = []

        print(
            f"[{ckpt_key}] Decoding each video once; exporting FID frames if enabled "
            f"({len(samples)} manifest rows)...",
            flush=True,
        )
        fid_val = float("nan")
        fvd_val = float("nan")
        fid_idx = 0
        pred_d: Optional[Path] = None
        gt_d: Optional[Path] = None

        fid_ctx = (
            tempfile.TemporaryDirectory(prefix="fid_frames_") if do_fid else contextlib.nullcontext()
        )
        with fid_ctx as tmp:
            if do_fid and tmp is not None:
                pred_d = Path(tmp) / "pred"
                gt_d = Path(tmp) / "gt"
                pred_d.mkdir(parents=True, exist_ok=True)
                gt_d.mkdir(parents=True, exist_ok=True)

            for row in tqdm(samples, desc=f"{ckpt_key}", leave=False):
                pred_path, gt_path = resolve_paths(row, ckpt_dir, root, args.category, args.trial)
                if pred_path is None:
                    continue

                pred = load_video_uint8(pred_path)
                gt = load_video_uint8(gt_path)  # type: ignore[arg-type]
                n = min(len(pred), len(gt))
                if args.max_frames is not None:
                    n = min(n, args.max_frames)

                rec: Dict[str, object] = {
                    "run_id": run_id,
                    "checkpoint": checkpoint_dir,
                    "step": step,
                    "ckpt_name": ckpt_key,
                    "trial": args.trial if args.trial else "",
                    "category": row.get("category", args.category),
                    "sample_id": row["sample_id"],
                    "view_id": row["view_id"],
                    "pred_relpath": str(pred_path.relative_to(root)),
                    "gt_relpath": row["gt_relpath"],
                    "num_eval_frames": n,
                }

                if do_fvd and i3d is not None and min_t_for_fvd is not None and n >= min_t_for_fvd >= 10:
                    t_use = min_t_for_fvd
                    frames_p = []
                    frames_g = []
                    for i in range(t_use):
                        pf, gf = center_crop_to_common(pred[i], gt[i])
                        frames_p.append(torch.from_numpy(pf).permute(2, 0, 1).float() / 255.0)
                        frames_g.append(torch.from_numpy(gf).permute(2, 0, 1).float() / 255.0)
                    pred_clips.append(torch.stack(frames_p, dim=1))
                    gt_clips.append(torch.stack(frames_g, dim=1))

                if do_fid and pred_d is not None and gt_d is not None:
                    for i in range(0, n, args.fid_frame_stride):
                        pf, gf = center_crop_to_common(pred[i], gt[i])
                        Image.fromarray(pf).save(pred_d / f"{fid_idx:07d}.png")
                        Image.fromarray(gf).save(gt_d / f"{fid_idx:07d}.png")
                        fid_idx += 1

                per_sample_rows.append(rec)

            if do_fvd and i3d is not None and len(pred_clips) >= 2:
                print(
                    f"[{ckpt_key}] Running I3D on {len(pred_clips)} clips (T={min_t_for_fvd}) for FVD...",
                    flush=True,
                )
                vp = torch.stack(pred_clips, dim=0)
                vg = torch.stack(gt_clips, dim=0)
                fp = fvd_mod.i3d_features_bcthw(vp, i3d, device, batch_size=args.fvd_batch_size)
                fg = fvd_mod.i3d_features_bcthw(vg, i3d, device, batch_size=args.fvd_batch_size)
                fvd_val = float(fvd_mod.frechet_distance_features(fp, fg))
                print(f"[{ckpt_key}] FVD = {fvd_val:.4f}", flush=True)
            elif do_fvd:
                fvd_val = float("nan")

            if do_fid and pred_d is not None and gt_d is not None and len(per_sample_rows) > 0:
                if fid_idx == 0:
                    fid_val = float("nan")
                else:
                    print(
                        f"[{ckpt_key}] Running torch-fidelity FID on {fid_idx} frame pairs "
                        f"(Inception pass; often several minutes, stderr progress from library)...",
                        flush=True,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        assert calculate_metrics is not None
                        metrics = calculate_metrics(
                            input1=str(pred_d),
                            input2=str(gt_d),
                            cuda=device.type == "cuda",
                            fid=True,
                            verbose=True,
                            batch_size=args.fid_batch_size,
                        )
                    fid_val = float(metrics[key_metric_fid])
                    print(f"[{ckpt_key}] FID = {fid_val:.4f}", flush=True)

        if len(per_sample_rows) == 0:
            print(f"Skipping {ckpt_key}: no matched prediction/GT pairs found.")
            continue

        per_sample_df = pd.DataFrame(per_sample_rows)
        summary = {
            "run_id": run_id,
            "checkpoint": checkpoint_dir,
            "step": step,
            "ckpt_name": ckpt_key,
            "trial": args.trial if args.trial else "",
            "category": args.category,
            "num_samples": int(len(per_sample_df)),
            "fid": fid_val,
            "fvd": fvd_val,
            "min_temporal_len_fvd": int(min_t_for_fvd) if min_t_for_fvd is not None else None,
            "num_eval_frames_mean": safe_mean(per_sample_df["num_eval_frames"]),
        }
        checkpoint_rows.append(summary)

        out_dir = raw_results_dir(raw_root, run_id, checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_sample_df.to_csv(out_dir / "fid_fvd_samples.csv", index=False)
        with (out_dir / "fid_fvd_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if len(checkpoint_rows) == 0:
        raise RuntimeError("No results computed. Check manifest and prediction file naming.")

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(["run_id", "step"])
    checkpoint_df.to_csv(agg_root / "fid_fvd_by_checkpoint.csv", index=False)
    print(f"Wrote aggregated FID/FVD metrics to {agg_root / 'fid_fvd_by_checkpoint.csv'}")


if __name__ == "__main__":
    main()
