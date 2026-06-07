import argparse
import json
import sys
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
from skimage.metrics import structural_similarity
from tqdm import tqdm

# PSNR when MSE is exactly 0 is undefined (infinite). Use a finite cap so CSV/JSON stay valid.
PSNR_PERFECT_MATCH = 100.0

try:
    import lpips
except ImportError as exc:
    raise ImportError(
        "Missing dependency `lpips`. Install with: pip install lpips"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS for videos.")
    parser.add_argument("--root", type=Path, default=Path("eval_framework"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest. If not provided, samples are discovered from gt/<category>/.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Override predictions directory. Default: <root>/data/predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on compared frames per video.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="prism",
        help="Category under data/gt/ and data/predictions/ to evaluate (default: prism).",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="trial_0",
        help="Prediction trial folder name (default: trial_0). Set empty string to disable trial level.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Only evaluate this training run folder under predictions/ (e.g. samples_20260507-010313).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps to include (e.g. 20000,40000). Requires matching checkpoint-* dirs.",
    )
    return parser.parse_args()


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


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def _crop(x: np.ndarray, hh: int, ww: int) -> np.ndarray:
        y0 = (x.shape[0] - hh) // 2
        x0 = (x.shape[1] - ww) // 2
        return x[y0 : y0 + hh, x0 : x0 + ww]

    return _crop(a, h, w), _crop(b, h, w)


def psnr_uint8(gt: np.ndarray, pred: np.ndarray, data_range: float = 255.0) -> float:
    """PSNR in dB; avoids skimage divide-by-zero when frames are identical."""
    g = gt.astype(np.float64)
    p = pred.astype(np.float64)
    mse = float(np.mean((g - p) ** 2))
    if mse <= 0.0 or not np.isfinite(mse):
        return PSNR_PERFECT_MATCH
    return float(10.0 * np.log10((data_range**2) / mse))


def compute_frame_metrics(
    pred_frame: np.ndarray,
    gt_frame: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    pred_frame, gt_frame = center_crop_to_common(pred_frame, gt_frame)

    psnr = psnr_uint8(gt_frame, pred_frame, data_range=255.0)
    ssim = float(
        structural_similarity(
            gt_frame,
            pred_frame,
            channel_axis=2,
            data_range=255,
        )
    )

    pred_t = (
        torch.from_numpy(pred_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5
        - 1.0
    )
    gt_t = (
        torch.from_numpy(gt_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5
        - 1.0
    )

    with torch.no_grad():
        lpips_value = float(lpips_model(pred_t, gt_t).item())

    return psnr, ssim, lpips_value


def evaluate_sample(
    pred_path: Path,
    gt_path: Path,
    lpips_model: torch.nn.Module,
    device: torch.device,
    max_frames: int = None,
) -> Dict[str, float]:
    pred = load_video_uint8(pred_path)
    gt = load_video_uint8(gt_path)
    n = min(len(pred), len(gt))
    if max_frames is not None:
        n = min(n, max_frames)
    if n == 0:
        raise ValueError(f"No overlapping frames for {pred_path} and {gt_path}")

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []

    for i in range(n):
        p, s, l = compute_frame_metrics(pred[i], gt[i], lpips_model, device)
        psnr_values.append(p)
        ssim_values.append(s)
        lpips_values.append(l)

    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "lpips": float(np.mean(lpips_values)),
        "num_eval_frames": n,
    }


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def main() -> None:
    args = parse_args()
    root = args.root
    manifest: Optional[Path] = args.manifest
    pred_root = args.predictions_dir or predictions_dir(root)
    raw_root = raw_results_root(root)
    agg_root = aggregated_metrics_dir(root)
    raw_root.mkdir(parents=True, exist_ok=True)
    agg_root.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        samples = read_manifest(manifest)
    else:
        samples = discover_samples_from_gt(root=root, category=args.category)

    steps_filter = _parse_steps_filter(args.steps)
    targets = list_prediction_targets(
        pred_root, run_filter=args.run, steps_filter=steps_filter
    )
    if len(targets) == 0:
        raise FileNotFoundError(
            f"No checkpoint folders found under {pred_root}"
            + (f" for run={args.run}" if args.run else "")
            + (f" steps={args.steps}" if args.steps else "")
        )

    device = torch.device(args.device)
    # lpips loads torchvision AlexNet with deprecated `pretrained=`; silence known noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="torchvision.models._utils",
        )
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    checkpoint_rows: List[Dict] = []
    for run_id, checkpoint_dir, ckpt_dir in targets:
        ckpt_key = f"{run_id}/{checkpoint_dir}"
        step = checkpoint_step_int(checkpoint_dir)
        per_sample_rows: List[Dict] = []
        print(f"Evaluating {ckpt_key} ({len(samples)} sample-view pairs)")
        for row in tqdm(samples, leave=False):
            category = row.get("category", args.category)
            sample_id = row["sample_id"]
            view_id = row["view_id"]
            gt_relpath = row["gt_relpath"]

            if args.trial:
                pred_path = ckpt_dir / category / args.trial / sample_id / f"{view_id}.mp4"
            else:
                pred_path = ckpt_dir / category / sample_id / f"{view_id}.mp4"
            gt_path = resolve_relpath(root, gt_relpath)
            if not pred_path.exists() or not gt_path.exists():
                continue

            scores = evaluate_sample(
                pred_path=pred_path,
                gt_path=gt_path,
                lpips_model=lpips_model,
                device=device,
                max_frames=args.max_frames,
            )
            per_sample_rows.append(
                {
                    "run_id": run_id,
                    "checkpoint": checkpoint_dir,
                    "step": step,
                    "ckpt_name": ckpt_key,
                    "trial": args.trial if args.trial else "",
                    "category": category,
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "pred_relpath": str(pred_path.relative_to(root)),
                    "gt_relpath": gt_relpath,
                    **scores,
                }
            )

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
            "psnr": safe_mean(per_sample_df["psnr"]),
            "ssim": safe_mean(per_sample_df["ssim"]),
            "lpips": safe_mean(per_sample_df["lpips"]),
            "psnr_var": float(per_sample_df["psnr"].var(ddof=0)),
            "ssim_var": float(per_sample_df["ssim"].var(ddof=0)),
            "lpips_var": float(per_sample_df["lpips"].var(ddof=0)),
            "num_eval_frames_mean": safe_mean(per_sample_df["num_eval_frames"]),
        }
        checkpoint_rows.append(summary)

        out_dir = raw_results_dir(raw_root, run_id, checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_sample_df.to_csv(out_dir / "psnr_ssim_lpips_samples.csv", index=False)
        with (out_dir / "psnr_ssim_lpips_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if len(checkpoint_rows) == 0:
        raise RuntimeError("No results computed. Check manifest and prediction file naming.")

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(["run_id", "step"])
    checkpoint_df.to_csv(agg_root / "metrics_by_checkpoint.csv", index=False)
    print(f"Wrote aggregated checkpoint metrics to {agg_root / 'metrics_by_checkpoint.csv'}")


if __name__ == "__main__":
    main()
