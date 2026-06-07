#!/usr/bin/env python3
"""Generate human-eval pairwise manifest from eval_framework layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys

_EFW = Path(__file__).resolve().parents[3]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import (
    gt_dir,
    gt_video_relpath,
    parse_checkpoint_spec,
    prediction_video_relpath,
    predictions_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate human preference manifest (JSONL).")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument("--category", type=str, default="prism")
    p.add_argument("--trial", type=str, default="trial_0")
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint selectors: run_id/checkpoint-<step> (e.g. ctrl-world/checkpoint-10000).",
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional psnr_ssim_lpips_samples.csv to attach metric_a/metric_b.",
    )
    p.add_argument("--metric-key", type=str, default="psnr", help="Column used as scalar metric.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("eval_framework/apps/human_eval/data/manifest_prism.jsonl"),
    )
    return p.parse_args()


def list_samples_and_views(gt_category_dir: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for sample_dir in sorted([p for p in gt_category_dir.iterdir() if p.is_dir()]):
        for view_file in sorted(sample_dir.glob("*.mp4")):
            rows.append((sample_dir.name, view_file.stem))
    return rows


def load_metrics(csv_path: Path, metric_key: str) -> Dict[Tuple[str, str, str], float]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if metric_key not in df.columns:
        raise ValueError(f"Column {metric_key} not in {csv_path}")
    out: Dict[Tuple[str, str, str], float] = {}
    for _, r in df.iterrows():
        ckpt = str(r.get("ckpt_name") or r.get("checkpoint") or r.get("exp"))
        sample = str(r["sample_id"])
        view = str(r["view_id"])
        out[(ckpt, sample, view)] = float(r[metric_key])
    return out


def main() -> None:
    args = parse_args()
    gt_category_dir = gt_dir(args.root) / args.category
    if not gt_category_dir.exists():
        raise FileNotFoundError(gt_category_dir)

    metrics_lookup = None
    if args.metrics_csv and args.metrics_csv.exists():
        metrics_lookup = load_metrics(args.metrics_csv, args.metric_key)

    units = list_samples_and_views(gt_category_dir)
    ckpt_keys: List[str] = []
    for spec in args.checkpoints:
        run_id, checkpoint_dir = parse_checkpoint_spec(spec)
        tag = f"{run_id}/{checkpoint_dir}"
        trial_root = (
            predictions_dir(args.root) / run_id / checkpoint_dir / args.category / args.trial
        )
        if not trial_root.exists():
            raise FileNotFoundError(f"Missing predictions: {trial_root}")
        ckpt_keys.append(tag)
    out_rows: List[Dict] = []
    pid = 0

    for sample_id, view_id in units:
        for i in range(len(ckpt_keys)):
            for j in range(i + 1, len(ckpt_keys)):
                ma, mb = ckpt_keys[i], ckpt_keys[j]
                run_a, ckpt_a = parse_checkpoint_spec(ma)
                run_b, ckpt_b = parse_checkpoint_spec(mb)
                row = {
                    "pair_id": f"pair_{pid:07d}",
                    "category": args.category,
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "gt_relpath": gt_video_relpath(args.category, sample_id, view_id),
                    "model_a": ma,
                    "model_b": mb,
                    "video_a_relpath": prediction_video_relpath(
                        run_a, ckpt_a, args.category, args.trial, sample_id, view_id
                    ),
                    "video_b_relpath": prediction_video_relpath(
                        run_b, ckpt_b, args.category, args.trial, sample_id, view_id
                    ),
                    "meta": {"trial": args.trial},
                }
                if metrics_lookup is not None:
                    row["metric_a"] = metrics_lookup.get((ma, sample_id, view_id))
                    row["metric_b"] = metrics_lookup.get((mb, sample_id, view_id))
                out_rows.append(row)
                pid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(out_rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
