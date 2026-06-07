#!/usr/bin/env python3
"""Merge per-checkpoint raw PSNR CSVs for a subset into one pool table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import raw_results_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect subset of checkpoint sample metrics CSVs.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument(
        "--ckpt-names",
        nargs="+",
        required=True,
        help="Checkpoint keys like run_id/checkpoint-45000",
    )
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = raw_results_root(args.root)
    frames = []
    missing = []
    for ckpt_name in args.ckpt_names:
        run_id, checkpoint = ckpt_name.split("/", 1)
        path = raw_root / run_id / checkpoint / "psnr_ssim_lpips_samples.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        if "ckpt_name" not in df.columns:
            df["ckpt_name"] = ckpt_name
            df["run_id"] = run_id
            df["checkpoint"] = checkpoint
        frames.append(df)

    if missing:
        raise FileNotFoundError("Missing raw sample CSVs:\n  " + "\n  ".join(missing))
    if not frames:
        raise RuntimeError("No CSVs loaded.")

    out_df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows ({out_df['ckpt_name'].nunique()} checkpoints) to {args.out}")


if __name__ == "__main__":
    main()
