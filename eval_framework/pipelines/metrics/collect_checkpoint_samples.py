import argparse
import sys
from pathlib import Path

import pandas as pd

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import aggregated_metrics_dir, raw_results_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect per-checkpoint sample metric CSVs into one table.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = raw_results_root(args.root)
    csvs = sorted(raw_root.glob("*/*/psnr_ssim_lpips_samples.csv"))
    if not csvs:
        raise FileNotFoundError(f"No per-checkpoint sample CSVs found under {raw_root}")

    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        if "ckpt_name" not in df.columns:
            run_id = p.parent.parent.name
            checkpoint = p.parent.name
            df["ckpt_name"] = f"{run_id}/{checkpoint}"
            df["run_id"] = run_id
            df["checkpoint"] = checkpoint
        frames.append(df)
    out_df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    n_ckpt = out_df["ckpt_name"].nunique()
    print(f"Wrote combined sample metrics CSV to: {args.out} ({len(csvs)} source files, {n_ckpt} checkpoints).")
    print("Checkpoints:", ", ".join(sorted(out_df["ckpt_name"].astype(str).unique())))


if __name__ == "__main__":
    main()

