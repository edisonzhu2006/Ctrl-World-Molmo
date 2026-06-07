"""
Bar charts for checkpoint-level FID and FVD (from fid_fvd_by_checkpoint.csv).

FID and FVD are one scalar per checkpoint (not per-sample distributions), so bar
plots are the appropriate format, styled similarly to analyze_checkpoint_metrics.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ckpt_utils import (
    add_rank_color_fig_note,
    checkpoint_step_labels,
    ordered_ckpt_names,
    plot_title_suffix,
    value_rank_palette,
)

BAR_WIDTH = 0.42  # match violin width in analyze_checkpoint_metrics.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FID and FVD by checkpoint from aggregated CSV.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input CSV (default: <root>/outputs/results/aggregated/metrics/fid_fvd_by_checkpoint.csv).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/outputs/results/aggregated/metrics).",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Optional checkpoint names to include. If omitted, include all.",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG path (default: <out-dir>/fid_fvd_barplot_by_checkpoint.png).",
    )
    return p.parse_args()


def _bar_labels(ax, rects, values: np.ndarray, fmt: str, *, higher_is_better: bool) -> None:
    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    for i, r in enumerate(rects):
        h = r.get_height()
        if not np.isfinite(h):
            continue
        text = (fmt % h) + (" ★" if i == best_idx else "")
        ax.annotate(
            text,
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold" if i == best_idx else "normal",
        )


def make_fid_fvd_barplot(df: pd.DataFrame, out_png: Path) -> None:
    """Expect rows with finite fid and fvd (caller filters)."""
    ckpt_order = ordered_ckpt_names(df)
    plot_df = df.set_index("ckpt_name").loc[ckpt_order].reset_index()
    x_labels = checkpoint_step_labels(plot_df, ckpt_order)
    x = np.arange(len(ckpt_order))
    title_suffix = plot_title_suffix(plot_df)
    x_margin = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle(title_suffix, fontsize=13, y=1.02)

    fid = plot_df["fid"].astype(float).to_numpy()
    fid_colors = [value_rank_palette(x_labels, fid, higher_is_better=False)[label] for label in x_labels]
    rects0 = axes[0].bar(
        x,
        fid,
        width=BAR_WIDTH,
        color=fid_colors,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    axes[0].set_xlim(-x_margin, len(x) - 1 + x_margin)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=0, ha="center")
    axes[0].set_ylabel("FID (↓ lower is better)")
    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_title("Fréchet Inception Distance")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    _bar_labels(axes[0], rects0, fid, "%.2f", higher_is_better=False)

    fvd = plot_df["fvd"].astype(float).to_numpy()
    fvd_colors = [value_rank_palette(x_labels, fvd, higher_is_better=False)[label] for label in x_labels]
    rects1 = axes[1].bar(
        x,
        fvd,
        width=BAR_WIDTH,
        color=fvd_colors,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    axes[1].set_xlim(-x_margin, len(x) - 1 + x_margin)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=0, ha="center")
    axes[1].set_ylabel("FVD (↓ lower is better)")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_title("Fréchet Video Distance")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    _bar_labels(axes[1], rects1, fvd, "%.1f", higher_is_better=False)

    fig.tight_layout(rect=[0, 0.10, 1, 0.98])
    add_rank_color_fig_note(fig, higher_is_better=False, style="scalar", y=0.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root
    csv_path = args.csv or (
        root / "outputs" / "results" / "aggregated" / "metrics" / "fid_fvd_by_checkpoint.csv"
    )
    out_dir = args.out_dir or (root / "outputs" / "results" / "aggregated" / "metrics")
    out_png = args.out_png or (out_dir / "fid_fvd_barplot_by_checkpoint.png")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"FID/FVD CSV not found: {csv_path}. Run compute_fid_fvd.py first."
        )

    df = pd.read_csv(csv_path)
    required = {"ckpt_name", "fid", "fvd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.checkpoints:
        df = df[df["ckpt_name"].isin(args.checkpoints)].copy()
        if df.empty:
            raise RuntimeError("No rows left after filtering by --checkpoints.")

    df = df.set_index("ckpt_name").loc[ordered_ckpt_names(df)].reset_index()

    fid_ok = df["fid"].apply(np.isfinite)
    fvd_ok = df["fvd"].apply(np.isfinite)
    plot_mask = fid_ok & fvd_ok
    if (~plot_mask).any():
        dropped = df.loc[~plot_mask, "ckpt_name"].tolist()
        print(f"Warning: skipping checkpoints with non-finite FID or FVD: {dropped}")
    plot_df = df.loc[plot_mask].reset_index(drop=True)
    if plot_df.empty:
        raise RuntimeError("No rows with finite FID and FVD to plot.")

    make_fid_fvd_barplot(plot_df, out_png)
    print(f"Wrote FID/FVD bar plot to: {out_png}")


if __name__ == "__main__":
    main()
