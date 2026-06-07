import argparse
import subprocess
import sys
from pathlib import Path

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import (
    aggregated_metrics_dir,
    aggregated_metrics_overview_dir,
    aggregated_metrics_run_comparison_dir,
    aggregated_metrics_run_dir,
    comparison_slug_from_checkpoints,
    filter_metrics_df_by_run,
    infer_run_id_from_checkpoints,
    resolve_metrics_figures_tables_dirs,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_ckpt_utils import (
    add_rank_color_axis_note,
    checkpoint_step_labels,
    legend_label_for_ckpt,
    ordered_ckpt_names,
    plot_title_suffix,
    value_rank_palette,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze and compare checkpoint metric distributions.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument(
        "--samples-csv",
        type=Path,
        default=Path(
            "eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv"
        ),
        help="Input combined per-sample-per-view metrics CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output base directory. Default: metrics/overview, metrics/runs/<run_id>, or "
            "metrics/runs/<run_id>/comparisons/<slug> when --checkpoints is set."
        ),
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Training run id. Filters all checkpoints for that run, or anchors comparison output.",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Checkpoint subset to compare (same training run). Writes under runs/<run>/comparisons/.",
    )
    p.add_argument(
        "--comparison",
        type=str,
        default=None,
        help="Comparison folder name (default: checkpoint-A_vs_checkpoint-B from --checkpoints).",
    )
    return p.parse_args()


def summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_view = (
        df.groupby(["ckpt_name", "view_id"], as_index=False)
        .agg(
            n=("sample_id", "count"),
            psnr_mean=("psnr", "mean"),
            psnr_var=("psnr", lambda x: x.var(ddof=0)),
            ssim_mean=("ssim", "mean"),
            ssim_var=("ssim", lambda x: x.var(ddof=0)),
            lpips_mean=("lpips", "mean"),
            lpips_var=("lpips", lambda x: x.var(ddof=0)),
        )
        .sort_values(["ckpt_name", "view_id"])
    )

    overall = (
        df.groupby(["ckpt_name"], as_index=False)
        .agg(
            n=("sample_id", "count"),
            psnr_mean=("psnr", "mean"),
            psnr_var=("psnr", lambda x: x.var(ddof=0)),
            ssim_mean=("ssim", "mean"),
            ssim_var=("ssim", lambda x: x.var(ddof=0)),
            lpips_mean=("lpips", "mean"),
            lpips_var=("lpips", lambda x: x.var(ddof=0)),
        )
        .sort_values(["ckpt_name"])
    )
    return per_view, overall


def _metric_series_by_ckpt(df: pd.DataFrame, ckpts: list[str], metric: str) -> list[np.ndarray]:
    return [df.loc[df["ckpt_name"] == ckpt, metric].to_numpy() for ckpt in ckpts]


def _overlay_sample_scatter(ax: plt.Axes, data: list[np.ndarray], rng: np.random.Generator) -> None:
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        x = np.full(len(vals), i, dtype=float) + jitter
        ax.scatter(x, vals, s=12, alpha=0.45, color="tab:blue", edgecolors="none")


def _violin_plot_df(df: pd.DataFrame, ckpts: list[str], metric: str) -> pd.DataFrame:
    """Long-form frame with ordered categorical x (checkpoint step labels)."""
    x_labels = checkpoint_step_labels(df, ckpts)
    label_by_ckpt = dict(zip(ckpts, x_labels))
    sub = df.loc[df["ckpt_name"].isin(ckpts), ["ckpt_name", metric]].copy()
    sub["checkpoint_step"] = sub["ckpt_name"].map(label_by_ckpt)
    sub["checkpoint_step"] = pd.Categorical(
        sub["checkpoint_step"], categories=x_labels, ordered=True
    )
    return sub


def _annotate_medians(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric: str,
    x_labels: list[str],
    median_by_label: pd.Series,
    higher_is_better: bool,
) -> None:
    """Place median text above each violin (not on the inner box)."""
    medians = np.array([float(median_by_label[label]) for label in x_labels], dtype=float)
    best_idx = int(np.argmax(medians) if higher_is_better else np.argmin(medians))
    ymin, ymax = ax.get_ylim()
    y_pad = 0.06 * (ymax - ymin)

    for i, label in enumerate(x_labels):
        vals = plot_df.loc[plot_df["checkpoint_step"] == label, metric]
        y_top = float(np.percentile(vals, 98))
        med = float(median_by_label[label])
        text = f"{med:.3f}" if i != best_idx else f"{med:.3f} ★"
        ax.text(
            i,
            y_top + y_pad,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            color="#2a2a2a",
            zorder=6,
            clip_on=False,
        )

    ax.set_ylim(ymin, ymax + 1.35 * y_pad)


def _draw_seaborn_violin_box_inner(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric: str,
    x_labels: list[str],
    higher_is_better: bool,
) -> None:
    """
    Seaborn tips-style violin: KDE body with tapered tails + mini boxplot inside.

    inner='box' -> IQR bar, median dot, whiskers (aligned with boxplot semantics).
    Color + median labels: green = better median, red = worse for this metric.
    """
    median_by_label = plot_df.groupby("checkpoint_step", observed=True)[metric].median()
    palette = value_rank_palette(x_labels, median_by_label, higher_is_better=higher_is_better)

    sns.violinplot(
        data=plot_df,
        x="checkpoint_step",
        y=metric,
        hue="checkpoint_step",
        ax=ax,
        inner="box",
        cut=2,
        density_norm="width",
        palette=palette,
        linewidth=1.0,
        saturation=0.95,
        width=0.42,
        dodge=False,
        legend=False,
    )
    for coll in ax.collections:
        coll.set_edgecolor("black")
        coll.set_alpha(0.65)

    _annotate_medians(ax, plot_df, metric, x_labels, median_by_label, higher_is_better)
    note_y = -0.28 if any("\n" in label for label in x_labels) else -0.20
    add_rank_color_axis_note(
        ax, higher_is_better=higher_is_better, style="median", y=note_y
    )


def make_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    metric_meta = {
        "psnr": {"label": "PSNR", "direction": "↑ better", "higher_is_better": True},
        "ssim": {"label": "SSIM", "direction": "↑ better", "higher_is_better": True},
        "lpips": {"label": "LPIPS", "direction": "↓ better", "higher_is_better": False},
    }
    ckpts = ordered_ckpt_names(df)
    x_labels = checkpoint_step_labels(df, ckpts)
    title_suffix = plot_title_suffix(df)
    positions = list(range(1, len(ckpts) + 1))
    rng = np.random.default_rng(2026)

    # Row 1: boxplots; row 2: violin plots (same checkpoint order and x labels).
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle(title_suffix, fontsize=13, y=1.01)
    for col, (metric, meta) in enumerate(metric_meta.items()):
        data = _metric_series_by_ckpt(df, ckpts, metric)
        ax_box = axes[0, col]
        ax_violin = axes[1, col]

        ax_box.boxplot(data, positions=positions, tick_labels=x_labels, showfliers=True)
        _overlay_sample_scatter(ax_box, data, rng)
        ax_box.set_title(f"{meta['label']} boxplot ({meta['direction']})")
        ax_box.set_ylabel(meta["label"])
        ax_box.grid(True, axis="y", linestyle="--", alpha=0.35)

        violin_df = _violin_plot_df(df, ckpts, metric)
        _draw_seaborn_violin_box_inner(
            ax_violin, violin_df, metric, x_labels, meta["higher_is_better"]
        )
        ax_violin.set_title(f"{meta['label']} violin ({meta['direction']})")
        ax_violin.set_xlabel("Checkpoint step")
        ax_violin.set_ylabel(meta["label"])
        ax_violin.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(rect=[0, 0.06, 1, 0.99])
    fig.savefig(out_dir / "boxplots_all_metrics_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Standalone violin figure (same data and ordering).
    fig_v, axes_v = plt.subplots(1, 3, figsize=(18, 5.2))
    fig_v.suptitle(title_suffix, fontsize=13, y=1.02)
    for ax, (metric, meta) in zip(axes_v, metric_meta.items()):
        violin_df = _violin_plot_df(df, ckpts, metric)
        _draw_seaborn_violin_box_inner(ax, violin_df, metric, x_labels, meta["higher_is_better"])
        ax.set_title(f"{meta['label']} ({meta['direction']})")
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(meta["label"])
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig_v.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig_v.savefig(out_dir / "violins_all_metrics_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig_v)


def make_view_errorbar_plots(per_view: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("psnr_mean", "psnr_var", "PSNR", "↑ better"),
        ("ssim_mean", "ssim_var", "SSIM", "↑ better"),
        ("lpips_mean", "lpips_var", "LPIPS", "↓ better"),
    ]
    views = sorted(per_view["view_id"].drop_duplicates())
    ckpts = ordered_ckpt_names(per_view)
    title_suffix = plot_title_suffix(per_view)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle(title_suffix, fontsize=13, y=1.02)
    for ax, (mean_col, var_col, label, direction) in zip(axes, metrics):
        x = list(range(len(views)))
        for ckpt in ckpts:
            sub = per_view[per_view["ckpt_name"] == ckpt].set_index("view_id").reindex(views)
            y = sub[mean_col].values
            std = sub[var_col].fillna(0).pow(0.5).values
            ax.errorbar(
                x,
                y,
                yerr=std,
                marker="o",
                capsize=3,
                label=legend_label_for_ckpt(per_view, ckpt),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(views)
        ax.set_xlabel("View")
        ax.set_ylabel(f"{label} mean ± std ({direction})")
        ax.set_title(f"{label} per view ({direction})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / "per_view_all_metrics_mean_std.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.samples_csv.exists():
        raise FileNotFoundError(
            f"Combined samples CSV not found: {args.samples_csv}. "
            "Create it first by concatenating per-checkpoint sample CSV files."
        )

    is_comparison = bool(args.checkpoints)

    if args.out_dir is None:
        if is_comparison:
            run_id = args.run or infer_run_id_from_checkpoints(args.checkpoints)
            slug = args.comparison or comparison_slug_from_checkpoints(args.checkpoints)
            out_dir = aggregated_metrics_run_comparison_dir(args.root, run_id, slug)
        elif args.run:
            out_dir = aggregated_metrics_run_dir(args.root, args.run)
        else:
            out_dir = aggregated_metrics_overview_dir(args.root)
    else:
        out_dir = args.out_dir
    figures_dir, tables_dir = resolve_metrics_figures_tables_dirs(out_dir, args.root)

    df = pd.read_csv(args.samples_csv)

    raw_root = args.root / "outputs" / "results" / "raw"
    if raw_root.is_dir():
        raw_csvs = sorted(raw_root.glob("*/*/psnr_ssim_lpips_samples.csv"))
        on_disk = {f"{p.parent.parent.name}/{p.parent.name}" for p in raw_csvs}
        in_csv = set(df["ckpt_name"].astype(str).unique())
        missing_in_csv = sorted(on_disk - in_csv)
        if missing_in_csv:
            print(
                "WARNING: PSNR per-sample CSVs exist under results/raw but these checkpoints "
                f"have no rows in {args.samples_csv}: {missing_in_csv}\n"
                "         Re-run collect to refresh the merged table, e.g.\n"
                f"         python eval_framework/pipelines/metrics/collect_checkpoint_samples.py --root {args.root}"
            )

    required_cols = {"ckpt_name", "sample_id", "view_id", "psnr", "ssim", "lpips"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if is_comparison:
        df = df[df["ckpt_name"].isin(args.checkpoints)].copy()
        if df.empty:
            raise RuntimeError("No rows left after filtering by --checkpoints.")
        if args.run:
            run_id = args.run
            bad = sorted(
                {c for c in args.checkpoints if not str(c).startswith(f"{run_id}/")}
            )
            if bad:
                raise ValueError(
                    f"--checkpoints must be under run {run_id!r}; foreign: {bad}"
                )
    elif args.run:
        df = filter_metrics_df_by_run(df, args.run)

    per_view, overall = summary_tables(df)
    per_view.to_csv(tables_dir / "metrics_summary_per_view.csv", index=False)
    overall.to_csv(tables_dir / "metrics_summary_overall.csv", index=False)
    df.to_csv(tables_dir / "metrics_samples.csv", index=False)

    make_boxplots(df, figures_dir)
    make_view_errorbar_plots(per_view, figures_dir)

    fid_pool = aggregated_metrics_dir(args.root) / "fid_fvd_by_checkpoint.csv"
    if fid_pool.is_file():
        fid_df = pd.read_csv(fid_pool)
        if is_comparison:
            fid_df = fid_df[fid_df["ckpt_name"].isin(args.checkpoints)].copy()
        elif args.run:
            fid_df = filter_metrics_df_by_run(fid_df, args.run, ckpt_col="ckpt_name")
        if not fid_df.empty:
            fid_csv = tables_dir / "fid_fvd_by_checkpoint.csv"
            fid_df.to_csv(fid_csv, index=False)
            plot_script = Path(__file__).resolve().parent / "analyze_fid_fvd_checkpoints.py"
            print(f"Rendering FID/FVD bar plot from {fid_csv.name} ...", flush=True)
            fid_cmd = [
                sys.executable,
                str(plot_script),
                "--root",
                str(args.root),
                "--csv",
                str(fid_csv),
                "--out-dir",
                str(figures_dir),
            ]
            if args.checkpoints:
                fid_cmd.extend(["--checkpoints", *args.checkpoints])
            proc = subprocess.run(fid_cmd, check=False)
            if proc.returncode != 0:
                print(
                    "WARNING: analyze_fid_fvd_checkpoints.py failed. "
                    "Install matplotlib (see eval_framework/pipelines/metrics/requirements_fid_fvd.txt) "
                    "or run that script manually.",
                    flush=True,
                )

    print(f"Wrote tables to: {tables_dir}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()

