#!/usr/bin/env python3
"""Generate figures and a short markdown report from human preference exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SHORT = {
    "ctrl-world/checkpoint-10000": "ctrl-world-10k",
    "samples_20260521-181822-8560524/checkpoint-65000": "samples-65k",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "eval_framework/apps/human_eval/data/exports/ctrl-world-10k_vs_samples65k/judgments_export.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "eval_framework/apps/human_eval/data/exports/ctrl-world-10k_vs_samples65k"
        ),
    )
    return p.parse_args()


def _load_summary(out_dir: Path) -> dict:
    acc_path = out_dir / "human_metric_accuracy.json"
    corr_path = out_dir / "human_metric_correlation.json"
    acc = json.loads(acc_path.read_text()) if acc_path.is_file() else {}
    corr = json.loads(corr_path.read_text()) if corr_path.is_file() else {}
    return {**acc, **corr}


def plot_figures(df: pd.DataFrame, out_dir: Path) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    valid = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)]
    n_total = len(df)
    n_valid = len(valid)
    n_tie = int((df["is_tie"] == 1).sum())
    n_invalid = int((df["is_invalid"] == 1).sum())

    wr = pd.read_csv(out_dir / "human_winrate_by_model.csv")
    wr["label"] = wr["model"].map(lambda m: SHORT.get(m, m))
    wr = wr.sort_values("win_rate", ascending=True)

    cm = pd.read_csv(out_dir / "human_metric_confusion.csv", index_col=0)
    cm.index = cm.index.astype(int)
    cm.columns = cm.columns.astype(int)
    row_labels = ["PSNR favors\nsamples-65k", "PSNR favors\nctrl-world-10k"]
    col_labels = ["Human prefers\nsamples-65k", "Human prefers\nctrl-world-10k"]
    cm_plot = cm.reindex(index=[-1, 1], columns=[-1, 1]).fillna(0).astype(int)
    cm_plot.index = row_labels
    cm_plot.columns = col_labels

    summary = _load_summary(out_dir)
    acc = summary.get("metric_accuracy", float("nan"))
    r = summary.get("pearson_r", float("nan"))
    pval = summary.get("pearson_p", float("nan"))

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.95], wspace=0.35)

    # Panel A: judgment breakdown
    ax0 = fig.add_subplot(gs[0, 0])
    breakdown_labels = ["Decisive\n(valid)", "Tie", "Invalid"]
    breakdown_vals = [n_valid, n_tie, n_invalid]
    colors = ["#4C78A8", "#BAB0AC", "#E45756"]
    bars = ax0.bar(breakdown_labels, breakdown_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax0.set_ylabel("Number of pairs")
    ax0.set_title(f"Annotations (n={n_total})")
    ax0.set_ylim(0, max(breakdown_vals) * 1.15)
    for b, v in zip(bars, breakdown_vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", va="bottom", fontsize=11)

    # Panel B: win rate
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.barh(wr["label"], wr["win_rate"] * 100, color=["#59A14F", "#EDC948"], edgecolor="white")
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Win rate on decisive pairs (%)")
    ax1.set_title(f"Human preference (n={n_valid})")
    for i, (_, row) in enumerate(wr.iterrows()):
        ax1.text(
            row["win_rate"] * 100 + 1.5,
            i,
            f"{int(row['wins'])}/{int(row['comparisons'])}",
            va="center",
            fontsize=10,
        )

    # Panel C: confusion matrix
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title("PSNR vs human (decisive)")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    n_agree_fig = int(np.diag(cm_plot.values).sum())
    fig.suptitle(
        f"Human eval: ctrl-world-10k vs samples-65k  |  "
        f"PSNR accuracy {acc:.1%} ({n_agree_fig}/{n_valid})  |  Pearson r={r:.2f} (p={pval:.2e})",
        fontsize=13,
        y=1.02,
    )
    out_png = figures_dir / "human_eval_summary.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Standalone confusion (for slides)
    fig2, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm_plot, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"PSNR vs human preference\n(accuracy {acc:.1%}, n={n_valid})")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "human_psnr_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {out_png}")
    print(f"Wrote {figures_dir / 'human_psnr_confusion_matrix.png'}")


def write_report(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)]
    n_total = len(df)
    n_valid = len(valid)
    n_tie = int((df["is_tie"] == 1).sum())
    n_invalid = int((df["is_invalid"] == 1).sum())

    wr = pd.read_csv(out_dir / "human_winrate_by_model.csv")
    cm = pd.read_csv(out_dir / "human_metric_confusion.csv", index_col=0)
    cm.index = cm.index.astype(int)
    cm.columns = cm.columns.astype(int)
    summary = _load_summary(out_dir)

    wins_65k = int(wr.loc[wr["model"].str.contains("65000"), "wins"].iloc[0])
    wins_ctrl = int(wr.loc[wr["model"].str.contains("ctrl-world"), "wins"].iloc[0])
    acc = summary["metric_accuracy"]
    r = summary["pearson_r"]
    pval = summary["pearson_p"]
    n_agree = int(np.diag(cm.reindex(index=[-1, 1], columns=[-1, 1]).fillna(0).values).sum())

    agree_both_65k = int(cm.loc[-1, -1])
    agree_both_ctrl = int(cm.loc[1, 1])
    disagree_65k_psnr_ctrl_human = int(cm.loc[1, -1])
    disagree_ctrl_psnr_65k_human = int(cm.loc[-1, 1])

    report = f"""# Human preference report: ctrl-world-10k vs samples-65k

**Comparison:** `ctrl-world/checkpoint-10000` vs `samples_20260521-181822-8560524/checkpoint-65000` (prism, 63 sample–view pairs).

## Annotation summary

| Category | Count |
|----------|------:|
| Total judgments | {n_total} |
| Decisive (used in analysis) | {n_valid} |
| Tie (excluded) | {n_tie} |
| Invalid (excluded) | {n_invalid} |

## Human preference (decisive pairs only)

| Model | Wins | Win rate |
|-------|-----:|---------:|
| **samples-65k** | {wins_65k} | **{wins_65k/n_valid:.1%}** |
| ctrl-world-10k | {wins_ctrl} | {wins_ctrl/n_valid:.1%} |

Humans preferred **samples-65k** in roughly **7 out of 10** decisive comparisons ({wins_65k}/{n_valid}).

## PSNR agreement with human judgment

| Metric | Value |
|--------|------:|
| PSNR–human agreement (accuracy) | **{acc:.1%}** ({n_agree}/{n_valid}) |
| Pearson r (metric_diff vs preference) | **{r:.2f}** (p = {pval:.2e}, n = {n_valid}) |

### Confusion matrix (rows = PSNR prediction, columns = human preference)

| | Human prefers samples-65k | Human prefers ctrl-world-10k |
|--|--:|--:|
| **PSNR favors samples-65k** | {agree_both_65k} | {disagree_ctrl_psnr_65k_human} |
| **PSNR favors ctrl-world-10k** | {disagree_65k_psnr_ctrl_human} | {agree_both_ctrl} |

- **Agree:** PSNR and human pick the same model ({agree_both_65k + agree_both_ctrl} pairs).
- **Disagree:** PSNR favors one checkpoint but the rater preferred the other ({disagree_65k_psnr_ctrl_human + disagree_ctrl_psnr_65k_human} pairs). Most disagreements occur when PSNR favors ctrl-world-10k but humans still prefer samples-65k ({disagree_65k_psnr_ctrl_human} cases).

## Figures

- `figures/human_eval_summary.png` — breakdown, win rates, confusion matrix
- `figures/human_psnr_confusion_matrix.png` — confusion matrix only

## Short conclusion

On this prism rollout set, **human raters clearly favor samples-65k over ctrl-world-10k** ({wins_65k}/{n_valid} decisive wins). **PSNR is a moderate proxy** for those preferences: it matches the human choice about two-thirds of the time and correlates positively (r ≈ {r:.2f}), but in {disagree_65k_psnr_ctrl_human} cases PSNR ranks ctrl-world-10k higher while humans still prefer samples-65k. Qualitative judgment and pixel-wise PSNR therefore diverge on a meaningful subset of pairs.
"""
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    plot_figures(df, args.out_dir)
    write_report(df, args.out_dir)


if __name__ == "__main__":
    main()
