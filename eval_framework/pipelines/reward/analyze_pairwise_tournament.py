import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Robometer pairwise tournament outputs.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "eval_framework/outputs/results/aggregated/reward/reward_pairwise_prism_tournament.jsonl"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_framework/outputs/results/aggregated/reward"),
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_winner(row: pd.Series) -> str:
    # pred_preference: left or right
    if row["pred_preference"] == "left":
        return row["left_tag"]
    return row["right_tag"]


def build_matrix(df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=tags, columns=tags, dtype=float)
    for a in tags:
        for b in tags:
            if a == b:
                mat.loc[a, b] = 0.5
                continue
            sub = df[
                ((df["left_tag"] == a) & (df["right_tag"] == b))
                | ((df["left_tag"] == b) & (df["right_tag"] == a))
            ]
            if len(sub) == 0:
                continue
            win_rate = (sub["winner_tag"] == a).mean()
            mat.loc[a, b] = float(win_rate)
    return mat


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input))
    required = {"left_tag", "right_tag", "pred_preference", "view_id", "sample_id", "order"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in pairwise result jsonl: {sorted(missing)}")

    df["winner_tag"] = df.apply(normalize_winner, axis=1)
    tags = sorted(set(df["left_tag"]).union(set(df["right_tag"])))

    # Win-rate matrix
    wr_mat = build_matrix(df, tags)
    wr_mat.to_csv(args.out_dir / "reward_pairwise_winrate_matrix.csv")

    # Per-view win rates (winner frequency per tag)
    per_view = (
        df.groupby(["view_id", "winner_tag"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    denom = df.groupby("view_id", as_index=False).size().rename(columns={"size": "total_pairs"})
    per_view = per_view.merge(denom, on="view_id", how="left")
    per_view["win_rate"] = per_view["wins"] / per_view["total_pairs"]
    per_view.to_csv(args.out_dir / "reward_pairwise_winrate_by_view.csv", index=False)

    # Order bias table for each unordered matchup
    bias_rows = []
    uniq_pairs = set()
    for _, r in df.iterrows():
        a, b = sorted([r["left_tag"], r["right_tag"]])
        uniq_pairs.add((a, b))
    for a, b in sorted(uniq_pairs):
        sub_forward = df[(df["left_tag"] == a) & (df["right_tag"] == b)]
        sub_reverse = df[(df["left_tag"] == b) & (df["right_tag"] == a)]
        if len(sub_forward) == 0 or len(sub_reverse) == 0:
            continue
        a_win_forward = (sub_forward["winner_tag"] == a).mean()
        a_win_reverse = (sub_reverse["winner_tag"] == a).mean()
        bias_rows.append(
            {
                "tag_a": a,
                "tag_b": b,
                "a_win_forward": float(a_win_forward),
                "a_win_reverse": float(a_win_reverse),
                "order_bias_abs_diff": float(abs(a_win_forward - a_win_reverse)),
                "a_win_adjusted_mean": float(0.5 * (a_win_forward + a_win_reverse)),
            }
        )
    order_bias = pd.DataFrame(bias_rows)
    order_bias.to_csv(args.out_dir / "reward_pairwise_order_bias.csv", index=False)

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    arr = wr_mat.values.astype(float)
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    ax.set_title("Pairwise win-rate matrix (row beats column)")
    for i in range(len(tags)):
        for j in range(len(tags)):
            if np.isnan(arr[i, j]):
                txt = "NA"
            else:
                txt = f"{arr[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.out_dir / "reward_pairwise_winrate_matrix_heatmap.png", dpi=220)
    plt.close(fig)

    # Winner frequency bar chart
    winner_counts = df["winner_tag"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(winner_counts.index.tolist(), winner_counts.values.tolist())
    ax.set_ylabel("Number of wins")
    ax.set_title("Tournament total wins by tag")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(args.out_dir / "reward_pairwise_total_wins_bar.png", dpi=220)
    plt.close(fig)

    print(f"Wrote analysis outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

