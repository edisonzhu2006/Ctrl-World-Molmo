#!/usr/bin/env python3
"""Analyze exported human preference judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate human preference export CSV.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("eval_framework/apps/human_eval/data/exports/judgments_export.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_framework/apps/human_eval/data/exports"),
    )
    p.add_argument("--metric-col", type=str, default="metric_diff")
    return p.parse_args()


def win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bradley-Terry style win counts per model tag."""
    models = sorted(set(df["model_a"]).union(set(df["model_b"])))
    wins = {m: 0 for m in models}
    total = {m: 0 for m in models}
    valid = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)]
    for _, r in valid.iterrows():
        a, b = r["model_a"], r["model_b"]
        total[a] += 1
        total[b] += 1
        if r["preference_model"] == a:
            wins[a] += 1
        elif r["preference_model"] == b:
            wins[b] += 1
    rows = []
    for m in models:
        rows.append({"model": m, "wins": wins[m], "comparisons": total[m], "win_rate": wins[m] / max(1, total[m])})
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def pairwise_matrix(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)]
    tags = sorted(set(valid["model_a"]).union(set(valid["model_b"])))
    mat = pd.DataFrame(np.nan, index=tags, columns=tags, dtype=float)
    for a in tags:
        for b in tags:
            if a == b:
                mat.loc[a, b] = 0.5
                continue
            sub = valid[
                ((valid["model_a"] == a) & (valid["model_b"] == b))
                | ((valid["model_a"] == b) & (valid["model_b"] == a))
            ]
            if len(sub) == 0:
                continue
            mat.loc[a, b] = float((sub["preference_model"] == a).mean())
    return mat


def metric_human_correlation(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    sub = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)].copy()
    sub = sub[sub[metric_col].notna() & sub["human_pref_binary"].notna()]
    if len(sub) < 3:
        return {"n": len(sub), "pearson_r": float("nan"), "pearson_p": float("nan")}
    r, p = pearsonr(sub[metric_col].astype(float), sub["human_pref_binary"].astype(float))
    return {"n": int(len(sub)), "pearson_r": float(r), "pearson_p": float(p)}


def metric_confusion(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Compare sign(metric_diff) vs human_pref_binary (+1 / -1)."""
    sub = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)].copy()
    sub = sub[sub[metric_col].notna() & (sub["human_pref_binary"] != 0)]
    if len(sub) == 0:
        return pd.DataFrame()
    sub["metric_pred"] = np.where(sub[metric_col] > 0, 1, np.where(sub[metric_col] < 0, -1, 0))
    sub = sub[sub["metric_pred"] != 0]
    sub["agree"] = sub["metric_pred"] == sub["human_pref_binary"]
    accuracy = float(sub["agree"].mean()) if len(sub) else float("nan")
    cm = pd.crosstab(sub["metric_pred"], sub["human_pref_binary"], rownames=["metric_pred"], colnames=["human"])
    cm.attrs["accuracy"] = accuracy
    return cm


def inter_rater_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of pair_ids where multiple raters agree on preference_model."""
    if "pair_id" not in df.columns or "user_id" not in df.columns:
        return pd.DataFrame()
    valid = df[(df["is_tie"] == 0) & (df["is_invalid"] == 0)]
    rows = []
    for pair_id, grp in valid.groupby("pair_id"):
        prefs = grp["preference_model"].tolist()
        if len(prefs) < 2:
            continue
        agree = len(set(prefs)) == 1
        rows.append({"pair_id": pair_id, "n_raters": len(prefs), "agree": int(agree)})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    summary = pd.DataFrame(
        [{"multi_rated_pairs": len(out), "agreement_rate": float(out["agree"].mean())}]
    )
    return summary


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    wr = win_rate_table(df)
    wr.to_csv(args.out_dir / "human_winrate_by_model.csv", index=False)

    mat = pairwise_matrix(df)
    mat.to_csv(args.out_dir / "human_winrate_matrix.csv")

    corr = metric_human_correlation(df, args.metric_col)
    with (args.out_dir / "human_metric_correlation.json").open("w", encoding="utf-8") as f:
        json.dump(corr, f, indent=2)

    cm = metric_confusion(df, args.metric_col)
    if len(cm):
        cm.to_csv(args.out_dir / "human_metric_confusion.csv")
        summary = {"metric_accuracy": float(cm.attrs.get("accuracy", float("nan"))), "n": int(cm.values.sum())}
    else:
        summary = {"metric_accuracy": float("nan"), "n": 0}
    with (args.out_dir / "human_metric_accuracy.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    irr = inter_rater_agreement(df)
    if len(irr):
        irr.to_csv(args.out_dir / "human_inter_rater_agreement.csv", index=False)

    print(f"Wrote analysis to {args.out_dir}")
    print(f"Pearson({args.metric_col}, human_pref_binary): r={corr.get('pearson_r')}, n={corr.get('n')}")
    print(f"Metric accuracy: {summary.get('metric_accuracy')}")


if __name__ == "__main__":
    main()
