from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import streamlit as st


import sys

ROOT = Path(__file__).resolve().parents[2]
_LIB = ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from layout import aggregated_metrics_dir, aggregated_reward_dir, gt_dir, predictions_dir, raw_results_root

GT_ROOT = gt_dir(ROOT)
PRED_ROOT = predictions_dir(ROOT)
RESULTS_RAW = raw_results_root(ROOT)
METRICS_OUT = aggregated_metrics_dir(ROOT)
REWARD_OUT = aggregated_reward_dir(ROOT)


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_categories() -> List[str]:
    if not GT_ROOT.exists():
        return []
    return sorted([p.name for p in GT_ROOT.iterdir() if p.is_dir()])


def list_samples(category: str) -> List[str]:
    d = GT_ROOT / category
    if not d.exists():
        return []
    names = [p.name for p in d.iterdir() if p.is_dir()]

    def _sample_sort_key(name: str):
        # Numeric-aware sort for names like sample_0, sample_10.
        m = re.search(r"(\d+)$", name)
        if m:
            return (name[: m.start()], int(m.group(1)))
        return (name, -1)

    return sorted(names, key=_sample_sort_key)


def list_views(category: str, sample: str) -> List[str]:
    d = GT_ROOT / category / sample
    if not d.exists():
        return []
    return sorted([p.stem for p in d.glob("*.mp4")])


def list_checkpoint_keys() -> List[str]:
    """All run_id/checkpoint-<step> keys under predictions/."""
    if not PRED_ROOT.exists():
        return []
    keys: List[str] = []
    for run_dir in sorted(PRED_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        for ckpt_dir in sorted(run_dir.iterdir()):
            if ckpt_dir.is_dir() and ckpt_dir.name.startswith("checkpoint-"):
                keys.append(f"{run_dir.name}/{ckpt_dir.name}")
    return keys


def detect_trials(run_id: str, checkpoint_dir: str, category: str) -> List[str]:
    base = PRED_ROOT / run_id / checkpoint_dir / category
    if not base.exists():
        return []
    trial_dirs = sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    return trial_dirs


def pred_video_path(
    run_id: str, checkpoint_dir: str, category: str, trial: Optional[str], sample: str, view: str
) -> Path:
    if trial:
        return PRED_ROOT / run_id / checkpoint_dir / category / trial / sample / f"{view}.mp4"
    return PRED_ROOT / run_id / checkpoint_dir / category / sample / f"{view}.mp4"


def gt_video_path(category: str, sample: str, view: str) -> Path:
    return GT_ROOT / category / sample / f"{view}.mp4"


@st.cache_data(show_spinner=False)
def load_pairwise_df() -> pd.DataFrame:
    candidates = [
        REWARD_OUT / "reward_pairwise_prism_tournament.jsonl",
        REWARD_OUT / "reward_pairwise_builtin.jsonl",
        REWARD_OUT / "reward_pairwise_results.jsonl",
    ]
    for path in candidates:
        rows = _read_jsonl(path)
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame()


def maybe_load_rewards(video_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    rewards_path = video_path.with_name(video_path.stem + "_rewards.npy")
    success_path = video_path.with_name(video_path.stem + "_rewards_success_probs.npy")
    rewards = None
    success = None
    if rewards_path.exists():
        try:
            rewards = np.load(rewards_path)
        except Exception:
            rewards = None
    if success_path.exists():
        try:
            success = np.load(success_path)
        except Exception:
            success = None
    return rewards, success


def resolve_candidate_video(
    category: str,
    sample: str,
    view: str,
    tag: str,
    preferred_trial: Optional[str],
) -> Optional[Tuple[Path, str]]:
    """Return (video_path, tag_label) for tag in {gt or ckpt_name}."""
    if tag == "gt":
        p = gt_video_path(category, sample, view)
        return (p, "gt") if p.exists() else None

    if "/" not in tag:
        return None
    run_id, checkpoint_dir = tag.split("/", 1)
    trials = detect_trials(run_id, checkpoint_dir, category)
    if preferred_trial and preferred_trial in trials:
        trial = preferred_trial
    elif trials:
        trial = trials[0]
    else:
        trial = None

    p = pred_video_path(run_id, checkpoint_dir, category, trial, sample, view)
    if not p.exists():
        return None
    return p, tag


def lookup_pairwise_result(pref_df: pd.DataFrame, left_rel: str, right_rel: str) -> Optional[pd.Series]:
    if pref_df.empty:
        return None
    fwd = pref_df[
        pref_df.get("left_video_relpath", pd.Series(dtype=str)).astype(str).eq(left_rel)
        & pref_df.get("right_video_relpath", pd.Series(dtype=str)).astype(str).eq(right_rel)
    ]
    if not fwd.empty:
        return fwd.iloc[0]
    rev = pref_df[
        pref_df.get("left_video_relpath", pd.Series(dtype=str)).astype(str).eq(right_rel)
        & pref_df.get("right_video_relpath", pd.Series(dtype=str)).astype(str).eq(left_rel)
    ]
    if not rev.empty:
        return rev.iloc[0]
    return None


def main() -> None:
    st.set_page_config(page_title="Eval Framework Dashboard", layout="wide")
    st.title("Eval Framework Dashboard (v1)")
    st.caption("GT / Predictions / Reward curves / Binary preferences / PSNR-SSIM-LPIPS")

    categories = list_categories()
    if not categories:
        st.error(f"No categories found under `{GT_ROOT}`.")
        return

    ckpts = list_checkpoint_keys()
    if not ckpts:
        st.warning(f"No prediction checkpoints found under `{PRED_ROOT}`.")

    with st.sidebar:
        st.header("Selection")
        category = st.selectbox("Category", categories)
        sample_options = list_samples(category)
        sample = st.selectbox("Sample", sample_options) if sample_options else ""

        ckpt = st.selectbox("Checkpoint (run/step)", ckpts) if ckpts else ""
        run_id, checkpoint_dir = (ckpt.split("/", 1) if ckpt else ("", ""))
        trials = detect_trials(run_id, checkpoint_dir, category) if ckpt else []
        trial = st.selectbox("Trial", trials) if trials else None

    if not sample or not ckpt:
        st.info("Select category/sample/checkpoint to start.")
        return

    view_options = list_views(category, sample)
    if not view_options:
        st.error(f"No view videos found under `{GT_ROOT / category / sample}`.")
        return

    st.subheader("All Camera Views (GT top, Prediction bottom)")
    cols = st.columns(len(view_options))
    pred_paths: List[Path] = []
    for idx, view in enumerate(view_options):
        gt_path = gt_video_path(category, sample, view)
        pred_path = pred_video_path(run_id, checkpoint_dir, category, trial, sample, view)
        pred_paths.append(pred_path)
        with cols[idx]:
            st.markdown(f"**{view}**")
            st.caption("GT")
            if gt_path.exists():
                st.video(str(gt_path))
            else:
                st.error("GT missing")
            st.caption("Prediction")
            if pred_path.exists():
                st.video(str(pred_path))
            else:
                st.error("Prediction missing")

    st.subheader("Reward Curves (if available)")
    reward_any = False
    for p in pred_paths:
        rewards, success = maybe_load_rewards(p)
        if rewards is None and success is None:
            continue
        reward_any = True
        with st.expander(f"{p.name} curves ({p.parent.name})", expanded=False):
            if rewards is not None:
                st.line_chart(pd.DataFrame({"reward": rewards}))
            if success is not None:
                st.line_chart(pd.DataFrame({"success_prob": success}))
    if not reward_any:
        st.info("No reward/success `.npy` files found next to prediction videos for this sample.")

    st.subheader("Pairwise Preferences (same sample/view)")
    pref_df = load_pairwise_df()
    compare_tags = ["gt"] + ckpts
    csel1, csel2, csel3 = st.columns([1, 1, 1])
    with csel1:
        pair_view = st.selectbox("Pairwise view", view_options, key="pairwise_view")
    with csel2:
        left_tag = st.selectbox("Left source (ckpt or gt)", compare_tags, index=0, key="left_tag")
    with csel3:
        right_default = 1 if len(compare_tags) > 1 else 0
        right_tag = st.selectbox("Right source (ckpt or gt)", compare_tags, index=right_default, key="right_tag")

    left_res = resolve_candidate_video(category, sample, pair_view, left_tag, trial)
    right_res = resolve_candidate_video(category, sample, pair_view, right_tag, trial)

    pv1, pv2 = st.columns(2)
    left_rel = right_rel = None
    with pv1:
        st.markdown(f"**Left: {left_tag}**")
        if left_res is None:
            st.error("Left video not found for selected source.")
        else:
            left_path, _ = left_res
            left_rel = str(left_path.relative_to(ROOT))
            st.caption(left_rel)
            st.video(str(left_path))
    with pv2:
        st.markdown(f"**Right: {right_tag}**")
        if right_res is None:
            st.error("Right video not found for selected source.")
        else:
            right_path, _ = right_res
            right_rel = str(right_path.relative_to(ROOT))
            st.caption(right_rel)
            st.video(str(right_path))

    if left_rel and right_rel and not pref_df.empty:
        row = lookup_pairwise_result(pref_df, left_rel, right_rel)
        if row is None:
            st.info("No pairwise result row found for this exact left/right pair.")
        else:
            winner = None
            if "pred_preference" in row and row["pred_preference"] in ("left", "right"):
                winner = left_tag if row["pred_preference"] == "left" else right_tag
            st.success(
                f"Winner: `{winner if winner else 'unknown'}`  |  "
                f"p(left over right) = `{row.get('prediction_prob_left_over_right', 'NA')}`  |  "
                f"logit = `{row.get('preference_logit', 'NA')}`"
            )
            cols_to_show = [
                c
                for c in [
                    "pair_id",
                    "order",
                    "pair_type",
                    "left_tag",
                    "right_tag",
                    "pred_preference",
                    "prediction_prob_left_over_right",
                    "preference_logit",
                ]
                if c in row.index
            ]
            if cols_to_show:
                st.dataframe(pd.DataFrame([row[cols_to_show].to_dict()]), width="stretch")
    elif pref_df.empty:
        st.info("No pairwise preference output found in `outputs/results/aggregated/reward`.")

    st.subheader("PSNR / SSIM / LPIPS Plots")
    # Row 1: checkpoint distribution
    plot1 = METRICS_OUT / "boxplots_all_metrics_by_checkpoint.png"
    st.markdown("**Checkpoint Distribution**")
    if plot1.exists():
        st.image(str(plot1), use_container_width=True)
    else:
        st.info("`boxplots_all_metrics_by_checkpoint.png` not found.")

    # Row 2: per-view comparison
    plot2 = METRICS_OUT / "per_view_all_metrics_mean_std.png"
    st.markdown("**Per-view Comparison**")
    if plot2.exists():
        st.image(str(plot2), use_container_width=True)
    else:
        st.info("`per_view_all_metrics_mean_std.png` not found.")


if __name__ == "__main__":
    main()

