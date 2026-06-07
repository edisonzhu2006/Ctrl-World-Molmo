"""Checkpoint ordering and axis labels for metrics plots."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from pathlib import Path

import sys

_LIB = Path(__file__).resolve().parents[2] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from layout import CHECKPOINT_DIR_RE, parse_checkpoint_spec

RUN_DATE_RE = re.compile(r"^samples_(\d{8})")

# Muted coral → sand → sage (relative rank per metric; sage = better).
RANK_CMAP = LinearSegmentedColormap.from_list(
    "checkpoint_rank",
    ["#d9928a", "#e8d8b8", "#8fbfb0"],
)


def value_rank_palette(
    labels: List[str],
    values: Union[Sequence[float], pd.Series],
    *,
    higher_is_better: bool,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Map each label to a rank color (sage = better, coral = worse for this metric)."""
    if isinstance(values, pd.Series):
        vals = np.array([float(values[label]) for label in labels], dtype=float)
    else:
        vals = np.asarray(values, dtype=float)
    order = np.argsort(vals)
    n = len(vals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    if higher_is_better:
        scores = ranks / max(n - 1, 1)
    else:
        scores = 1.0 - ranks / max(n - 1, 1)
    return {label: RANK_CMAP(float(scores[i])) for i, label in enumerate(labels)}


def rank_color_note_text(*, higher_is_better: bool, style: str = "median") -> str:
    """Legend line for sage/coral rank colors."""
    if style == "median":
        direction_hint = "higher median ↑ better" if higher_is_better else "lower median ↓ better"
    else:
        direction_hint = "higher ↑ better" if higher_is_better else "lower ↓ better"
    return f"sage = better · coral = worse · ★ best · {direction_hint}"


def add_rank_color_axis_note(
    ax: "Axes",
    *,
    higher_is_better: bool,
    style: str = "median",
    y: Optional[float] = None,
) -> None:
    """Place rubric below x-axis labels (outside bar/violin area)."""
    y_pos = y if y is not None else (-0.26 if style == "scalar" else -0.20)
    ax.text(
        0.5,
        y_pos,
        rank_color_note_text(higher_is_better=higher_is_better, style=style),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="0.4",
        clip_on=False,
    )


def add_rank_color_fig_note(
    fig: "Figure",
    *,
    higher_is_better: bool,
    style: str = "scalar",
    y: float = 0.02,
) -> None:
    """Single rubric centered under the whole figure (e.g. FID/FVD bar row)."""
    fig.text(
        0.5,
        y,
        rank_color_note_text(higher_is_better=higher_is_better, style=style),
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.4",
    )


def samples_date_from_run_id(run_id: str) -> Optional[str]:
    """samples_20260521-... -> 2026-05-21."""
    m = RUN_DATE_RE.match(run_id)
    if not m:
        return None
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _ckpt_meta_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ckpt_name with run_id and step."""
    if "ckpt_name" not in df.columns:
        raise ValueError("DataFrame missing ckpt_name column")

    if "run_id" in df.columns and "step" in df.columns:
        meta = df[["ckpt_name", "run_id", "step"]].drop_duplicates(subset=["ckpt_name"])
        meta["step"] = meta["step"].astype(int)
        return meta

    rows = []
    for ckpt_name in df["ckpt_name"].drop_duplicates():
        spec = str(ckpt_name)
        if "/" in spec:
            run_id, checkpoint_dir = parse_checkpoint_spec(spec)
        else:
            run_id, checkpoint_dir = "unknown", spec
        m = CHECKPOINT_DIR_RE.match(checkpoint_dir)
        step = int(m.group(1)) if m else 0
        rows.append({"ckpt_name": spec, "run_id": run_id, "step": step})
    return pd.DataFrame(rows)


def ordered_ckpt_names(df: pd.DataFrame) -> List[str]:
    """Sort by training run, then checkpoint step (numeric)."""
    meta = _ckpt_meta_from_df(df)
    meta = meta.sort_values(["run_id", "step"], kind="stable")
    return meta["ckpt_name"].astype(str).tolist()


def checkpoint_step_labels(df: pd.DataFrame, ckpt_names: List[str]) -> List[str]:
    """X-axis labels: checkpoint step; disambiguate if multiple runs share a step."""
    meta = _ckpt_meta_from_df(df).set_index("ckpt_name")
    n_runs = meta["run_id"].nunique()
    labels: List[str] = []
    for ckpt in ckpt_names:
        step = int(meta.loc[ckpt, "step"])
        if n_runs == 1:
            labels.append(str(step))
        else:
            run_id = str(meta.loc[ckpt, "run_id"])
            date = samples_date_from_run_id(run_id)
            suffix = date if date else run_id.replace("samples_", "")[:8]
            labels.append(f"{step}\n({suffix})")
    return labels


def plot_title_suffix(df: pd.DataFrame) -> str:
    """Title fragment with sample date(s) for the plotted checkpoints."""
    meta = _ckpt_meta_from_df(df)
    run_ids = meta["run_id"].drop_duplicates().tolist()
    if len(run_ids) == 1:
        run_id = run_ids[0]
        date = samples_date_from_run_id(run_id)
        return f"samples date {date}" if date else run_id

    dates = []
    for run_id in run_ids:
        date = samples_date_from_run_id(run_id)
        if date and date not in dates:
            dates.append(date)
    if len(dates) == 1:
        return f"samples date {dates[0]}"
    if dates:
        return "samples dates " + ", ".join(sorted(dates))
    return "multiple training runs"


def legend_label_for_ckpt(df: pd.DataFrame, ckpt_name: str) -> str:
    meta = _ckpt_meta_from_df(df).set_index("ckpt_name")
    step = int(meta.loc[ckpt_name, "step"])
    if meta["run_id"].nunique() == 1:
        return str(step)
    run_id = str(meta.loc[ckpt_name, "run_id"])
    date = samples_date_from_run_id(run_id)
    tag = date if date else run_id
    return f"{step} ({tag})"
