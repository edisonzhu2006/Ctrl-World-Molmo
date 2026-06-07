"""Path conventions for eval_framework (5-folder layout)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

LEGACY_CKPT_TAG_RE = re.compile(r"^ckpt(\d+)-(.+)$")
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def setup_sys_path(caller_file: str | Path) -> Path:
    """Insert eval_framework/lib on sys.path; return eval_framework root."""
    p = Path(caller_file).resolve()
    for parent in p.parents:
        if parent.name == "eval_framework":
            lib = parent / "lib"
            s = str(lib)
            if s not in sys.path:
                sys.path.insert(0, s)
            return parent
    raise RuntimeError(f"eval_framework root not found above {caller_file}")


def gt_dir(root: Path) -> Path:
    return root / "data" / "gt"


def predictions_dir(root: Path) -> Path:
    return root / "data" / "predictions"


def raw_results_root(root: Path) -> Path:
    return root / "outputs" / "results" / "raw"


def aggregated_metrics_dir(root: Path) -> Path:
    return root / "outputs" / "results" / "aggregated" / "metrics"


def aggregated_metrics_overview_dir(root: Path) -> Path:
    """Cross-run summary plots and tables (all checkpoints in the pool CSV)."""
    return aggregated_metrics_dir(root) / "overview"


def aggregated_metrics_runs_dir(root: Path) -> Path:
    return aggregated_metrics_dir(root) / "runs"


def aggregated_metrics_run_dir(root: Path, run_id: str) -> Path:
    """Per-training-run metrics: tables/ + figures/ for that run's checkpoints only."""
    return aggregated_metrics_runs_dir(root) / run_id


def aggregated_metrics_run_comparison_dir(
    root: Path, run_id: str, comparison_slug: str
) -> Path:
    """Compare checkpoints within one training run: runs/<run_id>/comparisons/<slug>/."""
    return aggregated_metrics_run_dir(root, run_id) / "comparisons" / comparison_slug


def infer_run_id_from_checkpoints(ckpt_names: List[str]) -> str:
    """All ckpt_name values must belong to the same run_id."""
    run_ids = {parse_checkpoint_spec(str(c))[0] for c in ckpt_names}
    if len(run_ids) != 1:
        raise ValueError(
            f"Checkpoints span multiple runs {sorted(run_ids)}; pass --run for a single training session."
        )
    return run_ids.pop()


def comparison_slug_from_checkpoints(ckpt_names: List[str]) -> str:
    """e.g. checkpoint-5000_vs_checkpoint-65000"""
    dirs = [parse_checkpoint_spec(str(c))[1] for c in ckpt_names]
    return "_vs_".join(dirs)


def metrics_output_uses_subdirs(out_dir: Path, root: Path) -> bool:
    """figures/ and tables/ under overview/ or runs/<run_id>/ (incl. comparisons/<slug>)."""
    try:
        rel = out_dir.resolve().relative_to(aggregated_metrics_dir(root).resolve())
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "overview":
        return True
    if parts[0] == "runs" and len(parts) >= 2:
        return True
    return False


def resolve_metrics_figures_tables_dirs(out_dir: Path, root: Path) -> Tuple[Path, Path]:
    """Return (figures_dir, tables_dir), creating them when using subdir layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_output_uses_subdirs(out_dir, root):
        figures_dir = out_dir / "figures"
        tables_dir = out_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        return figures_dir, tables_dir
    return out_dir, out_dir


def filter_metrics_df_by_run(df, run_id: str, *, ckpt_col: str = "ckpt_name"):
    prefix = f"{run_id}/"
    if ckpt_col in df.columns:
        mask = df[ckpt_col].astype(str).str.startswith(prefix)
    elif "run_id" in df.columns:
        mask = df["run_id"].astype(str) == run_id
    else:
        raise ValueError(f"Cannot filter by run_id={run_id}: missing {ckpt_col!r} or run_id")
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError(f"No rows for run_id={run_id!r}")
    return out


def aggregated_reward_dir(root: Path) -> Path:
    return root / "outputs" / "results" / "aggregated" / "reward"


def manifests_dir(root: Path) -> Path:
    return root / "outputs" / "manifests"


def reward_eval_logs_dir(root: Path) -> Path:
    return root / "outputs" / "logs" / "reward_eval"


def vendor_robometer_dir(root: Path) -> Path:
    return root / "vendor" / "robometer"


def run_and_checkpoint_from_ckpt_path(ckpt_path: str) -> Tuple[str, str]:
    """
    Resolve (run_id, checkpoint_dir_name) from a training checkpoint path.

    Examples:
      .../samples_20260521-181822-8560524/ckpts/checkpoint-15000
        -> (samples_20260521-181822-8560524, checkpoint-15000)
      .../models/ctrl-world/checkpoint-10000.pt
        -> (ctrl-world, checkpoint-10000)
    """
    path = Path(ckpt_path).resolve()

    def _run_id_from_parent(p: Path) -> str:
        if p.name == "ckpts":
            return p.parent.name
        return p.name

    if path.is_dir() and CHECKPOINT_DIR_RE.match(path.name):
        return _run_id_from_parent(path.parent), path.name

    parent = path.parent
    if CHECKPOINT_DIR_RE.match(parent.name):
        return _run_id_from_parent(parent.parent), parent.name

    match = re.search(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from path: {ckpt_path}")
    return _run_id_from_parent(parent), f"checkpoint-{match.group(1)}"


def parse_legacy_ckpt_tag(tag: str) -> Tuple[str, str]:
    """ckpt50000-samples_20260427-031822 -> (samples_20260427-031822, checkpoint-50000)."""
    m = LEGACY_CKPT_TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Not a legacy ckpt tag: {tag}")
    return m.group(2), f"checkpoint-{m.group(1)}"


def parse_checkpoint_spec(spec: str) -> Tuple[str, str]:
    """
    Parse a checkpoint selector.

    Accepts:
      - run_id/checkpoint-20000
      - legacy ckpt20000-samples_...
    """
    if "/" in spec:
        run_id, ckpt_dir = spec.split("/", 1)
        if not CHECKPOINT_DIR_RE.match(ckpt_dir):
            raise ValueError(f"Expected checkpoint-<step>, got {ckpt_dir}")
        return run_id, ckpt_dir
    m = LEGACY_CKPT_TAG_RE.match(spec)
    if m:
        return m.group(2), f"checkpoint-{m.group(1)}"
    raise ValueError(f"Unrecognized checkpoint spec: {spec}")


def checkpoint_step_int(checkpoint_dir: str) -> int:
    m = CHECKPOINT_DIR_RE.match(checkpoint_dir)
    if not m:
        raise ValueError(checkpoint_dir)
    return int(m.group(1))


def list_prediction_targets(
    pred_root: Path,
    run_filter: Optional[str] = None,
    steps_filter: Optional[List[int]] = None,
) -> List[Tuple[str, str, Path]]:
    """
    Discover prediction checkpoint directories.

    Layout: data/predictions/<run_id>/checkpoint-<step>/...

    Returns (run_id, checkpoint_dir_name, path_to_checkpoint_dir).
    """
    if not pred_root.exists():
        return []
    targets: List[Tuple[str, str, Path]] = []
    for run_dir in sorted(pred_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if run_filter is not None and run_dir.name != run_filter:
            continue
        for ckpt_dir in sorted(run_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            m = CHECKPOINT_DIR_RE.match(ckpt_dir.name)
            if not m:
                continue
            step = int(m.group(1))
            if steps_filter is not None and step not in steps_filter:
                continue
            targets.append((run_dir.name, ckpt_dir.name, ckpt_dir))
    return targets


def gt_video_relpath(category: str, sample_id: str, view_id: str) -> str:
    return f"data/gt/{category}/{sample_id}/{view_id}.mp4"


def prediction_video_relpath(
    run_id: str,
    checkpoint_dir: str,
    category: str,
    trial: str,
    sample_id: str,
    view_id: str,
) -> str:
    if trial:
        return (
            f"data/predictions/{run_id}/{checkpoint_dir}/{category}/{trial}/"
            f"{sample_id}/{view_id}.mp4"
        )
    return (
        f"data/predictions/{run_id}/{checkpoint_dir}/{category}/"
        f"{sample_id}/{view_id}.mp4"
    )


def resolve_relpath(root: Path, relpath: str) -> Path:
    """Resolve a manifest/CSV path relative to eval_framework root (legacy prefixes supported)."""
    relpath = str(relpath).replace("\\", "/")
    candidates = [root / relpath]
    if relpath.startswith("gt/"):
        candidates.append(gt_dir(root) / relpath[3:])
    elif relpath.startswith("predictions/"):
        candidates.append(predictions_dir(root) / relpath[len("predictions/") :])
    elif relpath.startswith("data/gt/"):
        candidates.append(root / relpath)
    elif relpath.startswith("data/predictions/"):
        candidates.append(root / relpath)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def raw_results_dir(raw_root: Path, run_id: str, checkpoint_dir: str) -> Path:
    return raw_root / run_id / checkpoint_dir
