"""Export human judgments to analysis-ready CSV."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from manifest import load_manifest, primary_metric
from store import JudgmentStore


def preference_to_model(
    preference: str,
    model_a: str,
    model_b: str,
    displayed_model_a: str,
    displayed_model_b: str,
    order_swapped: bool,
) -> Dict[str, Any]:
    """Map displayed A/B/tie/invalid to underlying model preference."""
    is_tie = preference == "tie"
    is_invalid = preference == "invalid"

    if is_tie or is_invalid:
        pref_model = "tie" if is_tie else "invalid"
        human_pref_binary = 0
    elif preference == "A":
        pref_model = displayed_model_a
        human_pref_binary = 1 if pref_model == model_a else (-1 if pref_model == model_b else 0)
    elif preference == "B":
        pref_model = displayed_model_b
        human_pref_binary = 1 if pref_model == model_a else (-1 if pref_model == model_b else 0)
    else:
        pref_model = None
        human_pref_binary = 0

    return {
        "preference_displayed": preference,
        "preference_model": pref_model,
        "is_tie": int(is_tie),
        "is_invalid": int(is_invalid),
        "human_pref_binary": human_pref_binary,
        "displayed_left_model": displayed_model_a,
        "displayed_right_model": displayed_model_b,
        "order_swapped": int(order_swapped),
    }


def export_csv(db_path: Path, manifest_path: Path, root: Path, out_path: Path) -> Path:
    store = JudgmentStore(db_path)
    judgments = store.fetch_all()
    manifest = {r["pair_id"]: r for r in load_manifest(manifest_path, root)}

    rows: List[Dict[str, Any]] = []
    for j in judgments:
        m = manifest.get(j["pair_id"], {})
        model_a = j.get("underlying_model_a") or m.get("model_a")
        model_b = j.get("underlying_model_b") or m.get("model_b")
        disp_a = j.get("displayed_model_a", model_a)
        disp_b = j.get("displayed_model_b", model_b)
        pref_fields = preference_to_model(
            j["preference"],
            model_a,
            model_b,
            disp_a,
            disp_b,
            bool(j.get("order_swapped")),
        )
        metric_a = j.get("metric_a")
        if metric_a is None:
            metric_a = primary_metric(m, "a")
        metric_b = j.get("metric_b")
        if metric_b is None:
            metric_b = primary_metric(m, "b")
        metric_diff = None
        if metric_a is not None and metric_b is not None:
            metric_diff = float(metric_a) - float(metric_b)

        rows.append(
            {
                "pair_id": j["pair_id"],
                "user_id": j.get("user_id") or j.get("session_id"),
                "session_id": j.get("session_id"),
                "model_a": model_a,
                "model_b": model_b,
                "displayed_left_model": pref_fields["displayed_left_model"],
                "displayed_right_model": pref_fields["displayed_right_model"],
                "preference_displayed": pref_fields["preference_displayed"],
                "preference_model": pref_fields["preference_model"],
                "is_tie": pref_fields["is_tie"],
                "is_invalid": pref_fields["is_invalid"],
                "metric_a": metric_a,
                "metric_b": metric_b,
                "metric_diff": metric_diff,
                "human_pref_binary": pref_fields["human_pref_binary"],
                "timestamp": j.get("timestamp"),
                "time_spent_seconds": j.get("time_spent_seconds"),
                "visual_realism": j.get("visual_realism"),
                "action_consistency": j.get("action_consistency"),
                "temporal_coherence": j.get("temporal_coherence"),
                "comments": j.get("comments"),
                "category": j.get("category") or m.get("category"),
                "sample_id": j.get("sample_id") or m.get("sample_id"),
                "view_id": j.get("view_id") or m.get("view_id"),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
