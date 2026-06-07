"""Load and validate human preference evaluation manifests (JSONL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import resolve_relpath


REQUIRED_FIELDS = {
    "pair_id",
    "gt_relpath",
    "video_a_relpath",
    "video_b_relpath",
    "model_a",
    "model_b",
}


def load_manifest(path: Path, root: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            pair_id = str(row["pair_id"])
            if pair_id in seen_ids:
                raise ValueError(f"{path}:{line_no} duplicate pair_id: {pair_id}")
            seen_ids.add(pair_id)
            row["_missing_files"] = _missing_files(row, root)
            rows.append(row)
    return rows


def _missing_files(row: Dict[str, Any], root: Path) -> List[str]:
    missing: List[str] = []
    for key in ("gt_relpath", "video_a_relpath", "video_b_relpath", "context_relpath"):
        rel = row.get(key)
        if not rel:
            continue
        p = resolve_relpath(root, str(rel))
        if not p.exists():
            missing.append(str(rel))
    return missing


def primary_metric(row: Dict[str, Any], side: str) -> Optional[float]:
    """Return a single scalar metric for side 'a' or 'b' if present."""
    block = row.get(f"metric_{side}")
    if block is None:
        return None
    if isinstance(block, (int, float)):
        return float(block)
    if isinstance(block, dict):
        for key in ("score", "psnr", "ssim", "lpips", "fvd"):
            if key in block and block[key] is not None:
                return float(block[key])
    return None
