#!/usr/bin/env python3
"""One-time migration: flat ckpt* tags -> predictions/<run>/checkpoint-<step>/ layout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import parse_legacy_ckpt_tag


def migrate_predictions(predictions_dir: Path) -> None:
    for entry in sorted(predictions_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("ckpt"):
            continue
        run_id, checkpoint_dir = parse_legacy_ckpt_tag(entry.name)
        dest = predictions_dir / run_id / checkpoint_dir
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(dest))
        print(f"predictions: {entry.name} -> {run_id}/{checkpoint_dir}")


def migrate_raw(raw_root: Path) -> None:
    for entry in sorted(raw_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("ckpt"):
            continue
        run_id, checkpoint_dir = parse_legacy_ckpt_tag(entry.name)
        dest = raw_root / run_id / checkpoint_dir
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(dest))
        print(f"raw: {entry.name} -> {run_id}/{checkpoint_dir}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    migrate_predictions(root / "data" / "predictions")
    migrate_raw(root / "outputs" / "results" / "raw")
    print("Migration complete.")


if __name__ == "__main__":
    main()
