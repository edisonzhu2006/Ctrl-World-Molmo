#!/usr/bin/env python3
"""Export human judgments SQLite DB to analysis-ready CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PKG))

from export_utils import export_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export human eval judgments to CSV.")
    p.add_argument("--db", type=Path, default=ROOT_PKG / "data" / "judgments.sqlite")
    p.add_argument("--manifest", type=Path, default=ROOT_PKG / "data" / "manifest_demo.jsonl")
    p.add_argument("--root", type=Path, default=ROOT_PKG.parent)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT_PKG / "data" / "exports" / "judgments_export.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = export_csv(args.db, args.manifest, args.root, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
