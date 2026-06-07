"""SQLite persistence for human preference judgments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT,
    pair_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    preference TEXT NOT NULL,
    displayed_a_relpath TEXT NOT NULL,
    displayed_b_relpath TEXT NOT NULL,
    underlying_model_a TEXT NOT NULL,
    underlying_model_b TEXT NOT NULL,
    order_swapped INTEGER NOT NULL,
    time_spent_seconds REAL,
    visual_realism INTEGER,
    action_consistency INTEGER,
    temporal_coherence INTEGER,
    comments TEXT,
    browser_metadata TEXT,
    raw_json TEXT NOT NULL,
    UNIQUE(session_id, pair_id)
);
"""


class JudgmentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def annotated_pair_ids(self, session_id: str) -> Set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pair_id FROM judgments WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {r["pair_id"] for r in rows}

    def insert(self, record: Dict[str, Any]) -> None:
        ts = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
        record = {**record, "timestamp": ts}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO judgments (
                    session_id, user_id, pair_id, timestamp, preference,
                    displayed_a_relpath, displayed_b_relpath,
                    underlying_model_a, underlying_model_b, order_swapped,
                    time_spent_seconds, visual_realism, action_consistency,
                    temporal_coherence, comments, browser_metadata, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["session_id"],
                    record.get("user_id"),
                    record["pair_id"],
                    ts,
                    record["preference"],
                    record["displayed_a_relpath"],
                    record["displayed_b_relpath"],
                    record["underlying_model_a"],
                    record["underlying_model_b"],
                    1 if record.get("order_swapped") else 0,
                    record.get("time_spent_seconds"),
                    record.get("visual_realism"),
                    record.get("action_consistency"),
                    record.get("temporal_coherence"),
                    record.get("comments"),
                    json.dumps(record.get("browser_metadata") or {}),
                    json.dumps(record),
                ),
            )

    def fetch_all(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT raw_json FROM judgments ORDER BY id ASC"
            ).fetchall()
        return [json.loads(r["raw_json"]) for r in rows]

    def progress(self, session_id: str, total: int) -> Dict[str, Any]:
        done = len(self.annotated_pair_ids(session_id))
        return {
            "session_id": session_id,
            "annotated": done,
            "total": total,
            "remaining": max(0, total - done),
            "complete": done >= total,
        }

    def admin_counts(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM judgments").fetchone()["c"]
            by_session = conn.execute(
                """
                SELECT session_id, COUNT(*) AS n
                FROM judgments GROUP BY session_id ORDER BY n DESC
                """
            ).fetchall()
            by_pref = conn.execute(
                """
                SELECT preference, COUNT(*) AS n
                FROM judgments GROUP BY preference ORDER BY n DESC
                """
            ).fetchall()
        return {
            "total_judgments": total,
            "by_session": [dict(r) for r in by_session],
            "by_preference": [dict(r) for r in by_pref],
        }
