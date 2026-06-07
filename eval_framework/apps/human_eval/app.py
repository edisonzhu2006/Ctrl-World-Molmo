"""FastAPI server for human pairwise video preference evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import sys

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import resolve_relpath

from export_utils import export_csv
from manifest import load_manifest, primary_metric
from store import JudgmentStore


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]  # eval_framework (apps/human_eval -> apps -> eval_framework)
DEFAULT_MANIFEST = HERE / "data" / "manifest_demo.jsonl"
DEFAULT_DB = HERE / "data" / "judgments.sqlite"


def _env_path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val) if val else default


ROOT = _env_path("HUMAN_EVAL_ROOT", DEFAULT_ROOT)
MANIFEST_PATH = _env_path("HUMAN_EVAL_MANIFEST", DEFAULT_MANIFEST)
DB_PATH = _env_path("HUMAN_EVAL_DB", DEFAULT_DB)

manifest_rows: List[Dict[str, Any]] = []
store = JudgmentStore(DB_PATH)

templates = Jinja2Templates(directory=str(HERE / "templates"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _reload_config()
    yield


app = FastAPI(title="Human Video Preference Eval", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


def _safe_relpath(rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise HTTPException(status_code=400, detail="Invalid media path")
    full = resolve_relpath(ROOT, rel).resolve()
    root_resolved = ROOT.resolve()
    if not str(full).startswith(str(root_resolved)):
        raise HTTPException(status_code=400, detail="Path escapes root")
    return full


def _media_url(rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    return f"/media/{rel.lstrip('/')}"


def _should_swap(session_id: str, pair_id: str) -> bool:
    """Deterministic per-session shuffle to debias left/right position."""
    digest = hashlib.sha256(f"{session_id}:{pair_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def _build_display_item(row: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    swap = _should_swap(session_id, row["pair_id"])
    if swap:
        disp_a_rel, disp_b_rel = row["video_b_relpath"], row["video_a_relpath"]
        model_a, model_b = row["model_b"], row["model_a"]
    else:
        disp_a_rel, disp_b_rel = row["video_a_relpath"], row["video_b_relpath"]
        model_a, model_b = row["model_a"], row["model_b"]

    ctx_rel = row.get("context_relpath")
    ctx_is_video = bool(ctx_rel and str(ctx_rel).lower().endswith((".mp4", ".webm", ".mov")))

    return {
        "pair_id": row["pair_id"],
        "category": row.get("category"),
        "sample_id": row.get("sample_id"),
        "view_id": row.get("view_id"),
        "action_id": row.get("action_id"),
        "missing_files": row.get("_missing_files", []),
        "context_url": _media_url(ctx_rel),
        "context_is_video": ctx_is_video,
        "gt_url": _media_url(row["gt_relpath"]),
        "video_a_url": _media_url(disp_a_rel),
        "video_b_url": _media_url(disp_b_rel),
        "order_swapped": swap,
        "underlying_model_a": row["model_a"],
        "underlying_model_b": row["model_b"],
        "displayed_model_a": model_a,
        "displayed_model_b": model_b,
        "displayed_a_relpath": disp_a_rel,
        "displayed_b_relpath": disp_b_rel,
    }


class SubmitBody(BaseModel):
    session_id: str
    pair_id: str
    preference: Literal["A", "B", "tie", "invalid"]
    time_spent_seconds: Optional[float] = None
    visual_realism: Optional[int] = Field(default=None, ge=1, le=5)
    action_consistency: Optional[int] = Field(default=None, ge=1, le=5)
    temporal_coherence: Optional[int] = Field(default=None, ge=1, le=5)
    comments: Optional[str] = None
    user_id: Optional[str] = None
    browser_metadata: Optional[Dict[str, Any]] = None


def _reload_config() -> None:
    global ROOT, MANIFEST_PATH, DB_PATH, store, manifest_rows
    ROOT = _env_path("HUMAN_EVAL_ROOT", DEFAULT_ROOT).resolve()
    MANIFEST_PATH = _env_path("HUMAN_EVAL_MANIFEST", DEFAULT_MANIFEST).resolve()
    DB_PATH = _env_path("HUMAN_EVAL_DB", DEFAULT_DB).resolve()
    store = JudgmentStore(DB_PATH)
    manifest_rows = load_manifest(MANIFEST_PATH, ROOT)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"total_pairs": len(manifest_rows)},
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    stats = store.admin_counts()
    return templates.TemplateResponse(
        request,
        "admin.html",
        {"stats": stats, "manifest": MANIFEST_PATH.name},
    )


@app.get("/media/{rel_path:path}")
def serve_media(rel_path: str) -> FileResponse:
    full = _safe_relpath(rel_path)
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")
    return FileResponse(full)


@app.get("/api/next")
def api_next(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    reset: bool = Query(default=False),
) -> JSONResponse:
    if not manifest_rows:
        raise HTTPException(status_code=404, detail="Manifest is empty")

    sid = session_id or secrets.token_hex(8)
    done = store.annotated_pair_ids(sid)
    if reset:
        done = set()

    pending = [r for r in manifest_rows if r["pair_id"] not in done]
    if not pending:
        return JSONResponse(
            {
                "done": True,
                "session_id": sid,
                "user_id": user_id,
                "progress": store.progress(sid, len(manifest_rows)),
            }
        )

    row = random.choice(pending)
    item = _build_display_item(row, sid)
    return JSONResponse(
        {
            "done": False,
            "session_id": sid,
            "user_id": user_id,
            "item": item,
            "progress": store.progress(sid, len(manifest_rows)),
        }
    )


@app.post("/api/submit")
def api_submit(body: SubmitBody) -> JSONResponse:
    row = next((r for r in manifest_rows if r["pair_id"] == body.pair_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pair_id")

    if body.pair_id in store.annotated_pair_ids(body.session_id):
        raise HTTPException(status_code=409, detail="Pair already annotated for this session")

    item = _build_display_item(row, body.session_id)
    record: Dict[str, Any] = {
        "session_id": body.session_id,
        "user_id": body.user_id,
        "pair_id": body.pair_id,
        "preference": body.preference,
        "time_spent_seconds": body.time_spent_seconds,
        "visual_realism": body.visual_realism,
        "action_consistency": body.action_consistency,
        "temporal_coherence": body.temporal_coherence,
        "comments": body.comments,
        "browser_metadata": body.browser_metadata,
        "order_swapped": item["order_swapped"],
        "displayed_a_relpath": item["displayed_a_relpath"],
        "displayed_b_relpath": item["displayed_b_relpath"],
        "underlying_model_a": item["underlying_model_a"],
        "underlying_model_b": item["underlying_model_b"],
        "displayed_model_a": item["displayed_model_a"],
        "displayed_model_b": item["displayed_model_b"],
        "category": row.get("category"),
        "sample_id": row.get("sample_id"),
        "view_id": row.get("view_id"),
        "action_id": row.get("action_id"),
        "metric_a": primary_metric(row, "a"),
        "metric_b": primary_metric(row, "b"),
    }

    import sqlite3

    try:
        store.insert(record)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Duplicate submission") from exc

    return JSONResponse({"ok": True, "progress": store.progress(body.session_id, len(manifest_rows))})


@app.get("/api/progress")
def api_progress(session_id: str) -> JSONResponse:
    return JSONResponse(store.progress(session_id, len(manifest_rows)))


@app.get("/api/export")
def api_export(format: str = Query(default="jsonl", pattern=r"^(jsonl|csv)$")) -> FileResponse:
    out_dir = HERE / "data" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        out_path = out_dir / "judgments_export.jsonl"
        rows = store.fetch_all()
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        out_path = out_dir / "judgments_export.csv"
        export_csv(DB_PATH, MANIFEST_PATH, ROOT, out_path)
    return FileResponse(out_path, filename=out_path.name)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run human preference evaluation server.")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 127.0.0.1 on cluster login nodes; tunnel via SSH.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=18765,
        help="TCP port (default 18765; 8765 is often taken on shared nodes). Use 0 for any free port.",
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    os.environ["HUMAN_EVAL_ROOT"] = str(args.root.resolve())
    os.environ["HUMAN_EVAL_MANIFEST"] = str(args.manifest.resolve())
    os.environ["HUMAN_EVAL_DB"] = str(args.db.resolve())

    port = args.port if args.port != 0 else _pick_free_port(args.host)
    print(f"Human eval: http://{args.host}:{port}/")
    print("From your laptop: ssh -L {port}:localhost:{port} <user>@<login-node>".format(port=port))

    try:
        uvicorn.run(
            "app:app",
            host=args.host,
            port=port,
            reload=args.reload,
        )
    except OSError as exc:
        if getattr(exc, "errno", None) == 98:
            raise SystemExit(
                f"Port {port} is already in use. Try: python app.py --port 0  (or another port, e.g. 18766)"
            ) from exc
        raise


if __name__ == "__main__":
    main()
