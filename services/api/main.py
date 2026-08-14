"""FastAPI localhost backend for handwriting generation pipeline."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cell_diff import FillCell, detect_fill_cells, render_overlay_png
from modal_client import generate_lines_remote
from paths import OUTPUTS_DIR, REPO_ROOT
from pdf_stamp import stamp_pdf
from store import (
    add_style,
    create_job,
    ensure_styles_prepared,
    get_active_style,
    get_job,
    list_reference_pairs,
    load_state,
    register_uploaded_pair,
    set_active_style,
    update_job,
    update_style_text,
)

app = FastAPI(title="Handwriting Generation", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    template_path: str
    synthetic_path: str
    pair_id: Optional[str] = None
    name: Optional[str] = None


class CellUpdate(BaseModel):
    id: str
    text: Optional[str] = None
    enabled: Optional[bool] = None
    bbox: Optional[List[float]] = None
    page: Optional[int] = None


class JobRequest(BaseModel):
    template_path: str
    synthetic_path: str
    cells: List[Dict[str, Any]]
    max_new_tokens: int = Field(default=128, ge=16, le=256)
    # Accept legacy "steps" from old UI builds and map to max_new_tokens
    steps: Optional[int] = Field(default=None, ge=16, le=256)
    seed: Optional[int] = 0
    pair_id: Optional[str] = None
    pair_name: Optional[str] = None


class StyleTextUpdate(BaseModel):
    style_text: str


@app.on_event("startup")
def _startup_refresh_styles():
    try:
        ensure_styles_prepared()
    except Exception:
        pass


@app.get("/api/health")
def health():
    return {"ok": True, "generator": "emuru"}


@app.get("/api/pairs/reference")
def pairs_reference():
    return {"pairs": list_reference_pairs()}


@app.get("/api/state")
def state():
    return load_state()


@app.post("/api/pairs/upload")
async def pairs_upload(
    template: UploadFile = File(...),
    synthetic: UploadFile = File(...),
    name: str = Form(""),
):
    t = await template.read()
    s = await synthetic.read()
    meta = register_uploaded_pair(t, s, name or template.filename or "pair")
    return meta


@app.post("/api/detect")
def detect(req: DetectRequest):
    t = Path(req.template_path)
    s = Path(req.synthetic_path)
    if not t.exists() or not s.exists():
        raise HTTPException(400, "PDF paths not found")
    debug = OUTPUTS_DIR / "debug_diff" / (req.pair_id or t.stem)
    try:
        cells = detect_fill_cells(t, s, debug_dir=debug)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    overlay = None
    try:
        overlay_bytes = render_overlay_png(t, cells, page=0)
        overlay_path = debug / "overlay_page0.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_bytes(overlay_bytes)
        overlay = str(overlay_path)
    except Exception:
        overlay = None
    return {
        "cells": [c.to_dict() for c in cells],
        "count": len(cells),
        "debug_dir": str(debug),
        "overlay_path": overlay,
    }


@app.get("/api/overlay")
def overlay(template_path: str, cells_json: str = "[]", page: int = 0):
    import json

    t = Path(template_path)
    cells_raw = json.loads(cells_json)
    cells = [
        FillCell(
            id=c["id"],
            page=c["page"],
            bbox=tuple(c["bbox"]),
            text=c.get("text", ""),
            enabled=c.get("enabled", True),
        )
        for c in cells_raw
    ]
    data = render_overlay_png(t, cells, page=page)
    return Response(content=data, media_type="image/png")


@app.get("/api/styles")
def styles_list():
    st = ensure_styles_prepared()
    return {"styles": list(st.get("styles", {}).values()), "active_style_id": st.get("active_style_id")}


@app.post("/api/styles/upload")
async def styles_upload(
    file: UploadFile = File(...),
    style_text: str = Form(...),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if not (style_text or "").strip():
        raise HTTPException(
            400,
            "style_text is required — type the exact transcription of the handwriting PNG",
        )
    try:
        meta = add_style(data, file.filename or "style.png", style_text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"Style preprocess failed: {e}") from e
    return meta


@app.patch("/api/styles/{style_id}")
def styles_patch(style_id: str, req: StyleTextUpdate):
    try:
        return update_style_text(style_id, req.style_text)
    except KeyError:
        raise HTTPException(404, "Style not found")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/styles/{style_id}/activate")
def styles_activate(style_id: str):
    try:
        set_active_style(style_id)
    except KeyError:
        raise HTTPException(404, "Style not found")
    return {"active_style_id": style_id}


@app.get("/api/styles/{style_id}/image")
def styles_image(style_id: str, prepared: bool = False):
    st = load_state()
    style = st.get("styles", {}).get(style_id)
    if not style:
        raise HTTPException(404, "Style not found")
    path = Path(style.get("prepared_path") if prepared and style.get("prepared_path") else style["path"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, media_type="image/png")


def _run_job(job_id: str) -> None:
    try:
        update_job(job_id, status="calling_modal")
        job = get_job(job_id)
        enabled = [c for c in job["cells"] if c.get("enabled", True)]
        texts = [c.get("text") or " " for c in enabled]
        style_png, style_text = get_active_style()
        max_new_tokens = int(job.get("max_new_tokens") or job.get("steps") or 128)
        pngs = generate_lines_remote(
            texts,
            style_png,
            style_text,
            max_new_tokens=max_new_tokens,
            seed=job.get("seed"),
        )
        update_job(job_id, status="stamping")
        out = OUTPUTS_DIR / job_id / "handwritten.pdf"
        stamp_pdf(Path(job["template_path"]), enabled, pngs, out)
        # also save line previews
        lines_dir = OUTPUTS_DIR / job_id / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)
        for i, png in enumerate(pngs):
            (lines_dir / f"{i:03d}.png").write_bytes(png)
        update_job(job_id, status="done", result_pdf=str(out))
    except Exception as e:
        update_job(job_id, status="error", error=f"{e}\n{traceback.format_exc()}")


@app.post("/api/jobs")
def jobs_create(req: JobRequest):
    pair = {
        "id": req.pair_id or "adhoc",
        "name": req.pair_name or Path(req.template_path).stem,
        "template_path": req.template_path,
        "synthetic_path": req.synthetic_path,
    }
    # Prefer max_new_tokens; legacy `steps` accepted as an alias when present.
    max_new_tokens = req.max_new_tokens
    if req.steps is not None:
        max_new_tokens = req.steps
    job = create_job(pair, req.cells, max_new_tokens, req.seed)
    thread = threading.Thread(target=_run_job, args=(job["id"],), daemon=True)
    thread.start()
    return job


@app.get("/api/jobs/{job_id}")
def jobs_get(job_id: str):
    try:
        return get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")


@app.get("/api/jobs/{job_id}/result.pdf")
def jobs_result(job_id: str):
    try:
        job = get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")
    path = job.get("result_pdf")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Result not ready")
    return FileResponse(path, media_type="application/pdf", filename=f"{job_id}_handwritten.pdf")


# Serve built UI if present
WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
