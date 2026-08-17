"""Simple JSON state for pairs, styles, and jobs."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import DEFAULT_STYLE_TEXT, OUTPUTS_DIR, PAIRS_DIR, REFERENCE_PDFS, STATE_FILE, STYLES_DIR
from style_prep import preprocess_style_png

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> Dict[str, Any]:
    return {
        "active_style_id": "default",
        "styles": {
            "default": {
                "id": "default",
                "name": "representative_text.png",
                "path": str(STYLES_DIR / "default" / "representative_text.png"),
                "style_text": DEFAULT_STYLE_TEXT,
                "created_at": _now(),
            }
        },
        "pairs": {},
        "jobs": {},
    }


def _ensure_style_text(style: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill style_text for styles created before Emuru cutover."""
    text = (style.get("style_text") or "").strip()
    if not text:
        style["style_text"] = (
            DEFAULT_STYLE_TEXT if style.get("id") == "default" else ""
        )
    return style


def _refresh_prepared_style(style: Dict[str, Any]) -> Dict[str, Any]:
    """Regenerate prepared.png from raw style path."""
    style = _ensure_style_text(style)
    raw_path = Path(style["path"])
    if not raw_path.exists():
        return style
    prepared_bytes = preprocess_style_png(raw_path.read_bytes())
    prep_path = raw_path.parent / "prepared.png"
    prep_path.write_bytes(prepared_bytes)
    style["prepared_path"] = str(prep_path)
    return style


def refresh_all_prepared_styles(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if state is None:
        state = load_state()
    changed = False
    for sid, style in list(state.get("styles", {}).items()):
        try:
            before = dict(style)
            state["styles"][sid] = _refresh_prepared_style(style)
            if state["styles"][sid] != before:
                changed = True
            else:
                changed = True  # prepared path may still refresh on disk
        except Exception:
            continue
    if changed:
        save_state(state)
    return state


def load_state() -> Dict[str, Any]:
    STYLES_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    default_style = STYLES_DIR / "default" / "representative_text.png"
    if not default_style.exists():
        seed = Path(__file__).resolve().parents[2] / "representative_text.png"
        default_style.parent.mkdir(parents=True, exist_ok=True)
        if seed.exists():
            shutil.copy2(seed, default_style)

    if not STATE_FILE.exists():
        state = _default_state()
        save_state(state)
        refresh_all_prepared_styles(state)
        return state
    with open(STATE_FILE) as f:
        state = json.load(f)
    # ensure default style
    if "default" not in state.get("styles", {}):
        state.setdefault("styles", {})["default"] = _default_state()["styles"]["default"]
        state.setdefault("active_style_id", "default")
        save_state(state)
    # migrate style_text onto existing entries
    migrated = False
    for sid, style in list(state.get("styles", {}).items()):
        if "style_text" not in style or style.get("style_text") is None:
            state["styles"][sid] = _ensure_style_text(style)
            migrated = True
    # If active style has empty transcription but default is ready, prefer default
    active_id = state.get("active_style_id", "default")
    active = state.get("styles", {}).get(active_id) or {}
    default = state.get("styles", {}).get("default") or {}
    if not (active.get("style_text") or "").strip() and (default.get("style_text") or "").strip():
        state["active_style_id"] = "default"
        migrated = True
    if migrated:
        save_state(state)
    return state


def ensure_styles_prepared() -> Dict[str, Any]:
    """Call on API startup / styles list to refresh prepared crops."""
    with _lock:
        state = load_state()
        return refresh_all_prepared_styles(state)


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def list_reference_pairs() -> List[Dict[str, str]]:
    """Discover TEMPLATE + synthetic pairs in reference-pdfs/."""
    if not REFERENCE_PDFS.exists():
        return []
    templates = sorted(REFERENCE_PDFS.glob("*_TEMPLATE.pdf"))
    pairs = []
    for tmpl in templates:
        stem = tmpl.name.replace("_TEMPLATE.pdf", "")
        synthetics = sorted(REFERENCE_PDFS.glob(f"{stem}_*_EXECUTED_synthetic.pdf"))
        for syn in synthetics:
            pairs.append(
                {
                    "id": f"ref::{syn.stem}",
                    "name": syn.stem,
                    "template_path": str(tmpl),
                    "synthetic_path": str(syn),
                    "source": "reference",
                }
            )
    return pairs


def register_uploaded_pair(template_bytes: bytes, synthetic_bytes: bytes, name: str) -> Dict[str, Any]:
    pair_id = uuid.uuid4().hex[:10]
    dest = PAIRS_DIR / pair_id
    dest.mkdir(parents=True, exist_ok=True)
    t_path = dest / "template.pdf"
    s_path = dest / "synthetic.pdf"
    t_path.write_bytes(template_bytes)
    s_path.write_bytes(synthetic_bytes)
    meta = {
        "id": pair_id,
        "name": name or pair_id,
        "template_path": str(t_path),
        "synthetic_path": str(s_path),
        "source": "upload",
        "created_at": _now(),
    }
    with _lock:
        state = load_state()
        state.setdefault("pairs", {})[pair_id] = meta
        save_state(state)
    return meta


def add_style(png_bytes: bytes, filename: str, style_text: str) -> Dict[str, Any]:
    style_text = (style_text or "").strip()
    if not style_text:
        raise ValueError("style_text is required (transcription of the handwriting PNG)")
    style_id = uuid.uuid4().hex[:10]
    dest_dir = STYLES_DIR / style_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / filename
    raw_path.write_bytes(png_bytes)
    prepared = preprocess_style_png(png_bytes)
    prepared_path = dest_dir / "prepared.png"
    prepared_path.write_bytes(prepared)
    meta = {
        "id": style_id,
        "name": filename,
        "path": str(raw_path),
        "prepared_path": str(prepared_path),
        "style_text": style_text,
        "created_at": _now(),
    }
    with _lock:
        state = load_state()
        state.setdefault("styles", {})[style_id] = meta
        state["active_style_id"] = style_id
        save_state(state)
    return meta


def update_style_text(style_id: str, style_text: str) -> Dict[str, Any]:
    style_text = (style_text or "").strip()
    if not style_text:
        raise ValueError("style_text is required")
    with _lock:
        state = load_state()
        if style_id not in state.get("styles", {}):
            raise KeyError(style_id)
        state["styles"][style_id]["style_text"] = style_text
        save_state(state)
        return state["styles"][style_id]


def set_active_style(style_id: str) -> None:
    with _lock:
        state = load_state()
        if style_id not in state.get("styles", {}):
            raise KeyError(style_id)
        state["active_style_id"] = style_id
        state["styles"][style_id] = _refresh_prepared_style(state["styles"][style_id])
        save_state(state)


def get_active_style_png() -> bytes:
    png, _ = get_active_style()
    return png


def get_active_style() -> Tuple[bytes, str]:
    """Return (prepared style PNG bytes, style_text transcription)."""
    state = load_state()
    sid = state.get("active_style_id", "default")
    style = _refresh_prepared_style(state["styles"][sid])
    style_text = (style.get("style_text") or "").strip()
    if not style_text:
        raise ValueError(
            f"Active style '{sid}' is missing style_text. "
            "Edit the transcription or re-upload the style PNG."
        )
    with _lock:
        state = load_state()
        state["styles"][sid] = style
        save_state(state)
    return Path(style["prepared_path"]).read_bytes(), style_text


def create_job(
    pair: Dict[str, Any],
    cells: List[Dict[str, Any]],
    max_new_tokens: int,
    seed: Optional[int],
    variation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "pair_id": pair.get("id"),
        "pair_name": pair.get("name"),
        "template_path": pair["template_path"],
        "synthetic_path": pair["synthetic_path"],
        "cells": cells,
        "max_new_tokens": max_new_tokens,
        # Keep legacy key for old UI sessions / debugging
        "steps": max_new_tokens,
        "seed": seed,
        "variation": variation,
        "created_at": _now(),
        "updated_at": _now(),
        "error": None,
        "result_pdf": None,
    }
    with _lock:
        state = load_state()
        state.setdefault("jobs", {})[job_id] = job
        save_state(state)
    return job


def update_job(job_id: str, **kwargs: Any) -> Dict[str, Any]:
    with _lock:
        state = load_state()
        job = state["jobs"][job_id]
        job.update(kwargs)
        job["updated_at"] = _now()
        state["jobs"][job_id] = job
        save_state(state)
        return job


def get_job(job_id: str) -> Dict[str, Any]:
    state = load_state()
    if job_id not in state.get("jobs", {}):
        raise KeyError(job_id)
    return state["jobs"][job_id]
