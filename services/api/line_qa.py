"""Ink QA for Emuru line PNGs. Shared by the local API fallback path."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import cv2
import numpy as np

from line_quality import INK_DARK_THRESH, MIN_WIDTH_PER_CHAR, qa_line_ok


DATE_TOKEN_RE = re.compile(
    r"(?i)^\s*\d{1,2}\s*"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"\s*\d{2,4}\s*$"
)
CONTROLLED_STATUS_VALUES = frozenset(
    {"approved", "fail", "pass", "verified"}
)


def _ink_bbox(arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(arr < INK_DARK_THRESH)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def is_short_digit_heavy(text: str) -> bool:
    """Conservative: 0.5, 10:1, 10, 34 — not initials/dates."""
    t = (text or "").replace(" ", "")
    if not t or len(t) > 6:
        return False
    alnum = [c for c in t if c.isalnum()]
    if not alnum:
        return False
    return sum(c.isdigit() for c in alnum) / len(alnum) >= 0.5


def is_date_like(text: str) -> bool:
    return DATE_TOKEN_RE.fullmatch(text or "") is not None


def is_controlled_status(text: str) -> bool:
    normalized = " ".join((text or "").casefold().split())
    return normalized in CONTROLLED_STATUS_VALUES


def short_digit_geometry_ok(arr: np.ndarray, gen_text: str) -> Tuple[bool, str]:
    """Implausible glyph count/width for a short numeric token."""
    nchar = max(1, len((gen_text or "").replace(" ", "")))
    bbox = _ink_bbox(arr)
    if bbox is None:
        return False, "no_ink"
    ink_w = bbox[2] - bbox[0]
    if ink_w < max(16, MIN_WIDTH_PER_CHAR * nchar * 0.45):
        return False, f"digit_too_narrow ink_w={ink_w} nchar={nchar}"
    if ink_w > max(220, MIN_WIDTH_PER_CHAR * nchar * 12):
        return False, f"digit_too_wide ink_w={ink_w} nchar={nchar}"

    ink = (arr < INK_DARK_THRESH).astype(np.uint8)
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n_labels <= 1:
        return False, "no_blobs"
    areas = stats[1:, cv2.CC_STAT_AREA]
    significant = int((areas >= 8).sum())
    if significant < nchar:
        return False, f"too_few_blobs n={significant} nchar={nchar}"
    if nchar <= 2 and significant > nchar:
        return False, f"too_many_blobs n={significant} nchar={nchar}"
    if significant > nchar * 5:
        return False, f"too_many_blobs n={significant} nchar={nchar}"
    return True, "ok"


def decode_gray_png(png: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(png, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return gray


def line_needs_vector(png: bytes, text: str) -> bool:
    """True if this Emuru PNG should be replaced with a vector line."""
    gray = decode_gray_png(png)
    if gray is None:
        return True
    ok, _reason = qa_line_ok(gray, text)
    if not ok:
        return True
    if (
        is_short_digit_heavy(text)
        or is_date_like(text)
        or is_controlled_status(text)
    ):
        # Pixel geometry cannot prove glyph identity. Exact-value fields use a
        # deterministic renderer even when the generated ink looks plausible.
        return True
    return False
