"""Detect fill cells from template vs filled PDFs.

Primary signal: synthetic text spans that are not already on the template at
that location (black or blue). Blue typed ink is still treated as a fill so
legacy synthetics keep working.

Each fill is paired to the Nth occurrence of its left-row label on the template
(instance-index matching). Header-over-value cells (SIGN-OFF, approvals) fall
back to the column header above. Synthetic-only overflow rows are dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

from paths import EMURU_ALLOWED
from table_geometry import (
    TABLE_RULE_INSET_PTS,
    HorizontalRule,
    VerticalRule,
    extract_horizontal_rules,
    extract_vertical_rules,
    nearest_right_rule,
    row_below_header,
)

DPI = 200
IOU_MATCH = 0.15
CENTER_DIST_PTS = 12.0
MERGE_GAP_PTS = 14.0
ANCHOR_Y_TOL_PTS = 8.0
ANCHOR_X_GAP_PTS = 4.0
ANCHOR_MATCH_TOL_PTS = 3.0
HEADER_ANCHOR_SLACK_PTS = 2.0
HEADER_ANCHOR_MAX_GAP_PTS = 56.0
MAX_CELL_CHARS = 48
SOFT_SPLIT_CHARS = 24
SPLIT_LINE_GAP_PTS = 2.0
SPLIT_MIN_LINE_PTS = 14.0
CHECK_EMPTY = frozenset("☐")  # U+2610
CHECK_MARKED = frozenset("☒☑✓✔")  # U+2612, U+2611, U+2713, U+2714
CHECK_ASCII_X = frozenset({"X", "x"})
CHECK_MATCH_DIST_PTS = 8.0
CHECK_BESIDE_GAP_PTS = 4.0
COLUMN_GAP_PTS = 8.0
COLUMN_PAGE_MARGIN_PTS = 18.0


@dataclass
class FillCell:
    id: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 PDF points
    text: str
    enabled: bool = True
    kind: str = "text"  # "text" | "check"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


def sanitize_text(text: str) -> str:
    replacements = {
        "×": "x",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "°": " ",
        "µ": "u",
        "μ": "u",
        "\n": " ",
        "\t": " ",
        "<": "",
        ">": "",
    }
    # 2.21 x 10^13 / 2.21×10¹³ style → ASCII
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*10\s*[\^¹]?[\s]*([0-9¹²³⁴⁵⁶⁷⁸⁹⁰]+)",
        lambda m: f"{m.group(1)} x 10e{_sup_to_ascii(m.group(2))}",
        text,
    )
    text = text.replace("^", "")
    # Strip thousands separators so Emuru sees 10000 not 10,000
    text = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)
    out = []
    for ch in text.strip():
        ch = replacements.get(ch, ch)
        if ch in EMURU_ALLOWED:
            out.append(ch)
    cleaned = "".join(out)
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(
        r"(?i)\b(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2,4})\b",
        lambda m: f"{m.group(1)} {m.group(2).upper()} {m.group(3)}",
        cleaned,
    )
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


def _sup_to_ascii(s: str) -> str:
    table = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    return s.translate(table)


def _norm(s: str) -> str:
    return " ".join(sanitize_text(s).lower().split())


@dataclass
class Span:
    text: str
    rect: fitz.Rect
    color: Optional[Tuple[float, float, float]] = None  # 0..1 RGB if available


def _extract_spans(page: fitz.Page) -> List[Span]:
    spans: List[Span] = []
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                color = span.get("color")
                rgb = None
                if isinstance(color, int):
                    r = ((color >> 16) & 255) / 255.0
                    g = ((color >> 8) & 255) / 255.0
                    b = (color & 255) / 255.0
                    rgb = (r, g, b)
                spans.append(Span(text=text, rect=fitz.Rect(span["bbox"]), color=rgb))
    return spans


def _iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    inter_area = abs(inter.width * inter.height)
    union_area = abs(a.width * a.height) + abs(b.width * b.height) - inter_area
    return inter_area / max(union_area, 1e-6)


def _center(r: fitz.Rect) -> Tuple[float, float]:
    return ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)


def _is_blue(color: Optional[Tuple[float, float, float]]) -> bool:
    if not color:
        return False
    r, g, b = color
    return b > 0.35 and b > r + 0.08 and b > g + 0.05


def _is_near_black(color: Optional[Tuple[float, float, float]]) -> bool:
    if not color:
        return True  # missing color → treat as printable label
    r, g, b = color
    return r < 0.35 and g < 0.35 and b < 0.35 and not _is_blue(color)


def _template_has_match(synth: Span, template_spans: List[Span]) -> bool:
    """True if template already has essentially the same text at this place."""
    sn = _norm(synth.text)
    if not sn:
        return True
    scx, scy = _center(synth.rect)
    for t in template_spans:
        tn = _norm(t.text)
        if not tn:
            continue
        # same normalized text near same location, or high IoU
        tcx, tcy = _center(t.rect)
        dist = ((tcx - scx) ** 2 + (tcy - scy) ** 2) ** 0.5
        if sn == tn and dist < CENTER_DIST_PTS * 2:
            return True
        if _iou(synth.rect, t.rect) >= IOU_MATCH and (sn in tn or tn in sn or sn == tn):
            return True
        # overlapping geometry with any template ink text (label already printed)
        if _iou(synth.rect, t.rect) >= 0.35:
            return True
    return False


def _is_fill_span(sp: Span, template_spans: List[Span]) -> bool:
    """True if span is a typed fill: blue ink, or text not already on the template."""
    if not (sp.text or "").strip():
        return False
    if _is_blue(sp.color):
        return True
    return not _template_has_match(sp, template_spans)


def _merge_span_groups(spans: List[Span]) -> List[Tuple[fitz.Rect, str]]:
    """Merge adjacent spans (same line or wrapped continuation) into cells."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (round(s.rect.y0, 1), s.rect.x0))
    groups: List[List[Span]] = []
    for sp in spans:
        placed = False
        for g in groups:
            last = g[-1]
            same_line = abs(sp.rect.y0 - last.rect.y0) < 4 and abs(sp.rect.y1 - last.rect.y1) < 6
            close_x = sp.rect.x0 <= last.rect.x1 + MERGE_GAP_PTS
            # wrapped continuation under previous span
            stacked = (
                sp.rect.y0 <= last.rect.y1 + 6
                and sp.rect.y0 >= last.rect.y0 - 2
                and not (sp.rect.x1 < last.rect.x0 - 4 or sp.rect.x0 > last.rect.x1 + 4)
            )
            if (same_line and close_x) or stacked:
                g.append(sp)
                placed = True
                break
        if not placed:
            groups.append([sp])

    out: List[Tuple[fitz.Rect, str]] = []
    for g in groups:
        g = sorted(g, key=lambda s: (s.rect.y0, s.rect.x0))
        rect = g[0].rect
        for sp in g[1:]:
            rect |= sp.rect
        text = sanitize_text(" ".join(sp.text for sp in g))
        if text:
            pad = 1.0
            rect = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
            out.append((rect, text))
    return out


def _find_left_anchor(
    fill: fitz.Rect,
    synth_spans: List[Span],
    template_spans: List[Span],
) -> Optional[Span]:
    """Rightmost non-fill span to the left of fill on roughly the same row."""
    candidates: List[Span] = []
    for sp in synth_spans:
        if _is_fill_span(sp, template_spans):
            continue
        if abs(sp.rect.y0 - fill.y0) > ANCHOR_Y_TOL_PTS:
            continue
        if sp.rect.x1 >= fill.x0 - ANCHOR_X_GAP_PTS:
            continue
        if not (sp.text or "").strip():
            continue
        candidates.append(sp)
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.rect.x1)


def _find_column_header_anchor(
    fill: fitz.Rect,
    synth_spans: List[Span],
    template_spans: List[Span],
) -> Optional[Span]:
    """Nearest non-fill span above the fill whose x-range covers the fill center."""
    cx = (float(fill.x0) + float(fill.x1)) / 2.0
    candidates: List[Span] = []
    for sp in synth_spans:
        if _is_fill_span(sp, template_spans):
            continue
        if not (sp.text or "").strip():
            continue
        if sp.rect.y1 > fill.y0 + HEADER_ANCHOR_SLACK_PTS:
            continue
        gap = float(fill.y0) - float(sp.rect.y1)
        if gap > HEADER_ANCHOR_MAX_GAP_PTS:
            continue
        if not (sp.rect.x0 <= cx <= sp.rect.x1):
            continue
        candidates.append(sp)
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.rect.y0)


def _find_fill_anchor(
    fill: fitz.Rect,
    synth_spans: List[Span],
    template_spans: List[Span],
) -> Optional[Span]:
    """Prefer a same-row left label; fall back to the column header above."""
    left = _find_left_anchor(fill, synth_spans, template_spans)
    if left is not None and _norm(left.text):
        return left
    header = _find_column_header_anchor(fill, synth_spans, template_spans)
    if header is not None and _norm(header.text):
        return header
    return None


def _label_instances(
    spans: List[Span],
    key: str,
    template_spans: List[Span],
    *,
    near_black_only: bool,
) -> List[Span]:
    """Occurrences of normalized label key, sorted top-to-bottom, position-deduped."""
    out: List[Span] = []
    for sp in spans:
        if _is_fill_span(sp, template_spans):
            continue
        if near_black_only and not _is_near_black(sp.color):
            continue
        if _norm(sp.text) != key:
            continue
        out.append(sp)
    out.sort(key=lambda s: (s.rect.y0, s.rect.x0))
    deduped: List[Span] = []
    for sp in out:
        if deduped and abs(deduped[-1].rect.y0 - sp.rect.y0) < 2 and abs(
            deduped[-1].rect.x0 - sp.rect.x0
        ) < 2:
            continue
        deduped.append(sp)
    return deduped


def _instance_index(anchor: Span, instances: List[Span]) -> Optional[int]:
    if not instances:
        return None
    for i, sp in enumerate(instances):
        if (
            abs(sp.rect.y0 - anchor.rect.y0) < ANCHOR_MATCH_TOL_PTS
            and abs(sp.rect.x0 - anchor.rect.x0) < ANCHOR_MATCH_TOL_PTS
        ):
            return i
    # Closest by y (then x)
    return min(
        range(len(instances)),
        key=lambda i: (
            abs(instances[i].rect.y0 - anchor.rect.y0),
            abs(instances[i].rect.x0 - anchor.rect.x0),
        ),
    )


def _field_rect(
    mapped: fitz.Rect,
    t_label: Span,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    """Expand mapped glyph box into a stamp field rect on the template."""
    h = max(float(mapped.height), float(t_label.rect.height) * 1.15)
    cy = (mapped.y0 + mapped.y1) / 2.0
    y0 = cy - h / 2.0
    y1 = cy + h / 2.0
    x0 = max(float(page_rect.x0), float(mapped.x0))
    x1 = min(float(page_rect.x1), float(mapped.x1))
    if x1 - x0 < 2:
        x1 = x0 + max(2.0, float(mapped.width))
    return fitz.Rect(x0, y0, x1, y1)


def _snap_header_field_y(
    field: fitz.Rect,
    header: fitz.Rect,
    horiz_rules: Sequence[HorizontalRule],
) -> fitz.Rect:
    """Fit a header-anchored field into the template row under the header."""
    row = row_below_header(horiz_rules, header)
    if row is None:
        return field
    y0 = row[0] + TABLE_RULE_INSET_PTS
    y1 = row[1] - TABLE_RULE_INSET_PTS
    if y1 - y0 < 2.0:
        return field
    return fitz.Rect(field.x0, y0, field.x1, y1)


def _map_fill_to_template_field(
    fill: fitz.Rect,
    synth_spans: List[Span],
    template_spans: List[Span],
    page_rect: fitz.Rect,
    horiz_rules: Optional[Sequence[HorizontalRule]] = None,
) -> Optional[fitz.Rect]:
    """
    Pair fill to the Nth template occurrence of its row label or column header.
    Returns field rect on the template, or None to skip (no anchor / overflow).
    """
    left = _find_left_anchor(fill, synth_spans, template_spans)
    anchor = _find_fill_anchor(fill, synth_spans, template_spans)
    if anchor is None:
        return None
    key = _norm(anchor.text)
    if not key:
        return None

    synth_inst = _label_instances(
        synth_spans, key, template_spans, near_black_only=False
    )
    tmpl_inst = _label_instances(
        template_spans, key, template_spans, near_black_only=True
    )
    if not tmpl_inst:
        tmpl_inst = _label_instances(
            template_spans, key, template_spans, near_black_only=False
        )
    if not synth_inst or not tmpl_inst:
        return None

    instance_i = _instance_index(anchor, synth_inst)
    if instance_i is None or instance_i >= len(tmpl_inst):
        return None

    s_label = synth_inst[instance_i]
    t_label = tmpl_inst[instance_i]
    dx = t_label.rect.x0 - s_label.rect.x0
    dy = t_label.rect.y0 - s_label.rect.y0
    mapped = fitz.Rect(fill.x0 + dx, fill.y0 + dy, fill.x1 + dx, fill.y1 + dy)
    field = _field_rect(mapped, t_label, page_rect)
    header_anchored = left is None
    if header_anchored and horiz_rules:
        field = _snap_header_field_y(field, t_label.rect, horiz_rules)
    return field


def _dedupe_cells(cells: List[FillCell]) -> List[FillCell]:
    seen = set()
    out: List[FillCell] = []
    for c in cells:
        key = (
            c.page,
            round(c.bbox[0]),
            round(c.bbox[1]),
            round(c.bbox[2]),
            round(c.bbox[3]),
            c.text,
            c.kind,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _same_row(a: FillCell, b: FillCell) -> bool:
    """True if two cells sit on the same table row."""
    _, ay0, _, ay1 = a.bbox
    _, by0, _, by1 = b.bbox
    acy = (ay0 + ay1) / 2.0
    bcy = (by0 + by1) / 2.0
    if abs(acy - bcy) <= ANCHOR_Y_TOL_PTS:
        return True
    return not (ay1 <= by0 or by1 <= ay0)


def _expand_text_fields_to_columns(
    cells: List[FillCell],
    page_rects: Dict[int, fitz.Rect],
    page_rules: Optional[Dict[int, List[VerticalRule]]] = None,
) -> None:
    """Fit text-cell x1 to the next fill or vector table rule on its row.

    Check cells are left unchanged so the vector X stays on the printed ☐.
    """
    rules_by_page = page_rules or {}
    by_page: Dict[int, List[FillCell]] = {}
    for c in cells:
        by_page.setdefault(c.page, []).append(c)

    for page_i, page_cells in by_page.items():
        page_rect = page_rects.get(page_i)
        if page_rect is None:
            continue
        page_limit = float(page_rect.x1) - COLUMN_PAGE_MARGIN_PTS
        for cell in page_cells:
            if cell.kind == "check":
                continue
            x0, y0, x1, y1 = cell.bbox
            next_x0: Optional[float] = None
            for other in page_cells:
                ox0 = other.bbox[0]
                if ox0 <= x0 + 0.5:
                    continue
                if not _same_row(cell, other):
                    continue
                if next_x0 is None or ox0 < next_x0:
                    next_x0 = ox0

            rule_x = nearest_right_rule(
                rules_by_page.get(page_i, ()),
                x0=x0,
                y0=y0,
                y1=y1,
            )
            limits = [page_limit]
            if next_x0 is not None:
                limits.append(next_x0 - COLUMN_GAP_PTS)
            if rule_x is not None:
                limits.append(rule_x - TABLE_RULE_INSET_PTS)
            elif next_x0 is None:
                # An unbounded last column must not be grown toward the page edge.
                limits.append(x1)

            limit = min(limits)
            if limit - x0 < 2.0:
                continue
            new_x1 = max(x0 + 2.0, limit)
            if abs(new_x1 - x1) > 0.5:
                cell.bbox = (x0, y0, new_x1, y1)


def _split_text_chunks(text: str, max_len: int = SOFT_SPLIT_CHARS) -> List[str]:
    """Split on spaces and '/' into chunks of length <= max_len."""
    text = text.strip()
    if len(text) <= max_len:
        # Split slash-separated compounds (molar ratios, DNA/PEI blobs) when multi-part
        if "/" in text and len(text) >= 12:
            parts = [p.strip() for p in text.split("/") if p.strip()]
            if len(parts) > 1 and all(len(p) <= max_len for p in parts):
                return parts
        return [text]
    # Prefer slash boundaries for molar-ratio style strings
    if "/" in text:
        slash_parts = [p.strip() for p in text.split("/") if p.strip()]
        if len(slash_parts) > 1 and all(len(p) <= max_len for p in slash_parts):
            return slash_parts
    words = text.replace("/", " / ").split(" ")
    chunks: List[str] = []
    cur = ""
    for w in words:
        if not w:
            continue
        if len(w) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(w), max_len):
                chunks.append(w[i : i + max_len])
            continue
        trial = w if not cur else f"{cur} {w}"
        if len(trial) <= max_len:
            cur = trial
        else:
            if cur:
                chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_len]]


def _emit_cells_for_text(
    page_i: int,
    rect: fitz.Rect,
    text: str,
    cell_i_start: int,
) -> Tuple[List[FillCell], int]:
    """Apply size limits: skip >MAX, soft-split 24..48 when height allows."""
    cells: List[FillCell] = []
    cell_i = cell_i_start
    if not text or rect.width < 2 or rect.height < 2:
        return cells, cell_i

    if len(text) > MAX_CELL_CHARS:
        return cells, cell_i

    if len(text) <= SOFT_SPLIT_CHARS:
        # Slash-separated compounds (e.g. molar ratios) still soft-split when multi-part
        if "/" in text and len(text) >= 12:
            slash_chunks = _split_text_chunks(text, SOFT_SPLIT_CHARS)
            if len(slash_chunks) > 1:
                chunks = slash_chunks
                n = len(chunks)
                min_h = n * SPLIT_MIN_LINE_PTS + (n - 1) * SPLIT_LINE_GAP_PTS
                if rect.height >= min_h:
                    line_h = (rect.height - (n - 1) * SPLIT_LINE_GAP_PTS) / n
                    for i, chunk in enumerate(chunks):
                        y0 = rect.y0 + i * (line_h + SPLIT_LINE_GAP_PTS)
                        y1 = y0 + line_h
                        cells.append(
                            FillCell(
                                id=f"p{page_i}_c{cell_i}",
                                page=page_i,
                                bbox=(
                                    float(rect.x0),
                                    float(y0),
                                    float(rect.x1),
                                    float(y1),
                                ),
                                text=chunk,
                                enabled=True,
                            )
                        )
                        cell_i += 1
                    return cells, cell_i
        cells.append(
            FillCell(
                id=f"p{page_i}_c{cell_i}",
                page=page_i,
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                text=text,
                enabled=True,
            )
        )
        return cells, cell_i + 1

    chunks = _split_text_chunks(text, SOFT_SPLIT_CHARS)
    n = len(chunks)
    # Need enough height to stack lines (~14pt each + gaps)
    min_h = n * SPLIT_MIN_LINE_PTS + (n - 1) * SPLIT_LINE_GAP_PTS
    if rect.height < min_h or n == 1:
        # Truncate to soft limit as a single cell
        cells.append(
            FillCell(
                id=f"p{page_i}_c{cell_i}",
                page=page_i,
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                text=text[:SOFT_SPLIT_CHARS].rstrip(),
                enabled=True,
            )
        )
        return cells, cell_i + 1

    line_h = (rect.height - (n - 1) * SPLIT_LINE_GAP_PTS) / n
    for i, chunk in enumerate(chunks):
        y0 = rect.y0 + i * (line_h + SPLIT_LINE_GAP_PTS)
        y1 = y0 + line_h
        cells.append(
            FillCell(
                id=f"p{page_i}_c{cell_i}",
                page=page_i,
                bbox=(float(rect.x0), float(y0), float(rect.x1), float(y1)),
                text=chunk,
                enabled=True,
            )
        )
        cell_i += 1
    return cells, cell_i


def _box_kind(text: str) -> Optional[str]:
    t = (text or "").strip()
    if t in CHECK_EMPTY:
        return "empty"
    if t in CHECK_MARKED:
        return "checked"
    return None


def _is_check_mark_span(sp: Span) -> bool:
    t = (sp.text or "").strip()
    return t in CHECK_MARKED or t in CHECK_ASCII_X


def _near_box(a: fitz.Rect, b: fitz.Rect) -> bool:
    if _iou(a, b) >= 0.35:
        return True
    acx, acy = _center(a)
    bcx, bcy = _center(b)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 <= CHECK_MATCH_DIST_PTS


def _beside_right(mark: fitz.Rect, box: fitz.Rect) -> bool:
    """True if mark sits just to the right of box on the same row."""
    _, mcy = _center(mark)
    _, bcy = _center(box)
    if abs(mcy - bcy) > CHECK_MATCH_DIST_PTS:
        return False
    gap = mark.x0 - box.x1
    return -1.0 <= gap <= CHECK_BESIDE_GAP_PTS


def _mark_pairs_to_box(mark: fitz.Rect, box: fitz.Rect) -> bool:
    return _near_box(mark, box) or _beside_right(mark, box)


def _box_already_marked(box: fitz.Rect, template_spans: List[Span]) -> bool:
    """True if the template already has X/☒ next to this empty box."""
    for sp in template_spans:
        if not _is_check_mark_span(sp):
            continue
        if _mark_pairs_to_box(sp.rect, box):
            return True
    return False


def _clamp_box(box: fitz.Rect, page_rect: fitz.Rect) -> Optional[fitz.Rect]:
    x0 = max(float(page_rect.x0), float(box.x0))
    y0 = max(float(page_rect.y0), float(box.y0))
    x1 = min(float(page_rect.x1), float(box.x1))
    y1 = min(float(page_rect.y1), float(box.y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def _detect_check_fills(
    synth_spans: List[Span],
    template_spans: List[Span],
    page_rect: fitz.Rect,
) -> Tuple[List[fitz.Rect], set]:
    """Pair ☒-on-☐ or X-beside-☐ to template empty boxes.

    Returns (template box rects, ids of consumed synth spans).
    Pre-marked template boxes are skipped; those synth marks are still consumed
    so they do not become text fills.
    """
    empty_boxes = [sp for sp in template_spans if _box_kind(sp.text) == "empty"]
    used = set()
    consumed: set = set()
    out: List[fitz.Rect] = []
    for sp in synth_spans:
        if not _is_check_mark_span(sp):
            continue
        candidates: List[int] = []
        for i, t in enumerate(empty_boxes):
            if i in used:
                continue
            if _mark_pairs_to_box(sp.rect, t.rect):
                candidates.append(i)
        if not candidates:
            continue
        unmarked = [
            i for i in candidates if not _box_already_marked(empty_boxes[i].rect, template_spans)
        ]
        if not unmarked:
            consumed.add(id(sp))
            continue
        scx, scy = _center(sp.rect)

        def _score(i: int) -> Tuple[float, float]:
            box = empty_boxes[i].rect
            tcx, tcy = _center(box)
            dist = ((tcx - scx) ** 2 + (tcy - scy) ** 2) ** 0.5
            return (-_iou(sp.rect, box), dist)

        best_i = min(unmarked, key=_score)
        used.add(best_i)
        consumed.add(id(sp))
        clamped = _clamp_box(empty_boxes[best_i].rect, page_rect)
        if clamped is not None:
            out.append(clamped)
    return out, consumed


def _render_page(doc: fitz.Document, page_index: int, dpi: int = DPI):
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img, zoom


def detect_fill_cells(
    template_pdf: Path,
    synthetic_pdf: Path,
    debug_dir: Optional[Path] = None,
) -> List[FillCell]:
    tdoc = fitz.open(template_pdf)
    sdoc = fitz.open(synthetic_pdf)
    # Synthetic may have extra overflow pages from filled text; only diff
    # pages that exist on both docs (stamp targets the template).
    shared_pages = min(len(tdoc), len(sdoc))

    cells: List[FillCell] = []
    cell_i = 0
    page_rects: Dict[int, fitz.Rect] = {}
    page_rules: Dict[int, List[VerticalRule]] = {}
    page_horiz: Dict[int, List[HorizontalRule]] = {}

    for page_i in range(shared_pages):
        spage = sdoc[page_i]
        tpage = tdoc[page_i]
        s_spans = _extract_spans(spage)
        t_spans = _extract_spans(tpage)
        page_rect = tpage.rect
        page_rects[page_i] = fitz.Rect(page_rect)
        page_rules[page_i] = extract_vertical_rules(tpage)
        page_horiz[page_i] = extract_horizontal_rules(tpage)

        check_rects, consumed_marks = _detect_check_fills(s_spans, t_spans, page_rect)

        # Typed fills: blue ink, or synthetic text not already on the template.
        # Check-mark spans (☒ or lone X beside ☐) are handled above.
        fills = [
            sp
            for sp in s_spans
            if id(sp) not in consumed_marks and _is_fill_span(sp, t_spans)
        ]

        merged = _merge_span_groups(fills)

        remapped: List[Tuple[fitz.Rect, str]] = []
        skipped_no_anchor = 0
        skipped_overflow = 0
        for rect, text in merged:
            if rect.width < 2 or rect.height < 2:
                continue
            anchor = _find_fill_anchor(rect, s_spans, t_spans)
            if anchor is None or not _norm(anchor.text):
                skipped_no_anchor += 1
                continue
            mapped = _map_fill_to_template_field(
                rect,
                s_spans,
                t_spans,
                page_rect,
                horiz_rules=page_horiz[page_i],
            )
            if mapped is None:
                skipped_overflow += 1
                continue
            remapped.append((mapped, text))

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            simg, zoom = _render_page(sdoc, page_i)
            dbg = simg.copy()
            for rect, _text in merged:
                x0, y0, x1, y1 = [int(v * zoom) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 180, 0), 2)
            cv2.imwrite(
                str(debug_dir / f"page_{page_i}_fills.png"),
                cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR),
            )
            timg, tzoom = _render_page(tdoc, page_i)
            tdbg = timg.copy()
            for rect, _text in remapped:
                x0, y0, x1, y1 = [int(v * tzoom) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]
                cv2.rectangle(tdbg, (x0, y0), (x1, y1), (37, 99, 235), 2)
            for rect in check_rects:
                x0, y0, x1, y1 = [int(v * tzoom) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]
                cv2.rectangle(tdbg, (x0, y0), (x1, y1), (37, 99, 235), 2)
            cv2.imwrite(
                str(debug_dir / f"page_{page_i}_template_remap.png"),
                cv2.cvtColor(tdbg, cv2.COLOR_RGB2BGR),
            )
            skip_path = debug_dir / f"page_{page_i}_skip.json"
            skip_path.write_text(
                json.dumps(
                    {
                        "page": page_i,
                        "merged": len(merged),
                        "kept": len(remapped),
                        "checks": len(check_rects),
                        "skipped_no_anchor": skipped_no_anchor,
                        "skipped_overflow": skipped_overflow,
                    },
                    indent=2,
                )
                + "\n"
            )

        for rect, text in remapped:
            new_cells, cell_i = _emit_cells_for_text(page_i, rect, text, cell_i)
            cells.extend(new_cells)

        for rect in check_rects:
            cells.append(
                FillCell(
                    id=f"p{page_i}_c{cell_i}",
                    page=page_i,
                    bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                    text="X",
                    enabled=True,
                    kind="check",
                )
            )
            cell_i += 1

    tdoc.close()
    sdoc.close()

    cells = _dedupe_cells(cells)
    _expand_text_fields_to_columns(cells, page_rects, page_rules)
    # Re-id after dedupe for stable sequential ids
    for i, c in enumerate(cells):
        c.id = f"p{c.page}_c{i}"

    if not cells:
        raise ValueError(
            "No fill cells detected. Ensure the filled PDF has typed values "
            "that are not already on the template."
        )
    return cells


def render_overlay_png(template_pdf: Path, cells: List[FillCell], page: int = 0) -> bytes:
    doc = fitz.open(template_pdf)
    img, zoom = _render_page(doc, page)
    doc.close()
    canvas = img.copy()
    for c in cells:
        if c.page != page:
            continue
        x0, y0, x1, y1 = [int(v * zoom) for v in c.bbox]
        color = (37, 99, 235) if c.enabled else (160, 160, 160)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return buf.tobytes()
