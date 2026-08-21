"""Extract conservative table boundaries from vector PDF drawings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import fitz


VERTICAL_TOLERANCE_PTS = 0.75
MIN_RULE_LENGTH_PTS = 8.0
MIN_ROW_OVERLAP_FRAC = 0.60
RULE_SEARCH_EPSILON_PTS = 2.0
TABLE_RULE_INSET_PTS = 1.0


@dataclass(frozen=True)
class VerticalRule:
    x: float
    y0: float
    y1: float


@dataclass(frozen=True)
class HorizontalRule:
    y: float
    x0: float
    x1: float


def _vertical_rule(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Optional[VerticalRule]:
    if abs(x1 - x0) > VERTICAL_TOLERANCE_PTS:
        return None
    top, bottom = sorted((float(y0), float(y1)))
    if bottom - top < MIN_RULE_LENGTH_PTS:
        return None
    return VerticalRule(x=(float(x0) + float(x1)) / 2.0, y0=top, y1=bottom)


def _horizontal_rule(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Optional[HorizontalRule]:
    if abs(y1 - y0) > VERTICAL_TOLERANCE_PTS:
        return None
    left, right = sorted((float(x0), float(x1)))
    if right - left < MIN_RULE_LENGTH_PTS:
        return None
    return HorizontalRule(y=(float(y0) + float(y1)) / 2.0, x0=left, x1=right)


def _vertical_rules_from_item(item: Sequence[object]) -> Iterable[VerticalRule]:
    if not item:
        return ()
    kind = item[0]
    if kind == "l" and len(item) >= 3:
        p0, p1 = item[1], item[2]
        rule = _vertical_rule(p0.x, p0.y, p1.x, p1.y)
        return (rule,) if rule is not None else ()
    if kind == "re" and len(item) >= 2:
        rect = fitz.Rect(item[1])
        rules = (
            _vertical_rule(rect.x0, rect.y0, rect.x0, rect.y1),
            _vertical_rule(rect.x1, rect.y0, rect.x1, rect.y1),
        )
        return tuple(rule for rule in rules if rule is not None)
    return ()


def _horizontal_rules_from_item(item: Sequence[object]) -> Iterable[HorizontalRule]:
    if not item:
        return ()
    kind = item[0]
    if kind == "l" and len(item) >= 3:
        p0, p1 = item[1], item[2]
        rule = _horizontal_rule(p0.x, p0.y, p1.x, p1.y)
        return (rule,) if rule is not None else ()
    if kind == "re" and len(item) >= 2:
        rect = fitz.Rect(item[1])
        rules = (
            _horizontal_rule(rect.x0, rect.y0, rect.x1, rect.y0),
            _horizontal_rule(rect.x0, rect.y1, rect.x1, rect.y1),
        )
        return tuple(rule for rule in rules if rule is not None)
    return ()


def extract_vertical_rules(page: fitz.Page) -> List[VerticalRule]:
    """Return de-duplicated vertical line/rectangle edges on a PDF page."""
    raw: List[VerticalRule] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            raw.extend(_vertical_rules_from_item(item))

    raw.sort(key=lambda rule: (rule.x, rule.y0, rule.y1))
    deduped: List[VerticalRule] = []
    for rule in raw:
        if deduped:
            previous = deduped[-1]
            if (
                abs(previous.x - rule.x) <= VERTICAL_TOLERANCE_PTS
                and abs(previous.y0 - rule.y0) <= VERTICAL_TOLERANCE_PTS
                and abs(previous.y1 - rule.y1) <= VERTICAL_TOLERANCE_PTS
            ):
                continue
        deduped.append(rule)
    return deduped


def nearest_right_rule(
    rules: Sequence[VerticalRule],
    *,
    x0: float,
    y0: float,
    y1: float,
) -> Optional[float]:
    """Find the nearest trusted vertical rule to the right of a row field."""
    row_top, row_bottom = sorted((float(y0), float(y1)))
    row_height = max(1.0, row_bottom - row_top)
    candidates: List[float] = []
    for rule in rules:
        if rule.x <= float(x0) + RULE_SEARCH_EPSILON_PTS:
            continue
        overlap = max(
            0.0,
            min(row_bottom, rule.y1) - max(row_top, rule.y0),
        )
        if overlap / row_height < MIN_ROW_OVERLAP_FRAC:
            continue
        candidates.append(rule.x)
    return min(candidates) if candidates else None


def extract_horizontal_rules(page: fitz.Page) -> List[HorizontalRule]:
    """Return de-duplicated horizontal line/rectangle edges on a PDF page."""
    raw: List[HorizontalRule] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            raw.extend(_horizontal_rules_from_item(item))

    raw.sort(key=lambda rule: (rule.y, rule.x0, rule.x1))
    deduped: List[HorizontalRule] = []
    for rule in raw:
        if deduped:
            previous = deduped[-1]
            if (
                abs(previous.y - rule.y) <= VERTICAL_TOLERANCE_PTS
                and abs(previous.x0 - rule.x0) <= VERTICAL_TOLERANCE_PTS
                and abs(previous.x1 - rule.x1) <= VERTICAL_TOLERANCE_PTS
            ):
                continue
        deduped.append(rule)
    return deduped


def _rule_overlaps_header(rule: HorizontalRule, header: fitz.Rect) -> bool:
    overlap = max(0.0, min(rule.x1, float(header.x1)) - max(rule.x0, float(header.x0)))
    header_width = max(1.0, float(header.x1) - float(header.x0))
    return overlap / header_width >= MIN_ROW_OVERLAP_FRAC


def enclosing_row(
    rules: Sequence[HorizontalRule],
    *,
    x0: float,
    x1: float,
    y: float,
) -> Optional[Tuple[float, float]]:
    """Return the horizontal-rule row containing a field anchor."""
    left, right = sorted((float(x0), float(x1)))
    width = max(1.0, right - left)
    overlapping = [
        rule
        for rule in rules
        if max(0.0, min(rule.x1, right) - max(rule.x0, left)) / width
        >= MIN_ROW_OVERLAP_FRAC
    ]
    top = max(
        (
            rule.y
            for rule in overlapping
            if rule.y < float(y) - VERTICAL_TOLERANCE_PTS
        ),
        default=None,
    )
    bottom = min(
        (
            rule.y
            for rule in overlapping
            if rule.y > float(y) + VERTICAL_TOLERANCE_PTS
        ),
        default=None,
    )
    if top is None or bottom is None or bottom - top < 2.0:
        return None
    return (top, bottom)


def row_below_header(
    rules: Sequence[HorizontalRule],
    header: fitz.Rect,
) -> Optional[Tuple[float, float]]:
    """Return (y0, y1) of the data row immediately under a column header."""
    overlapping = [rule for rule in rules if _rule_overlaps_header(rule, header)]
    overlapping.sort(key=lambda rule: rule.y)
    top: Optional[HorizontalRule] = None
    for rule in overlapping:
        if rule.y + RULE_SEARCH_EPSILON_PTS < float(header.y1):
            continue
        top = rule
        break
    if top is None:
        return None
    bottom: Optional[HorizontalRule] = None
    for rule in overlapping:
        if rule.y <= top.y + VERTICAL_TOLERANCE_PTS:
            continue
        bottom = rule
        break
    if bottom is None:
        return None
    y0, y1 = top.y, bottom.y
    if y1 - y0 < 2.0:
        return None
    return (y0, y1)
