from __future__ import annotations

import fitz

from cell_diff import (
    HEADER_ANCHOR_MAX_GAP_PTS,
    Span,
    _find_column_header_anchor,
    _find_fill_anchor,
    _find_left_anchor,
    _map_fill_to_template_field,
)
from table_geometry import TABLE_RULE_INSET_PTS, HorizontalRule


BLUE = (0.04, 0.17, 0.55)
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
PAGE = fitz.Rect(0, 0, 612, 792)


def _span(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color=BLACK,
) -> Span:
    return Span(text=text, rect=fitz.Rect(x0, y0, x1, y1), color=color)


def test_signoff_uses_column_headers_when_left_row_is_empty() -> None:
    headers = [
        _span("Performed By / Date", 59.5, 88.2, 145.5, 98.2, WHITE),
        _span("Reviewed By / Date", 227.5, 88.2, 310.0, 98.2, WHITE),
        _span("QA Approved By / Date", 395.5, 88.2, 494.4, 98.2, WHITE),
    ]
    fills = [
        (_span("ST 24JUL26", 59.5, 110.4, 120.6, 122.7, BLUE), "Performed By / Date"),
        (_span("UV 24JUL26", 227.5, 110.4, 289.8, 122.7, BLUE), "Reviewed By / Date"),
        (
            _span("S. Lindqvist 12AUG2026", 395.5, 110.4, 515.9, 122.7, BLUE),
            "QA Approved By / Date",
        ),
    ]
    template = [_span("SIGN-OFF", 59.5, 64.9, 107.2, 76.0, WHITE), *headers]
    synth = [*template, *(fill for fill, _ in fills)]

    for fill, expected in fills:
        assert _find_left_anchor(fill.rect, synth, template) is None
        anchor = _find_fill_anchor(fill.rect, synth, template)
        assert anchor is not None
        assert anchor.text == expected
        mapped = _map_fill_to_template_field(fill.rect, synth, template, PAGE)
        assert mapped is not None
        assert abs(mapped.y0 - fill.rect.y0) < 3.0


def test_approvals_remap_applies_header_page_shift() -> None:
    synth_header = _span(
        "Manufacturing Review (Signature / Date)",
        59.5,
        424.6,
        222.0,
        434.6,
        BLACK,
    )
    tmpl_header = _span(
        "Manufacturing Review (Signature / Date)",
        59.5,
        413.3,
        222.0,
        423.3,
        BLACK,
    )
    section = _span("BATCH RECORD APPROVALS", 59.5, 401.2, 205.5, 412.4, WHITE)
    fill = _span("R. Okafor  23JUL2026", 59.5, 446.8, 168.9, 459.1, BLUE)
    qa = _span(
        "Quality Assurance Review (Signature / Date)",
        311.5,
        424.6,
        490.0,
        434.6,
        BLACK,
    )
    synth = [section, synth_header, qa, fill]
    template = [
        _span("BATCH RECORD APPROVALS", 59.5, 389.9, 205.5, 401.1, WHITE),
        tmpl_header,
        _span(
            "Quality Assurance Review (Signature / Date)",
            311.5,
            413.3,
            490.0,
            423.3,
            BLACK,
        ),
    ]

    assert _find_left_anchor(fill.rect, synth, template) is None
    anchor = _find_column_header_anchor(fill.rect, synth, template)
    assert anchor is not None
    assert anchor.text == "Manufacturing Review (Signature / Date)"

    mapped = _map_fill_to_template_field(fill.rect, synth, template, PAGE)
    assert mapped is not None
    expected_y0 = fill.rect.y0 + (tmpl_header.rect.y0 - synth_header.rect.y0)
    assert abs(mapped.y0 - expected_y0) < 3.0
    assert mapped.y0 < fill.rect.y0 - 8.0


def test_same_row_left_label_wins_over_header_above() -> None:
    header = _span("Performed By / Date", 395.5, 248.0, 481.5, 258.0, WHITE)
    label = _span("Endotoxin", 227.4, 265.0, 269.9, 275.0, BLACK)
    fill = _span("JR 21JUL26", 395.5, 270.2, 445.0, 280.2, BLUE)
    synth = [header, label, fill]
    template = [header, label]

    left = _find_left_anchor(fill.rect, synth, template)
    assert left is not None
    assert left.text == "Endotoxin"
    chosen = _find_fill_anchor(fill.rect, synth, template)
    assert chosen is not None
    assert chosen.text == "Endotoxin"


def test_right_column_fill_does_not_use_left_column_header() -> None:
    left_header = _span(
        "Manufacturing Review (Signature / Date)",
        59.5,
        424.6,
        222.0,
        434.6,
        BLACK,
    )
    fill = _span("S. Lindqvist  12AUG2026", 311.5, 446.8, 434.9, 459.1, BLUE)
    synth = [left_header, fill]
    template = [left_header]

    assert _find_column_header_anchor(fill.rect, synth, template) is None
    assert _find_fill_anchor(fill.rect, synth, template) is None
    assert _map_fill_to_template_field(fill.rect, synth, template, PAGE) is None


def test_approvals_snaps_into_short_template_row() -> None:
    synth_header = _span(
        "Manufacturing Review (Signature / Date)",
        59.5,
        424.6,
        222.0,
        434.6,
        BLACK,
    )
    tmpl_header = _span(
        "Manufacturing Review (Signature / Date)",
        59.5,
        413.3,
        222.0,
        423.3,
        BLACK,
    )
    fill = _span("R. Okafor  23JUL2026", 59.5, 446.8, 168.9, 459.1, BLUE)
    synth = [synth_header, fill]
    template = [tmpl_header]
    rules = [
        HorizontalRule(y=402.6, x0=54.2, x1=557.8),
        HorizontalRule(y=424.9, x0=54.2, x1=557.8),
        HorizontalRule(y=435.8, x0=53.8, x1=558.2),
    ]

    mapped = _map_fill_to_template_field(
        fill.rect, synth, template, PAGE, horiz_rules=rules
    )
    assert mapped is not None
    assert mapped.y0 == 424.9 + TABLE_RULE_INSET_PTS
    assert mapped.y1 == 435.8 - TABLE_RULE_INSET_PTS
    assert mapped.y1 < 448.0


def test_process_comments_snaps_into_template_row() -> None:
    header = _span("Page / Step", 59.5, 341.5, 108.0, 351.5, WHITE)
    fill = _span("N/A", 59.5, 362.7, 74.5, 373.7, BLUE)
    synth = [header, fill]
    template = [header]
    rules = [
        HorizontalRule(y=330.9, x0=54.2, x1=557.8),
        HorizontalRule(y=353.0, x0=54.2, x1=557.8),
        HorizontalRule(y=364.0, x0=53.8, x1=558.2),
    ]

    mapped = _map_fill_to_template_field(
        fill.rect, synth, template, PAGE, horiz_rules=rules
    )
    assert mapped is not None
    assert mapped.y0 == 353.0 + TABLE_RULE_INSET_PTS
    assert mapped.y1 == 364.0 - TABLE_RULE_INSET_PTS
    assert mapped.y0 < 362.7


def test_left_row_fill_y_is_not_snapped_to_header_row() -> None:
    header = _span("Performed By / Date", 395.5, 248.0, 481.5, 258.0, WHITE)
    label = _span("Endotoxin", 227.4, 265.0, 269.9, 275.0, BLACK)
    fill = _span("JR 21JUL26", 395.5, 270.2, 445.0, 280.2, BLUE)
    synth = [header, label, fill]
    template = [header, label]
    rules = [
        HorizontalRule(y=258.0, x0=54.0, x1=558.0),
        HorizontalRule(y=282.0, x0=54.0, x1=558.0),
    ]

    mapped = _map_fill_to_template_field(
        fill.rect, synth, template, PAGE, horiz_rules=rules
    )
    assert mapped is not None
    assert abs(mapped.y0 - fill.rect.y0) < 3.0
    assert mapped.y0 > 265.0


def test_header_beyond_max_gap_is_ignored() -> None:
    header = _span("Performed By / Date", 59.5, 50.0, 145.5, 60.0, WHITE)
    fill = _span("ST 24JUL26", 59.5, 60.0 + HEADER_ANCHOR_MAX_GAP_PTS + 20.0, 120.6, 92.0, BLUE)
    synth = [header, fill]
    template = [header]

    assert _find_column_header_anchor(fill.rect, synth, template) is None
    assert _find_fill_anchor(fill.rect, synth, template) is None
