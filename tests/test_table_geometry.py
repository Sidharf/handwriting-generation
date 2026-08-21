from __future__ import annotations

import fitz

from cell_diff import FillCell, _expand_text_fields_to_columns
from table_geometry import (
    TABLE_RULE_INSET_PTS,
    VerticalRule,
    extract_vertical_rules,
    nearest_right_rule,
)


def test_extracts_row_spanning_vertical_rules_and_ignores_short_boxes() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(100, 80), fitz.Point(100, 180))
    shape.draw_line(fitz.Point(300, 80), fitz.Point(300, 180))
    shape.draw_rect(fitz.Rect(220, 112, 226, 118))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()

    rules = extract_vertical_rules(page)
    right = nearest_right_rule(rules, x0=160, y0=105, y1=125)

    assert right is not None
    assert abs(right - 300.0) < 0.1
    doc.close()


def test_expansion_shrinks_stale_last_column_to_table_rule() -> None:
    cell = FillCell(
        id="p0_c0",
        page=0,
        bbox=(478.45, 269.15, 573.46, 281.20),
        text="KL 21 JUL 26",
    )
    rules = {0: [VerticalRule(x=558.0, y0=250.0, y1=300.0)]}

    _expand_text_fields_to_columns(
        [cell],
        {0: fitz.Rect(0, 0, 612, 792)},
        rules,
    )

    assert cell.bbox[2] == 558.0 - TABLE_RULE_INSET_PTS


def test_unbounded_last_column_is_not_expanded_to_page_margin() -> None:
    cell = FillCell(
        id="p0_c0",
        page=0,
        bbox=(478.0, 100.0, 530.0, 112.0),
        text="KL 21 JUL 26",
    )

    _expand_text_fields_to_columns(
        [cell],
        {0: fitz.Rect(0, 0, 612, 792)},
        {0: []},
    )

    assert cell.bbox[2] == 530.0


def test_next_field_remains_a_conservative_fallback_boundary() -> None:
    left = FillCell(
        id="p0_c0",
        page=0,
        bbox=(300.0, 100.0, 330.0, 112.0),
        text="JR 20 JUL 26",
    )
    right = FillCell(
        id="p0_c1",
        page=0,
        bbox=(450.0, 100.0, 500.0, 112.0),
        text="KL 20 JUL 26",
    )

    _expand_text_fields_to_columns(
        [left, right],
        {0: fitz.Rect(0, 0, 612, 792)},
        {0: []},
    )

    assert left.bbox[2] == 442.0
    assert right.bbox[2] == 500.0
