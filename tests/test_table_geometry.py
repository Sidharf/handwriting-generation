from __future__ import annotations

import fitz

from cell_diff import FillCell, _expand_text_fields_to_columns
from table_geometry import (
    TABLE_RULE_INSET_PTS,
    HorizontalRule,
    VerticalRule,
    extract_horizontal_rules,
    extract_vertical_rules,
    nearest_right_rule,
    row_below_header,
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


def test_extracts_horizontal_rules_and_row_below_header() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(54.0, 379.1, 558.0, 402.4))
    shape.draw_rect(fitz.Rect(54.0, 402.4, 306.0, 424.5))
    shape.draw_rect(fitz.Rect(306.0, 402.4, 558.0, 424.5))
    shape.draw_line(fitz.Point(53.8, 435.8), fitz.Point(558.2, 435.8))
    shape.draw_rect(fitz.Rect(220, 410, 226, 412))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()

    rules = extract_horizontal_rules(page)
    header = fitz.Rect(59.5, 413.3, 222.0, 423.3)
    row = row_below_header(rules, header)

    assert row is not None
    assert abs(row[0] - 424.5) < 0.5
    assert abs(row[1] - 435.8) < 0.5
    title = fitz.Rect(59.5, 389.9, 205.5, 401.1)
    title_row = row_below_header(rules, title)
    assert title_row is not None
    assert title_row[1] < 430.0
    doc.close()


def test_row_below_header_skips_title_band_for_column_header() -> None:
    rules = [
        HorizontalRule(y=379.4, x0=53.8, x1=558.2),
        HorizontalRule(y=402.6, x0=54.2, x1=557.8),
        HorizontalRule(y=424.9, x0=54.2, x1=557.8),
        HorizontalRule(y=435.8, x0=53.8, x1=558.2),
    ]
    header = fitz.Rect(59.5, 413.3, 222.0, 423.3)
    row = row_below_header(rules, header)
    assert row == (424.9, 435.8)
