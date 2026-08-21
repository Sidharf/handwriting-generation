from __future__ import annotations

import cv2
import numpy as np

from line_qa import is_controlled_status, is_date_like, line_needs_vector
from line_quality import analyze_line, candidate_quality_key


def _text_line(text: str = "NORMAL", value: int = 25) -> np.ndarray:
    canvas = np.full((64, 260), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (8, 43),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1.15,
        value,
        2,
        cv2.LINE_AA,
    )
    return canvas


def _png(arr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", arr)
    assert ok
    return encoded.tobytes()


def test_good_connected_handwriting_passes() -> None:
    arr = np.full((64, 240), 255, dtype=np.uint8)
    points = []
    for x in range(12, 220):
        y = int(round(34 + 8 * np.sin(x / 12.0)))
        points.append((x, y))
    cv2.polylines(arr, [np.array(points, dtype=np.int32)], False, 25, 2, cv2.LINE_AA)
    cv2.line(arr, (30, 20), (30, 48), 25, 2, cv2.LINE_AA)
    cv2.line(arr, (105, 18), (105, 48), 25, 2, cv2.LINE_AA)

    result = analyze_line(arr, "connectedline")

    assert result.ok, result.reason
    assert not line_needs_vector(_png(arr), "connectedline")


def test_low_contrast_full_width_line_is_rejected_as_faint() -> None:
    arr = _text_line(value=225)

    result = analyze_line(arr, "NORMAL")

    assert not result.ok
    assert result.reason.startswith("faint")
    assert line_needs_vector(_png(arr), "NORMAL")


def test_one_weak_glyph_is_rejected_inside_an_otherwise_dark_line() -> None:
    arr = np.full((64, 180), 255, dtype=np.uint8)
    cv2.putText(
        arr,
        "0.",
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        25,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        arr,
        "5",
        (88, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        225,
        2,
        cv2.LINE_AA,
    )

    result = analyze_line(arr, "0.5")

    assert not result.ok
    assert result.reason.startswith("weak_glyph")


def test_one_dark_but_thin_glyph_is_rejected_by_relative_stroke_weight() -> None:
    arr = np.full((64, 180), 255, dtype=np.uint8)
    cv2.putText(
        arr,
        "0",
        (8, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        25,
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        arr,
        "5",
        (90, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        25,
        1,
        cv2.LINE_AA,
    )

    result = analyze_line(arr, "05")

    assert not result.ok
    assert result.reason.startswith("weak_stroke")


def test_missing_short_numeric_glyph_is_rejected() -> None:
    arr = np.full((64, 180), 255, dtype=np.uint8)
    cv2.putText(
        arr,
        "94.",
        (8, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        25,
        3,
        cv2.LINE_AA,
    )

    result = analyze_line(arr, "94.8")

    assert not result.ok
    assert result.reason.startswith("missing_glyphs")


def test_valid_short_numeric_token_still_uses_exact_vector_fallback() -> None:
    arr = _text_line(text="8.9")

    assert analyze_line(arr, "8.9").ok
    assert line_needs_vector(_png(arr), "8.9")


def test_valid_date_token_uses_exact_vector_fallback() -> None:
    text = "01 OCT 2026"
    arr = np.full((64, 420), 255, dtype=np.uint8)
    cv2.putText(
        arr,
        text,
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        25,
        2,
        cv2.LINE_AA,
    )

    assert analyze_line(arr, text).ok
    assert is_date_like(text)
    assert is_date_like("01OCT2026")
    assert not is_date_like("ST 01 OCT 2026")
    assert line_needs_vector(_png(arr), text)


def test_valid_controlled_status_uses_exact_vector_fallback() -> None:
    text = "Verified"
    arr = _text_line(text=text)

    assert analyze_line(arr, text).ok
    assert is_controlled_status("  VERIFIED ")
    assert not is_controlled_status("Verification complete")
    assert line_needs_vector(_png(arr), text)


def test_localized_blob_is_rejected_without_global_width_trigger() -> None:
    arr = np.full((64, 260), 255, dtype=np.uint8)
    cv2.line(arr, (10, 32), (240, 32), 20, thickness=3, lineType=cv2.LINE_AA)
    cv2.rectangle(arr, (112, 13), (132, 50), 20, thickness=-1)

    result = analyze_line(arr, "NORMAL")

    assert not result.ok
    assert result.reason.startswith("local_smear")
    assert result.metrics["ink_w"] < 400
    assert line_needs_vector(_png(arr), "NORMAL")


def test_collapsed_numeric_line_is_rejected() -> None:
    arr = np.full((64, 260), 255, dtype=np.uint8)
    cv2.rectangle(arr, (8, 20), (15, 42), 20, thickness=-1)

    result = analyze_line(arr, "10000")

    assert not result.ok
    assert result.reason.startswith("too_narrow")


def test_valid_candidate_always_outranks_dense_smear() -> None:
    valid = _text_line()
    smear = _text_line()
    cv2.rectangle(smear, (105, 14), (135, 46), 20, thickness=-1)

    assert candidate_quality_key(valid, "NORMAL") > candidate_quality_key(
        smear, "NORMAL"
    )
