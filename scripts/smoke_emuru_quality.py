#!/usr/bin/env python3
"""Quality smoke for Emuru: numbers, signatures, short IDs must not collapse/truncate."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services" / "api"))

from line_quality import INK_DARK_THRESH, analyze_line, qa_line_ok  # noqa: E402
from line_qa import line_needs_vector  # noqa: E402
from pdf_stamp import render_vector_line_png  # noqa: E402
from style_prep import preprocess_style_png  # noqa: E402
import modal  # noqa: E402


CASES = [
    ("10000", {"min_chars_equiv": 4}),
    ("10,000", {"min_chars_equiv": 4, "expect_sanitize": "10000"}),
    ("TN 10JUN26", {"min_chars_equiv": 6, "forbid_left_only": True}),
    ("PA 14JUN26", {"min_chars_equiv": 6, "forbid_left_only": True}),
    ("X", {"min_chars_equiv": 1}),
    ("8.9", {"min_chars_equiv": 2}),
    ("DS-26052", {"min_chars_equiv": 5}),
    ("30 NOV 2026", {"min_chars_equiv": 7}),
    ("94.8", {"min_chars_equiv": 3}),
    ("K562-2026-01/100", {"min_chars_equiv": 10}),
    ("KL 21 JUL 26", {"min_chars_equiv": 7}),
]


def sanitize_text_local(text: str) -> str:
    import re

    text = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)
    return text.strip()


def _png_wh(png: bytes) -> tuple[int, int]:
    return struct.unpack(">II", png[16:24])


def _decode_gray(png: bytes) -> np.ndarray:
    arr = np.frombuffer(png, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to decode PNG")
    return img


def _local_unit_checks() -> None:
    assert sanitize_text_local("10,000") == "10000"
    assert sanitize_text_local("8.9") == "8.9"
    blank = np.full((64, 800), 255, dtype=np.uint8)
    blank[20:40, 10:18] = 30
    ok, reason = qa_line_ok(blank, "10000")
    assert not ok, f"expected fail on collapse, got {reason}"
    print("local unit checks OK")


def main() -> int:
    _local_unit_checks()

    style_path = REPO / "data" / "styles" / "default" / "representative_text.png"
    if not style_path.exists():
        style_path = REPO / "representative_text.png"
    style = preprocess_style_png(style_path.read_bytes())
    style_text = "An erratic some-CAPS"

    texts = [c[0] for c in CASES]
    print(f"Calling Modal EmuruService quality smoke ({len(texts)} lines)…")
    Cls = modal.Cls.from_name("handwriting-emuru", "EmuruService")
    pngs = Cls().generate_lines.remote(
        texts, style, style_text, max_new_tokens=128, seed=7
    )

    out_dir = REPO / "data" / "outputs" / "smoke_emuru_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    fallback_count = 0
    for i, ((raw, rules), png) in enumerate(zip(CASES, pngs)):
        expected = rules.get("expect_sanitize", sanitize_text_local(raw))
        if sanitize_text_local(raw) != expected:
            failures.append(
                f"{raw}: sanitize got {sanitize_text_local(raw)!r} want {expected!r}"
            )

        raw_quality = analyze_line(_decode_gray(png), expected)
        source = "emuru"
        if line_needs_vector(png, expected):
            png = render_vector_line_png(expected, seed=7 + i)
            fallback_count += 1
            fallback_reason = (
                raw_quality.reason if not raw_quality.ok else "exact_numeric"
            )
            source = f"vector({fallback_reason})"

        gray = _decode_gray(png)
        w, h = _png_wh(png)
        safe = raw.replace(",", "c").replace(" ", "_").replace("/", "_")
        path = out_dir / f"{i:02d}_{safe}.png"
        path.write_bytes(png)

        quality = analyze_line(gray, expected)
        ok, reason = quality.ok, quality.reason
        dens = quality.metrics["dark_density"]
        ink_w = int(quality.metrics["ink_w"])
        print(
            f"{raw!r} -> {path.name} {w}x{h} dens={dens:.4f} "
            f"ink_w={ink_w} qa={reason} source={source}"
        )

        if not ok:
            failures.append(f"{raw}: QA fail ({reason})")
            continue

        min_chars = int(rules["min_chars_equiv"])
        if ink_w < max(20, 8 * min_chars):
            failures.append(
                f"{raw}: ink width {ink_w}px too small for ~{min_chars} chars"
            )

        if rules.get("forbid_left_only") and w >= 400:
            left = (gray[:, : max(1, int(w * 0.15))] < INK_DARK_THRESH).sum()
            total = max(1, int((gray < INK_DARK_THRESH).sum()))
            if dens < 0.04 and left / total > 0.85:
                failures.append(f"{raw}: left-only truncation pattern")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"OK — quality smoke passed ({fallback_count} vector fallbacks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
