#!/usr/bin/env python3
"""Smoke-test One-DM on Modal with short strings + representative style."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services" / "api"))

from style_prep import preprocess_style_png  # noqa: E402
import modal  # noqa: E402


def main():
    style_path = REPO / "data" / "styles" / "default" / "representative_text.png"
    if not style_path.exists():
        style_path = REPO / "representative_text.png"
    style = preprocess_style_png(style_path.read_bytes())
    (REPO / "data" / "styles" / "default" / "prepared.png").write_bytes(style)

    texts = ["Pass", "DS-26051", "08JUN2026", "HPLC system"]
    print(f"Calling Modal OneDMService for {len(texts)} lines…")
    Cls = modal.Cls.from_name("handwriting-onedm", "OneDMService")
    pngs = Cls().generate_lines.remote(texts, style, steps=50, seed=0)

    out_dir = REPO / "data" / "outputs" / "smoke_onedm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (t, p) in enumerate(zip(texts, pngs)):
        path = out_dir / f"{i}_{t.replace(' ', '_')}.png"
        path.write_bytes(p)
        w, h = struct.unpack(">II", p[16:24])
        print(t, "->", path.name, f"{w}x{h}", len(p), "bytes")
    print("OK")


if __name__ == "__main__":
    main()
