#!/usr/bin/env python3
"""Smoke-test DiffBrush on Modal with a short string + representative style."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services" / "modal_diffbrush"))

from app import DiffBrushService  # noqa: E402

STYLE = REPO / "data" / "styles" / "default" / "representative_text.png"
OUT = REPO / "data" / "outputs" / "smoke_line.png"


def main():
    texts = ["Yes", "WS-AEX-26051", "92.0"]
    style = STYLE.read_bytes()
    print(f"Calling Modal DiffBrushService for {len(texts)} lines…")
    pngs = DiffBrushService().generate_lines.remote(texts, style, steps=20, seed=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i, png in enumerate(pngs):
        path = OUT.parent / f"smoke_line_{i}.png"
        path.write_bytes(png)
        print(f"  wrote {path} ({len(png)} bytes)")
    print("OK")


if __name__ == "__main__":
    main()
