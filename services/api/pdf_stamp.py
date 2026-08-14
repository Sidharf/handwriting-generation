"""Stamp generated handwriting line PNGs as #2563EB ink onto a template PDF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence, Tuple

import fitz
import numpy as np
from PIL import Image

from paths import INK_BLUE

# Embed raster at this multiple of PDF-point size so viewers don't upsample mush.
STAMP_DPI_SCALE = 4


def grayscale_to_blue_rgba(png_bytes: bytes, ink_rgb: Tuple[int, int, int] = INK_BLUE) -> Image.Image:
    """White paper -> transparent; dark ink -> blue with alpha from darkness."""
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.asarray(im).astype(np.float32)
    darkness = (255.0 - arr) / 255.0
    darkness = np.clip(darkness, 0, 1)
    # Soft floor: Emuru clean backgrounds; keep thin strokes visible as blue.
    darkness = np.where(darkness < 0.05, 0.0, darkness)
    darkness = np.power(darkness, 0.85)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = ink_rgb[0]
    rgba[:, :, 1] = ink_rgb[1]
    rgba[:, :, 2] = ink_rgb[2]
    rgba[:, :, 3] = (darkness * 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def crop_ink(im: Image.Image, pad: int = 2) -> Image.Image:
    arr = np.asarray(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min() - pad), min(im.width, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(im.height, ys.max() + pad + 1)
    return im.crop((x0, y0, x1, y1))


def stamp_pdf(
    template_pdf: Path,
    cells: Sequence[dict],
    line_pngs: Sequence[bytes],
    out_pdf: Path,
) -> Path:
    """
    cells: list of {page, bbox:[x0,y0,x1,y1], enabled}
    line_pngs: one PNG per enabled cell, in order of enabled cells

    Scale primarily to cell height (~85%), allow width up to 1.15x cell width.
    Place rect is in PDF points; embedded PNG is high-DPI (not point-sized pixels).
    """
    doc = fitz.open(template_pdf)
    png_i = 0
    for cell in cells:
        if not cell.get("enabled", True):
            continue
        if png_i >= len(line_pngs):
            break
        page = doc[cell["page"]]
        x0, y0, x1, y1 = cell["bbox"]
        inset = 1.0
        target = fitz.Rect(x0 + inset, y0 + inset, x1 - inset, y1 - inset)
        if target.width < 2 or target.height < 2:
            target = fitz.Rect(x0, y0, x1, y1)

        rgba = crop_ink(grayscale_to_blue_rgba(line_pngs[png_i]))
        tw, th = float(target.width), float(target.height)

        # Height-first: aim for ~85% of cell height (geometry in PDF points)
        target_h = max(1.0, th * 0.85)
        aspect = float(rgba.width) / max(float(rgba.height), 1.0)
        nh = target_h
        nw = nh * aspect

        max_w = tw * 1.15
        if nw > max_w:
            nw = max_w
            nh = nw / max(aspect, 1e-6)

        if nh > th:
            nh = th
            nw = nh * aspect

        nw = max(1.0, nw)
        nh = max(1.0, nh)

        dx = target.x0
        dy = target.y0 + max(0.0, (th - nh) / 2.0)
        place = fitz.Rect(dx, dy, dx + nw, dy + nh)

        # High-DPI raster for the point-sized place rect (never use points as pixels)
        px_w = max(1, int(round(nw * STAMP_DPI_SCALE)))
        px_h = max(1, int(round(nh * STAMP_DPI_SCALE)))
        # Prefer denser of native vs 4x place so we don't upsample mush from tiny gens
        if rgba.width >= px_w and rgba.height >= px_h:
            hi = rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)
        else:
            # Upscale native to at least 4x place size
            hi = rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        hi.save(buf, format="PNG")
        page.insert_image(place, stream=buf.getvalue(), keep_proportion=False, overlay=True)
        png_i += 1

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf)
    doc.close()
    return out_pdf
