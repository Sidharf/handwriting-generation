"""Stamp generated handwriting line PNGs as #2563EB ink onto a template PDF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import fitz
import numpy as np
from PIL import Image

from paths import INK_BLUE
from variation import (
    height_fit_frac,
    placement_offsets,
    resolve_variation,
    stroke_ink_params,
)

# Embed raster at this multiple of PDF-point size so viewers don't upsample mush.
STAMP_DPI_SCALE = 4


def grayscale_to_blue_rgba(
    png_bytes: bytes,
    ink_rgb: Tuple[int, int, int] = INK_BLUE,
    alpha_floor: float = 0.05,
    darkness_power: float = 0.85,
    alpha_scale: float = 1.0,
) -> Image.Image:
    """White paper -> transparent; dark ink -> blue with alpha from darkness."""
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.asarray(im).astype(np.float32)
    darkness = (255.0 - arr) / 255.0
    darkness = np.clip(darkness, 0, 1)
    # Soft floor: Emuru clean backgrounds; keep thin strokes visible as blue.
    darkness = np.where(darkness < alpha_floor, 0.0, darkness)
    darkness = np.power(darkness, darkness_power)
    darkness = np.clip(darkness * alpha_scale, 0.0, 1.0)
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
    variation: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
) -> Path:
    """
    cells: list of {page, bbox:[x0,y0,x1,y1], enabled}
    line_pngs: one PNG per enabled cell, in order of enabled cells

    Scale primarily to cell height (~85%), allow width up to 1.15x cell width.
    Place rect is in PDF points; embedded PNG is high-DPI (not point-sized pixels).
    variation=None or all-zero axes → legacy placement (85% height, centered, no rotate).
    """
    axes = resolve_variation(variation)
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

        floor, power, a_scale = stroke_ink_params(axes.stroke, seed, png_i)
        rgba = crop_ink(
            grayscale_to_blue_rgba(
                line_pngs[png_i],
                alpha_floor=floor,
                darkness_power=power,
                alpha_scale=a_scale,
            )
        )

        # Micro-slant before fit (expand canvas so corners aren't clipped)
        _, _, rot = placement_offsets(axes.placement, seed, png_i, 0.0, 0.0)
        if abs(rot) > 0.05:
            rgba = rgba.rotate(rot, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
            rgba = crop_ink(rgba)

        tw, th = float(target.width), float(target.height)

        fit = height_fit_frac(axes.size, seed, png_i)
        target_h = max(1.0, th * fit)
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

        leftover_x = max(0.0, tw - nw)
        leftover_y = max(0.0, th - nh)
        jx, jy, _ = placement_offsets(axes.placement, seed, png_i, leftover_x, leftover_y)

        # Legacy: left-aligned x, vertically centered. Jitter around that baseline.
        base_dx = target.x0
        base_dy = target.y0 + leftover_y / 2.0
        dx = base_dx + jx
        dy = base_dy + jy
        # Keep stamp fully inside target box
        dx = min(max(dx, target.x0), target.x0 + leftover_x)
        dy = min(max(dy, target.y0), target.y0 + leftover_y)
        place = fitz.Rect(dx, dy, dx + nw, dy + nh)

        px_w = max(1, int(round(nw * STAMP_DPI_SCALE)))
        px_h = max(1, int(round(nh * STAMP_DPI_SCALE)))
        hi = rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        hi.save(buf, format="PNG")
        page.insert_image(place, stream=buf.getvalue(), keep_proportion=False, overlay=True)
        png_i += 1

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf)
    doc.close()
    return out_pdf
