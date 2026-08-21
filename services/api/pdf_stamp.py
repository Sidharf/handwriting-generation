"""Stamp generated handwriting line PNGs as #2563EB ink onto a template PDF."""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image

from paths import INK_BLUE
from table_geometry import (
    TABLE_RULE_INSET_PTS,
    VerticalRule,
    extract_vertical_rules,
    nearest_right_rule,
)
from variation import (
    VariationAxes,
    height_fit_frac,
    line_weight_params,
    placement_offsets,
    resolve_variation,
    rng_unit,
    stroke_ink_params,
)

# Embed raster at this multiple of PDF-point size so viewers don't upsample mush.
STAMP_DPI_SCALE = 4
SHORT_FIELD_DPI_SCALE = 8
INK_DARK_THRESH = 200
# Pre-inset cell height; inset 15→13pt boxes must still count as short.
SHORT_FIELD_CELL_HEIGHT_PTS = 16.0
SHORT_FIELD_FIT = 0.94


def thin_ink(png_bytes: bytes, iterations: int = 1) -> bytes:
    """Erode dark ink to produce thinner strokes. Mirrors thicken_ink in sampler."""
    if iterations <= 0:
        return png_bytes
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return png_bytes
    ink = (gray < INK_DARK_THRESH).astype(np.uint8) * 255
    if ink.max() == 0:
        return png_bytes
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(ink, kernel, iterations=iterations)
    out = np.full_like(gray, 255)
    out[eroded > 0] = gray[eroded > 0]
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return png_bytes
    return buf.tobytes()


def thicken_ink(png_bytes: bytes, iterations: int = 1) -> bytes:
    """Dilate dark ink so hairlines survive short-field downscale."""
    if iterations <= 0:
        return png_bytes
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return png_bytes
    ink = (gray < INK_DARK_THRESH).astype(np.uint8) * 255
    if ink.max() == 0:
        return png_bytes
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(ink, kernel, iterations=iterations)
    out = gray.copy()
    out[thick > 0] = np.minimum(out[thick > 0], 40)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return png_bytes
    return buf.tobytes()


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
    return Image.fromarray(rgba)


def crop_ink(im: Image.Image, pad: int = 2) -> Image.Image:
    arr = np.asarray(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min() - pad), min(im.width, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(im.height, ys.max() + pad + 1)
    return im.crop((x0, y0, x1, y1))


def ensure_minimum_ink_visibility(im: Image.Image) -> Image.Image:
    """Strengthen only low-opacity hairlines after final raster resizing."""
    arr = np.array(im.convert("RGBA"))
    alpha = arr[:, :, 3]
    visible = alpha > 10
    if not visible.any():
        return im

    visible_alpha = alpha[visible]
    core_ratio = float((visible_alpha >= 160).mean())
    alpha_p75 = float(np.percentile(visible_alpha, 75))
    if core_ratio >= 0.18 or alpha_p75 >= 150:
        return im

    # One output-pixel growth is enough to survive PDF viewer resampling without
    # changing the apparent weight of normal, already-opaque handwriting.
    kernel = np.ones((2, 2), np.uint8)
    mask = visible.astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    grown = cv2.dilate(closed, kernel, iterations=1) > 0
    arr[:, :, 3] = np.where(grown, np.maximum(alpha, 180), alpha)
    return Image.fromarray(arr)


def compute_stamp_rect(
    target: fitz.Rect,
    image_width: int,
    image_height: int,
    fit: float,
    axes: VariationAxes,
    seed: Optional[int],
    cell_index: int,
) -> fitz.Rect:
    """Fit and place an image wholly inside a target rectangle."""
    tw, th = float(target.width), float(target.height)
    aspect = float(image_width) / max(float(image_height), 1.0)
    nh = min(th, max(1.0, th * float(fit)))
    nw = nh * aspect

    if nw > tw:
        nw = tw
        nh = min(th, nw / max(aspect, 1e-6))
    if nh > th:
        nh = th
        nw = min(tw, nh * aspect)

    nw = min(tw, max(1.0, nw))
    nh = min(th, max(1.0, nh))
    leftover_x = max(0.0, tw - nw)
    leftover_y = max(0.0, th - nh)
    jx, jy, _ = placement_offsets(
        axes.placement,
        seed,
        cell_index,
        leftover_x,
        leftover_y,
    )

    dx = target.x0 + jx
    dy = target.y0 + leftover_y / 2.0 + jy
    max_dx = max(target.x0, target.x1 - nw)
    max_dy = max(target.y0, target.y1 - nh)
    dx = min(max(dx, target.x0), max_dx)
    dy = min(max(dy, target.y0), max_dy)
    return fitz.Rect(dx, dy, min(target.x1, dx + nw), min(target.y1, dy + nh))


def render_vector_line_png(text: str, seed: Optional[int] = None) -> bytes:
    """Hershey stroke letters on white paper. Fallback when Emuru collapses."""
    raw = (text or "").strip() or "_"
    rng = random.Random(0 if seed is None else int(seed))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.55
    thickness = 2
    gap = 2
    widths = []
    for ch in raw:
        (cw, _chh), _ = cv2.getTextSize(ch, font, font_scale, thickness)
        widths.append(max(cw, 6))
    total_w = int(sum(widths) + gap * max(0, len(raw) - 1) + 28)
    h = 64
    canvas = np.full((h, max(80, total_w)), 255, dtype=np.uint8)
    x = 12
    y = int(h * 0.72)
    for ch, cw in zip(raw, widths):
        jx = int(round(rng.uniform(-1.2, 1.2)))
        jy = int(round(rng.uniform(-2.0, 2.0)))
        cv2.putText(
            canvas,
            ch,
            (x + jx, y + jy),
            font,
            font_scale,
            25,
            thickness,
            cv2.LINE_AA,
        )
        x += cw + gap
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode vector line PNG")
    return buf.tobytes()


def _ink_rgb01() -> Tuple[float, float, float]:
    return (INK_BLUE[0] / 255.0, INK_BLUE[1] / 255.0, INK_BLUE[2] / 255.0)


def _rotate_pt(
    x: float, y: float, cx: float, cy: float, deg: float
) -> Tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def stamp_check_x(
    page: fitz.Page,
    bbox: Sequence[float],
    axes: VariationAxes,
    seed: Optional[int],
    cell_index: int,
    line_weight: int,
) -> None:
    """Draw a jittered two-stroke X over a printed ☐ in blue ink."""
    bx0, by0, bx1, by1 = [float(v) for v in bbox]
    w, h = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
    # ☐ glyphs usually sit on the left of a wide advance; square that region.
    side = min(w, h)
    x0 = bx0
    y0 = by0 + (h - side) / 2.0
    x1 = x0 + side
    y1 = y0 + side
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pad = side
    inset = pad * 0.12
    overshoot = pad * 0.04
    ax0 = x0 + inset - overshoot
    ay0 = y0 + inset - overshoot
    ax1 = x1 - inset + overshoot
    ay1 = y1 - inset + overshoot

    leftover_x = max(0.0, side * 0.15)
    leftover_y = max(0.0, side * 0.15)
    jx, jy, rot = placement_offsets(axes.placement, seed, cell_index, leftover_x, leftover_y)
    amp = pad * 0.12

    def j(channel: int) -> float:
        return amp * rng_unit(seed, cell_index, channel)

    p1 = (ax0 + j(10) + jx, ay0 + j(11) + jy)
    p2 = (ax1 + j(12) + jx, ay1 + j(13) + jy)
    p3 = (ax1 + j(14) + jx, ay0 + j(15) + jy)
    p4 = (ax0 + j(16) + jx, ay1 + j(17) + jy)
    if abs(rot) > 0.05:
        p1 = _rotate_pt(p1[0], p1[1], cx, cy, rot)
        p2 = _rotate_pt(p2[0], p2[1], cx, cy, rot)
        p3 = _rotate_pt(p3[0], p3[1], cx, cy, rot)
        p4 = _rotate_pt(p4[0], p4[1], cx, cy, rot)

    lw = max(-2, min(2, int(line_weight)))
    base_w = 0.75 + 0.16 * lw
    _, _, a_scale = stroke_ink_params(axes.stroke, seed, cell_index)
    w1 = float(max(0.45, min(1.1, base_w * (1.0 + 0.08 * rng_unit(seed, cell_index, 18)))))
    w2 = float(max(0.45, min(1.1, base_w * (1.0 + 0.08 * rng_unit(seed, cell_index, 19)))))
    opacity = float(max(0.75, min(1.0, 0.92 * a_scale)))
    color = _ink_rgb01()

    shape = page.new_shape()
    shape.draw_line(fitz.Point(*p1), fitz.Point(*p2))
    shape.finish(
        color=color,
        width=w1,
        stroke_opacity=opacity,
        closePath=False,
        lineCap=1,
        lineJoin=1,
    )
    shape.draw_line(fitz.Point(*p3), fitz.Point(*p4))
    shape.finish(
        color=color,
        width=w2,
        stroke_opacity=opacity,
        closePath=False,
        lineCap=1,
        lineJoin=1,
    )
    shape.commit()


def stamp_pdf(
    template_pdf: Path,
    cells: Sequence[dict],
    line_pngs: Sequence[bytes],
    out_pdf: Path,
    variation: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    line_weight: int = 0,
) -> Path:
    """
    cells: list of {page, bbox:[x0,y0,x1,y1], enabled, kind?}
    line_pngs: one PNG per enabled *text* cell, in order of those cells.
    kind=="check" cells get a vector X and do not consume a PNG.

    Scale primarily to cell height (~85%) and keep every pixel inside the cell.
    Short cells (pre-inset height < 16pt) use a higher fit, DPI, and a lower
    alpha floor. Line weight still applies (thin_ink when negative).
    Place rect is in PDF points; embedded PNG is high-DPI (not point-sized pixels).
    variation=None or all-zero axes → legacy placement (85% height, centered, no rotate).
    line_weight: -2..+2; negative applies erosion (thinning) before stamping.
    """
    axes = resolve_variation(variation)
    _, thin_iters = line_weight_params(line_weight)
    doc = fitz.open(template_pdf)
    rules_by_page: dict[int, Sequence[VerticalRule]] = {}
    png_i = 0
    cell_i = 0
    for cell in cells:
        if not cell.get("enabled", True):
            continue
        page = doc[cell["page"]]
        if cell.get("kind") == "check":
            stamp_check_x(page, cell["bbox"], axes, seed, cell_i, line_weight)
            cell_i += 1
            continue
        if png_i >= len(line_pngs):
            cell_i += 1
            continue
        x0, y0, x1, y1 = cell["bbox"]
        page_i = int(cell["page"])
        if page_i not in rules_by_page:
            rules_by_page[page_i] = extract_vertical_rules(page)
        rule_x = nearest_right_rule(
            rules_by_page[page_i],
            x0=float(x0),
            y0=float(y0),
            y1=float(y1),
        )
        if rule_x is not None:
            safe_x1 = rule_x - TABLE_RULE_INSET_PTS
            if safe_x1 - float(x0) >= 2.0:
                x1 = min(float(x1), safe_x1)
        short_field = (float(y1) - float(y0)) < SHORT_FIELD_CELL_HEIGHT_PTS
        inset = 1.0
        target = fitz.Rect(x0 + inset, y0 + inset, x1 - inset, y1 - inset)
        if target.width < 2 or target.height < 2:
            target = fitz.Rect(x0, y0, x1, y1)

        floor, power, a_scale = stroke_ink_params(axes.stroke, seed, png_i)
        png_data = line_pngs[png_i]
        if short_field:
            floor = min(floor, 0.02)
            power = min(power, 0.85)
        if thin_iters > 0:
            png_data = thin_ink(png_data, iterations=thin_iters)
        rgba = crop_ink(
            grayscale_to_blue_rgba(
                png_data,
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

        fit = height_fit_frac(axes.size, seed, png_i)
        if short_field:
            fit = max(fit, SHORT_FIELD_FIT)
        place = compute_stamp_rect(
            target,
            rgba.width,
            rgba.height,
            fit,
            axes,
            seed,
            png_i,
        )
        nw, nh = float(place.width), float(place.height)

        dpi_scale = SHORT_FIELD_DPI_SCALE if short_field else STAMP_DPI_SCALE
        px_w = max(1, int(round(nw * dpi_scale)))
        px_h = max(1, int(round(nh * dpi_scale)))
        hi = rgba.resize((px_w, px_h), Image.Resampling.LANCZOS)
        if thin_iters == 0:
            hi = ensure_minimum_ink_visibility(hi)

        buf = io.BytesIO()
        hi.save(buf, format="PNG")
        page.insert_image(place, stream=buf.getvalue(), keep_proportion=False, overlay=True)
        png_i += 1
        cell_i += 1

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf)
    doc.close()
    return out_pdf
