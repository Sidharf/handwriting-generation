"""Prepare user handwriting PNGs for Emuru style conditioning.

Produces single-line grayscale crops: white background, height 64,
width capped at style_len=352. Short styles padded to >=128.
Emuru sampler converts the crop to RGB tensors at inference time.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

IMG_H = 64
STYLE_LEN = 352
MIN_STYLE_W = 128
INK_THRESH = 40  # darkness above this counts as ink (255 - gray)
EDGE_PAD = 8  # white padding so strokes are not edge-clipped


def _to_gray_white_bg(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid PNG")

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        flat = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = np.where(gray > 160, 255, gray).astype(np.uint8)

    h, w = gray.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = gray.copy()
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if filled[seed[1], seed[0]] > 140:
            cv2.floodFill(filled, mask, seed, 255, loDiff=40, upDiff=40)
    gray = filled

    if float(np.mean(gray)) < 127:
        gray = 255 - gray

    return gray


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    return (255 - gray) > INK_THRESH


def _select_densest_line_band(gray: np.ndarray) -> np.ndarray:
    """If multiple horizontal ink bands exist, keep the densest one."""
    ink = _ink_mask(gray)
    row_sum = ink.sum(axis=1).astype(np.float32)
    if row_sum.max() < 5:
        return gray

    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(row_sum, kernel, mode="same")
    thresh = max(3.0, 0.15 * float(smooth.max()))
    active = smooth >= thresh

    bands = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(active) - 1))

    if len(bands) <= 1:
        return gray

    best = max(bands, key=lambda b: float(row_sum[b[0] : b[1] + 1].sum()))
    y0, y1 = best
    pad = 4
    y0 = max(0, y0 - pad)
    y1 = min(gray.shape[0] - 1, y1 + pad)
    return gray[y0 : y1 + 1, :]


def _crop_ink_bbox(gray: np.ndarray, pad: int = 4) -> np.ndarray:
    ink = _ink_mask(gray)
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return gray
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    return gray[y0:y1, x0:x1]


def _pad_white(gray: np.ndarray, pad: int = EDGE_PAD) -> np.ndarray:
    """Add white border so strokes are not flush with the crop edge."""
    if pad <= 0:
        return gray
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _densest_horizontal_window(gray: np.ndarray, target_w: int) -> np.ndarray:
    """Pick the densest contiguous horizontal window of width target_w."""
    h, w = gray.shape[:2]
    if w <= target_w:
        return gray
    ink = _ink_mask(gray)
    col_sum = ink.sum(axis=0).astype(np.float32)
    # Sliding window sum
    csum = np.cumsum(col_sum)
    best_score = -1.0
    best_x0 = 0
    for x0 in range(0, w - target_w + 1):
        x1 = x0 + target_w
        score = float(csum[x1 - 1] - (csum[x0 - 1] if x0 > 0 else 0.0))
        if score > best_score:
            best_score = score
            best_x0 = x0
    # Prefer densest; if tied near-empty, fall back to center
    if best_score < 5:
        best_x0 = max(0, (w - target_w) // 2)
    return gray[:, best_x0 : best_x0 + target_w]


def _resize_pad_line(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = IMG_H / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    gray = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_AREA)

    if new_w > STYLE_LEN:
        # Densest horizontal window (not always left-truncate)
        gray = _densest_horizontal_window(gray, STYLE_LEN)
    elif new_w < MIN_STYLE_W:
        pad = np.full((IMG_H, MIN_STYLE_W), 255, dtype=np.uint8)
        pad[:, :new_w] = gray
        gray = pad
    return gray


def preprocess_style_png(png_bytes: bytes) -> bytes:
    """Flatten, whiten, pick densest line, crop, pad, resize height=64, cap width."""
    gray = _to_gray_white_bg(png_bytes)
    gray = _select_densest_line_band(gray)
    gray = _crop_ink_bbox(gray)
    gray = _pad_white(gray, EDGE_PAD)
    gray = _resize_pad_line(gray)

    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Failed to encode prepared style")
    return encoded.tobytes()


def preview_jpeg(png_bytes: bytes, max_w: int = 640) -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, max(1, int(h * max_w / w))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
