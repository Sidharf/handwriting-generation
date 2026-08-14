"""DiffBrush single-device line sampler (runs inside Modal GPU container)."""

from __future__ import annotations

import io
import os
import pickle
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
FIXED_LEN = 1024
IMG_H = 64
WRITER_NUMS = 496
CRITICAL_WIDTH = 512
INK_THRESH = 40
# DiffBrush short-text tiles rarely leave a clean 24px gutter; use softer valleys.
REPEAT_GAP_PX = 10
COL_SMOOTH = 7
VALLEY_FRAC = 0.15
PX_PER_CHAR_MIN = 18
PX_PER_CHAR_MAX = 32


def sanitize_text(text: str) -> str:
    """Map common characters into the DiffBrush IAM charset."""
    replacements = {
        "×": "x",
        "x": "x",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "°": " ",
        "µ": "u",
        "μ": "u",
        "^": "",
        "\n": " ",
        "\t": " ",
    }
    out = []
    for ch in text.strip():
        ch = replacements.get(ch, ch)
        if ch in LETTERS:
            out.append(ch)
        elif ch == " ":
            out.append(" ")
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip() or " "


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    return (255 - gray.astype(np.int16)) > INK_THRESH


def _to_gray_white_bg_arr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        # imdecode gives BGRA
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


def _select_densest_line_band(gray: np.ndarray) -> np.ndarray:
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


def _resize_pad_line(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = IMG_H / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    gray = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    if new_w < CRITICAL_WIDTH:
        pad = np.full((IMG_H, CRITICAL_WIDTH), 255, dtype=np.uint8)
        pad[:, :new_w] = gray
        gray = pad
    elif new_w > FIXED_LEN:
        gray = gray[:, :FIXED_LEN]
    return gray


def prepare_style_gray(style_png: bytes) -> np.ndarray:
    """Single-line white IAM-like style crop as uint8 gray."""
    arr = np.frombuffer(style_png, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode style PNG")
    gray = _to_gray_white_bg_arr(img)
    gray = _select_densest_line_band(gray)
    gray = _crop_ink_bbox(gray)
    return _resize_pad_line(gray)


def crop_generated_line(
    gray: np.ndarray, text: str = "", pad: int = 4
) -> np.ndarray:
    """
    Left-biased crop: keep ink from the left until a density valley after the
    first cluster(s). Caps width from text length so short-token tiles don't
    stamp as full 1024-wide mush.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    # Normalize: ink dark on white
    if float(np.mean(gray)) < 127:
        gray = 255 - gray

    # Slightly stricter ink for gap finding (gray haze is common)
    ink = (255 - gray.astype(np.int16)) > max(INK_THRESH, 50)
    col = ink.sum(axis=0).astype(np.float32)
    if col.max() < 2:
        return gray

    k = COL_SMOOTH
    sm = np.convolve(col, np.ones(k, dtype=np.float32) / k, mode="same")
    thr = max(2.0, VALLEY_FRAC * float(sm.max()))

    ink_cols = np.where(sm > thr)[0]
    if len(ink_cols) == 0:
        return gray
    x_start = int(ink_cols[0])

    n = max(1, len((text or "").strip()) or 4)
    min_w = max(36, int(n * PX_PER_CHAR_MIN))
    if n <= 12:
        max_w = min(gray.shape[1], int(n * PX_PER_CHAR_MAX) + 24)
    else:
        max_w = gray.shape[1]

    seen_ink = False
    gap = 0
    x_end = int(ink_cols[-1])
    for x in range(x_start, gray.shape[1]):
        width = x_end - x_start + 1
        if width >= max_w:
            x_end = x_start + max_w - 1
            break
        if sm[x] > thr:
            seen_ink = True
            gap = 0
            x_end = x
        elif seen_ink:
            gap += 1
            width = x_end - x_start + 1
            if gap >= REPEAT_GAP_PX and width >= min_w:
                break

    if x_end - x_start + 1 > max_w:
        x_end = x_start + max_w - 1

    x0 = max(0, x_start - pad)
    x1 = min(gray.shape[1], x_end + pad + 1)

    band = gray[:, x0:x1]
    row = ink[:, x0:x1].sum(axis=1)
    ys = np.where(row > 0)[0]
    if len(ys) > 0:
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
        band = band[y0:y1, :]
    return band


def encode_gray_png(gray: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return encoded.tobytes()


class DiffBrushSampler:
    def __init__(
        self,
        ckpt_path: str,
        vae_path: str,
        unifont_path: str,
        device: str = "cuda",
    ):
        from models.unet import UNetModel
        from models.diffusion import Diffusion
        from diffusers import AutoencoderKL

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.letter2index = {label: n for n, label in enumerate(LETTERS)}
        self.con_symbols = self._load_symbols(unifont_path)

        self.diffusion = Diffusion(device=str(self.device))
        self.unet = UNetModel(
            in_channels=4,
            model_channels=512,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=4,
            context_dim=512,
            nb_classes=WRITER_NUMS,
        ).to(self.device)

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        self.unet.load_state_dict(state)
        self.unet.eval()

        if os.path.basename(os.path.normpath(vae_path)) == "vae":
            stable_path = os.path.dirname(vae_path)
            subfolder = "vae"
        elif os.path.isdir(os.path.join(vae_path, "vae")):
            stable_path = vae_path
            subfolder = "vae"
        else:
            stable_path = vae_path
            subfolder = None

        if subfolder:
            self.vae = AutoencoderKL.from_pretrained(stable_path, subfolder=subfolder)
        else:
            self.vae = AutoencoderKL.from_pretrained(stable_path)
        self.vae = self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

    def _load_symbols(self, unifont_path: str) -> torch.Tensor:
        with open(unifont_path, "rb") as f:
            symbols = pickle.load(f)
        symbols = {sym["idx"][0]: sym["mat"].astype(np.float32) for sym in symbols}
        contents = []
        for char in LETTERS:
            contents.append(torch.from_numpy(symbols[ord(char)]).float())
        contents.append(torch.zeros_like(contents[0]))
        return torch.stack(contents)

    def get_content(self, label: str) -> torch.Tensor:
        idxs = [self.letter2index[c] for c in label]
        content_ref = self.con_symbols[idxs]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)

    def preprocess_style(self, style_png: bytes) -> torch.Tensor:
        """Return style tensor [1, 1, H, W] float in [0,1]."""
        gray = prepare_style_gray(style_png)
        style = gray.astype(np.float32) / 255.0
        return torch.from_numpy(style).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate_line(
        self,
        text: str,
        style_png: bytes,
        steps: int = 50,
        seed: Optional[int] = None,
        eta: float = 0.0,
    ) -> bytes:
        text = sanitize_text(text)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        style = self.preprocess_style(style_png).to(self.device)
        content = self.get_content(text).to(self.device)

        x = torch.randn(
            (1, 4, style.shape[2] // 8, FIXED_LEN // 8),
            device=self.device,
        )
        images = self.diffusion.ddim_sample(
            self.unet,
            self.vae,
            1,
            x,
            style,
            content,
            steps,
            eta,
        )
        im = torchvision.transforms.ToPILImage()(images[0])
        gray = np.array(im.convert("L"))
        gray = crop_generated_line(gray, text=text)
        return encode_gray_png(gray)

    def generate_lines(
        self,
        texts: List[str],
        style_png: bytes,
        steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[bytes]:
        out: List[bytes] = []
        for i, text in enumerate(texts):
            line_seed = None if seed is None else seed + i
            out.append(self.generate_line(text, style_png, steps=steps, seed=line_seed))
        return out
