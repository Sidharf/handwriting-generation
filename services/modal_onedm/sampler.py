"""One-DM word-level handwriting sampler (runs inside Modal GPU container)."""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# IAM charset from One-DM data_loader/loader.py
LETTERS = (
    '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
)
IMG_H = 64
STYLE_LEN = 352
MAX_WORD_LEN = 9
CHUNK_GAP_PX = 8
INK_THRESH = 40


def sanitize_text(text: str) -> str:
    """Map common characters into the One-DM IAM charset."""
    replacements = {
        "×": "x",
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
        "/": "/",
        "\n": " ",
        "\t": " ",
        "<": "",
        ">": "",
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
    return cleaned.strip() or "_"


def chunk_text(text: str, max_len: int = MAX_WORD_LEN) -> List[str]:
    """Split into pieces of length <= max_len (prefer spaces)."""
    text = sanitize_text(text)
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    for word in text.split(" "):
        if not word:
            continue
        if len(word) <= max_len:
            chunks.append(word)
            continue
        for i in range(0, len(word), max_len):
            chunks.append(word[i : i + max_len])
    return chunks or ["_"]


def _to_gray_white_bg(img: np.ndarray) -> np.ndarray:
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
    return (255 - gray.astype(np.int16)) > INK_THRESH


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


EDGE_PAD = 8
MIN_STYLE_W = 128


def _pad_white(gray: np.ndarray, pad: int = EDGE_PAD) -> np.ndarray:
    if pad <= 0:
        return gray
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _densest_horizontal_window(gray: np.ndarray, target_w: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if w <= target_w:
        return gray
    ink = _ink_mask(gray)
    col_sum = ink.sum(axis=0).astype(np.float32)
    csum = np.cumsum(col_sum)
    best_score = -1.0
    best_x0 = 0
    for x0 in range(0, w - target_w + 1):
        x1 = x0 + target_w
        score = float(csum[x1 - 1] - (csum[x0 - 1] if x0 > 0 else 0.0))
        if score > best_score:
            best_score = score
            best_x0 = x0
    if best_score < 5:
        best_x0 = max(0, (w - target_w) // 2)
    return gray[:, best_x0 : best_x0 + target_w]


def prepare_style_gray(style_png: bytes) -> np.ndarray:
    """Single-line white style crop as uint8 gray, H=64, W<=STYLE_LEN."""
    arr = np.frombuffer(style_png, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode style PNG")

    # API often sends already-prepared style (H=64, W<=352 grayscale) — skip re-crop.
    if img.ndim == 2 and img.shape[0] == IMG_H and img.shape[1] <= STYLE_LEN:
        gray = img
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if gray.shape[1] < MIN_STYLE_W:
            pad = np.full((IMG_H, MIN_STYLE_W), 255, dtype=np.uint8)
            pad[:, : gray.shape[1]] = gray
            return pad
        return gray
    if (
        img.ndim == 3
        and img.shape[0] == IMG_H
        and img.shape[1] <= STYLE_LEN
        and img.shape[2] in (1, 3, 4)
    ):
        if img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = _to_gray_white_bg(img)
        if gray.shape[0] == IMG_H and gray.shape[1] <= STYLE_LEN:
            if gray.shape[1] < MIN_STYLE_W:
                pad = np.full((IMG_H, MIN_STYLE_W), 255, dtype=np.uint8)
                pad[:, : gray.shape[1]] = gray
                return pad
            return gray

    gray = _to_gray_white_bg(img)
    gray = _select_densest_line_band(gray)
    gray = _crop_ink_bbox(gray)
    gray = _pad_white(gray, EDGE_PAD)

    h, w = gray.shape[:2]
    scale = IMG_H / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    gray = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    if new_w > STYLE_LEN:
        gray = _densest_horizontal_window(gray, STYLE_LEN)
    elif new_w < MIN_STYLE_W:
        pad = np.full((IMG_H, MIN_STYLE_W), 255, dtype=np.uint8)
        pad[:, :new_w] = gray
        gray = pad
    return gray


def compute_laplace(gray: np.ndarray) -> np.ndarray:
    """High-frequency map in [0,1], same HxW as style."""
    # Absolute Laplacian on float [0,1] style (ink dark ≈ 0)
    style_f = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(style_f, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    mx = float(lap.max())
    if mx > 1e-6:
        lap = lap / mx
    return lap.astype(np.float32)


def encode_gray_png(gray: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return encoded.tobytes()


def concat_grays(parts: List[np.ndarray], gap: int = CHUNK_GAP_PX) -> np.ndarray:
    if not parts:
        return np.full((IMG_H, 32), 255, dtype=np.uint8)
    # Normalize heights
    resized = []
    for p in parts:
        if p.ndim == 3:
            p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
        if p.shape[0] != IMG_H:
            scale = IMG_H / float(p.shape[0])
            p = cv2.resize(
                p,
                (max(1, int(round(p.shape[1] * scale))), IMG_H),
                interpolation=cv2.INTER_AREA,
            )
        resized.append(p)
    spacer = np.full((IMG_H, gap), 255, dtype=np.uint8)
    out = [resized[0]]
    for p in resized[1:]:
        out.append(spacer)
        out.append(p)
    return np.concatenate(out, axis=1)


class OneDMSampler:
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
        contents.append(torch.zeros_like(contents[0]))  # PAD
        return torch.stack(contents)

    def get_content(self, label: str) -> torch.Tensor:
        idxs = [self.letter2index[c] for c in label]
        content_ref = self.con_symbols[idxs]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)

    def preprocess_style(
        self, style_png: bytes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return style and laplace tensors [1, 1, H, W] float in [0,1]."""
        gray = prepare_style_gray(style_png)
        style = gray.astype(np.float32) / 255.0
        lap = compute_laplace(gray)
        style_t = torch.from_numpy(style).unsqueeze(0).unsqueeze(0)
        lap_t = torch.from_numpy(lap).unsqueeze(0).unsqueeze(0)
        return style_t, lap_t

    @torch.no_grad()
    def generate_word(
        self,
        text: str,
        style: torch.Tensor,
        laplace: torch.Tensor,
        steps: int = 50,
        seed: Optional[int] = None,
        eta: float = 0.0,
    ) -> np.ndarray:
        text = sanitize_text(text)
        if len(text) > MAX_WORD_LEN:
            text = text[:MAX_WORD_LEN]
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        content = self.get_content(text).to(self.device)
        style = style.to(self.device)
        laplace = laplace.to(self.device)

        # Variable width: n_chars * 32 pixels → latent width n_chars * 4
        x = torch.randn(
            (
                1,
                4,
                style.shape[2] // 8,
                (content.shape[1] * 32) // 8,
            ),
            device=self.device,
        )
        images = self.diffusion.ddim_sample(
            self.unet,
            self.vae,
            1,
            x,
            style,
            laplace,
            content,
            steps,
            eta,
        )
        im = torchvision.transforms.ToPILImage()(images[0])
        gray = np.array(im.convert("L"))
        return _crop_ink_bbox(gray, pad=2)

    def generate_line(
        self,
        text: str,
        style_png: bytes,
        steps: int = 50,
        seed: Optional[int] = None,
    ) -> bytes:
        style, laplace = self.preprocess_style(style_png)
        parts = chunk_text(text)
        grays: List[np.ndarray] = []
        for i, part in enumerate(parts):
            line_seed = None if seed is None else seed + i
            grays.append(
                self.generate_word(
                    part, style, laplace, steps=steps, seed=line_seed
                )
            )
        return encode_gray_png(concat_grays(grays))

    def generate_lines(
        self,
        texts: List[str],
        style_png: bytes,
        steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[bytes]:
        out: List[bytes] = []
        for i, text in enumerate(texts):
            line_seed = None if seed is None else seed + i * 17
            out.append(
                self.generate_line(text, style_png, steps=steps, seed=line_seed)
            )
        return out
