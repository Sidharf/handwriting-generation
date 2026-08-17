"""Emuru styled text-image sampler (runs inside Modal GPU container)."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


IMG_H = 64
MAX_STYLE_W = 768
MIN_STYLE_W = 64
INK_DARK_THRESH = 200
MIN_INK_DENSITY = 0.04
LEFT_INK_FRAC = 0.15  # truncation if almost all ink is in left 15%
MIN_WIDTH_PER_CHAR = 10  # px of cropped ink width expected per character


def _sup_to_ascii(s: str) -> str:
    return s.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁰", "0123456789"))


def normalize_thousands(text: str) -> str:
    """Strip thousands separators: 10,000 → 10000 (keep decimals)."""
    return re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)


def sanitize_text(text: str) -> str:
    """Normalize unicode punctuation; strip thousands commas; printable ASCII."""
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
        "\n": " ",
        "\t": " ",
        "\r": " ",
    }
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*10\s*[\^¹]?[\s]*([0-9¹²³⁴⁵⁶⁷⁸⁹⁰]+)",
        lambda m: f"{m.group(1)} x 10e{_sup_to_ascii(m.group(2))}",
        text,
    )
    text = text.replace("^", "")
    text = normalize_thousands(text)
    out: List[str] = []
    for ch in text.strip():
        ch = replacements.get(ch, ch)
        o = ord(ch)
        if ch == " " or 32 <= o <= 126:
            out.append(ch)
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip() or "_"


def tokens_for_line(max_new_tokens_ceiling: int, text: str) -> int:
    """Scale tokens by content length, capped by job ceiling (and 256)."""
    needed = 16 + 12 * max(1, len(text))
    return int(max(64, min(256, min(max_new_tokens_ceiling, needed))))


def style_png_to_tensor(style_png: bytes, device: torch.device) -> torch.Tensor:
    """Decode style PNG → RGB float tensor [1,3,H,W] in [-1, 1], H=64."""
    arr = np.frombuffer(style_png, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode style PNG")

    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 4:
        bgr3 = bgr[:, :, :3].astype(np.float32)
        alpha = bgr[:, :, 3:4].astype(np.float32) / 255.0
        flat = (bgr3 * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if h != IMG_H:
        new_w = max(1, int(round(w * IMG_H / h)))
        rgb = cv2.resize(rgb, (new_w, IMG_H), interpolation=cv2.INTER_AREA)

    if rgb.shape[1] > MAX_STYLE_W:
        rgb = rgb[:, :MAX_STYLE_W]
    if rgb.shape[1] < MIN_STYLE_W:
        pad = MIN_STYLE_W - rgb.shape[1]
        rgb = cv2.copyMakeBorder(
            rgb, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    pil = Image.fromarray(rgb)
    t = TF.to_tensor(pil)  # [0,1], C=3
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
    return t.unsqueeze(0).to(device)


def pil_to_gray_arr(pil: Image.Image) -> np.ndarray:
    gray = pil.convert("L")
    arr = np.array(gray)
    if float(arr.mean()) < 127:
        arr = 255 - arr
    return arr


def encode_gray_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Failed to encode Emuru PNG")
    return buf.tobytes()


def thicken_ink(arr: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate dark ink to reduce fragile stroke washout. iterations=0 skips."""
    iters = max(0, int(iterations))
    if iters <= 0:
        return arr
    ink = (arr < INK_DARK_THRESH).astype(np.uint8) * 255
    if ink.max() == 0:
        return arr
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(ink, kernel, iterations=iters)
    out = arr.copy()
    out[thick > 0] = np.minimum(out[thick > 0], 40)
    return out


def _ink_bbox(arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(arr < INK_DARK_THRESH)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def qa_line_ok(arr: np.ndarray, gen_text: str) -> Tuple[bool, str]:
    """Return (ok, reason). Detect sparse / truncated / collapsed generations."""
    h, w = arr.shape[:2]
    ink_mask = arr < INK_DARK_THRESH
    dens = float(ink_mask.mean())
    bbox = _ink_bbox(arr)
    nchar = max(1, len(gen_text.replace(" ", "")))

    if dens < 0.008:
        return False, f"near_empty dens={dens:.4f}"

    if bbox is None:
        return False, "no_ink"

    x0, _, x1, _ = bbox
    ink_w = x1 - x0
    # Collapsed content (e.g. 10000 → single stroke)
    if ink_w < max(24, MIN_WIDTH_PER_CHAR * min(nchar, 8) * 0.35):
        return False, f"too_narrow ink_w={ink_w} nchar={nchar}"

    # Signature truncation: wide canvas, ink only on far left
    if w >= 400 and dens < MIN_INK_DENSITY:
        left = ink_mask[:, : max(1, int(w * LEFT_INK_FRAC))].sum()
        total = max(1, int(ink_mask.sum()))
        if left / total > 0.85:
            return False, f"left_truncated dens={dens:.4f} w={w}"

    if w >= 700 and dens < 0.035:
        return False, f"sparse_wide dens={dens:.4f} w={w}"

    return True, "ok"


class EmuruSampler:
    REQUIRED_FILES = ("config.json", "configuration_emuru.py", "modeling_emuru.py")

    @staticmethod
    def _import_emuru_class(model_dir: str):
        """Load Emuru class from local files (avoids Hub auto_map repo-- lookups)."""
        import importlib.util
        import os
        import sys
        import types

        pkg_name = "emuru_offline"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [model_dir]
            pkg.__file__ = os.path.join(model_dir, "__init__.py")
            sys.modules[pkg_name] = pkg

        def _load_submodule(mod_name: str, filename: str):
            full_name = f"{pkg_name}.{mod_name}"
            if full_name in sys.modules:
                return sys.modules[full_name]
            path = os.path.join(model_dir, filename)
            spec = importlib.util.spec_from_file_location(
                full_name,
                path,
                submodule_search_locations=[model_dir],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            return module

        _load_submodule("configuration_emuru", "configuration_emuru.py")
        modeling = _load_submodule("modeling_emuru", "modeling_emuru.py")
        return modeling.Emuru

    def __init__(self, model_dir: str, device: str = "cuda"):
        import json
        import os

        model_dir = os.path.abspath(model_dir)
        listing = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
        missing = [
            name
            for name in self.REQUIRED_FILES
            if not os.path.isfile(os.path.join(model_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"Emuru model dir {model_dir} missing {missing}. "
                f"Contents: {listing}. Re-run scripts/populate_modal_emuru_volume.py."
            )

        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(
            "EmuruSampler loading:",
            f"model_dir={model_dir}",
            f"vae={cfg.get('vae_name_or_path')}",
            f"tokenizer={cfg.get('tokenizer_name_or_path')}",
            f"t5={cfg.get('t5_name_or_path')}",
            f"_name_or_path={cfg.get('_name_or_path')}",
            f"auto_map={cfg.get('auto_map')}",
            flush=True,
        )

        Emuru = self._import_emuru_class(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Emuru.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"EmuruSampler ready on {self.device}", flush=True)

    def _sample_once(
        self,
        gen_text: str,
        style_prompt: str,
        style_img: torch.Tensor,
        max_new_tokens: int,
        seed: Optional[int],
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        pil = self.model.generate(
            style_text=style_prompt,
            gen_text=gen_text,
            style_img=style_img,
            max_new_tokens=max_new_tokens,
        )
        return pil_to_gray_arr(pil)

    @torch.inference_mode()
    def generate_line(
        self,
        text: str,
        style_png: bytes,
        style_text: str,
        max_new_tokens: int = 128,
        seed: Optional[int] = None,
        thicken: int = 1,
    ) -> bytes:
        gen_text = sanitize_text(text)
        style_prompt = sanitize_text(style_text)
        if not style_prompt or style_prompt == "_":
            raise ValueError("style_text is required for Emuru")

        tokens = tokens_for_line(max_new_tokens, gen_text)
        # Pure / mostly-digit strings collapse more often — start higher.
        alnum = [c for c in gen_text if c.isalnum()]
        digit_heavy = bool(alnum) and sum(c.isdigit() for c in alnum) / len(alnum) >= 0.7
        if digit_heavy:
            tokens = int(min(256, max(tokens, min(max_new_tokens, 96))))

        style_img = style_png_to_tensor(style_png, self.device)

        def score(a: np.ndarray) -> float:
            bb = _ink_bbox(a)
            dens = float((a < INK_DARK_THRESH).mean())
            width = 0 if bb is None else (bb[2] - bb[0])
            return dens * max(width, 1)

        arr = self._sample_once(gen_text, style_prompt, style_img, tokens, seed)
        ok, reason = qa_line_ok(arr, gen_text)
        attempts = 1 if ok else (3 if digit_heavy else 2)
        cur_tokens = tokens
        for attempt in range(1, attempts):
            if ok:
                break
            cur_tokens = int(min(256, max(cur_tokens + 32, int(cur_tokens * 1.5))))
            retry_seed = None if seed is None else seed + attempt
            print(
                f"Emuru QA retry text={gen_text!r} reason={reason} "
                f"tokens={tokens}->{cur_tokens} attempt={attempt}",
                flush=True,
            )
            arr2 = self._sample_once(
                gen_text, style_prompt, style_img, cur_tokens, retry_seed
            )
            ok2, reason2 = qa_line_ok(arr2, gen_text)
            if ok2 or score(arr2) > score(arr):
                arr = arr2
                ok, reason = ok2, reason2
            if not ok:
                print(
                    f"Emuru QA still weak text={gen_text!r} reason={reason}",
                    flush=True,
                )

        arr = thicken_ink(arr, iterations=thicken)
        return encode_gray_png(arr)

    def generate_lines(
        self,
        texts: List[str],
        style_png: bytes,
        style_text: str,
        max_new_tokens: int = 128,
        seed: Optional[int] = None,
        seed_stride: int = 17,
        thicken: int = 1,
        start_index: int = 0,
    ) -> List[bytes]:
        stride = max(1, int(seed_stride))
        start = max(0, int(start_index))
        out: List[bytes] = []
        for i, text in enumerate(texts):
            line_seed = None if seed is None else seed + (start + i) * stride
            out.append(
                self.generate_line(
                    text,
                    style_png,
                    style_text,
                    max_new_tokens=max_new_tokens,
                    seed=line_seed,
                    thicken=thicken,
                )
            )
        return out
