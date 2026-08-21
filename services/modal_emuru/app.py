"""Modal app: Emuru styled text-image generation on L4 (A10G fallback)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import modal

APP_NAME = "handwriting-emuru"
VOLUME_NAME = "emuru-weights"
WEIGHTS_MOUNT = "/weights"
MODEL_DIR = f"{WEIGHTS_MOUNT}/emuru"

_THIS = Path(__file__).resolve()
_PARENTS = _THIS.parents
REPO_ROOT = _PARENTS[2] if len(_PARENTS) >= 3 else Path("/opt")
SAMPLER_LOCAL = _THIS.parent / "sampler.py"
QUALITY_LOCAL = REPO_ROOT / "services" / "api" / "line_quality.py"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "einops==0.8.0",
        "opencv-python-headless==4.10.0.84",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "safetensors==0.4.4",
        "huggingface-hub==0.24.6",
        "tqdm==4.66.5",
        "sentencepiece==0.2.0",
        "protobuf==5.28.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env(
        {
            "HF_HOME": "/tmp/hf",
            "TRANSFORMERS_CACHE": "/tmp/hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_file(str(SAMPLER_LOCAL), remote_path="/opt/sampler.py")
    .add_local_file(str(QUALITY_LOCAL), remote_path="/opt/line_quality.py")
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=["L4", "A10G"],
    timeout=1800,
    scaledown_window=60,
    volumes={WEIGHTS_MOUNT: volume},
)
class EmuruService:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/opt")
        from sampler import EmuruSampler

        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(
                f"Missing {MODEL_DIR}. Run scripts/populate_modal_emuru_volume.py first."
            )
        listing = sorted(os.listdir(MODEL_DIR))
        print(f"EmuruService.load: {MODEL_DIR} → {listing[:50]}", flush=True)
        for required in (
            "config.json",
            "configuration_emuru.py",
            "modeling_emuru.py",
        ):
            path = os.path.join(MODEL_DIR, required)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing {path}. Volume listing: {listing}. "
                    "Re-run scripts/populate_modal_emuru_volume.py."
                )
        # Ensure remote-code modules resolve relative imports from the snapshot
        sys.path.insert(0, MODEL_DIR)
        self.sampler = EmuruSampler(model_dir=MODEL_DIR, device="cuda")

    @modal.method()
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
        return self.sampler.generate_lines(
            texts,
            style_png,
            style_text,
            max_new_tokens=max_new_tokens,
            seed=seed,
            seed_stride=seed_stride,
            thicken=thicken,
            start_index=start_index,
        )


@app.local_entrypoint()
def main(
    text: str = "Pass",
    style_text: str = "An erratic some-CAPS",
    max_new_tokens: int = 128,
):
    """Smoke: modal run services/modal_emuru/app.py --text 'Pass'"""
    style_path = REPO_ROOT / "data" / "styles" / "default" / "representative_text.png"
    if not style_path.exists():
        style_path = REPO_ROOT / "representative_text.png"
    prepared = style_path.parent / "prepared.png"
    raw = style_path.read_bytes()
    try:
        import sys

        api_dir = str(REPO_ROOT / "services" / "api")
        if api_dir not in sys.path:
            sys.path.insert(0, api_dir)
        from style_prep import preprocess_style_png

        style_png = preprocess_style_png(raw)
        prepared.write_bytes(style_png)
    except Exception:
        style_png = prepared.read_bytes() if prepared.exists() else raw

    svc = EmuruService()
    pngs = svc.generate_lines.remote(
        [text],
        style_png,
        style_text,
        max_new_tokens=max_new_tokens,
        seed=0,
    )
    out = REPO_ROOT / "data" / "outputs" / "smoke_emuru.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pngs[0])
    print(f"Wrote {out} ({len(pngs[0])} bytes)")
