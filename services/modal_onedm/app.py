"""Modal app: One-DM handwriting word generation on T4 (L4 fallback)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import modal

APP_NAME = "handwriting-onedm"
VOLUME_NAME = "onedm-weights"
WEIGHTS_MOUNT = "/weights"
ONEDM_ROOT = "/opt/One-DM"

_THIS = Path(__file__).resolve()
_PARENTS = _THIS.parents
REPO_ROOT = _PARENTS[2] if len(_PARENTS) >= 3 else Path("/opt")
VENDOR_ONEDM = REPO_ROOT / "vendor" / "One-DM"
SAMPLER_LOCAL = _THIS.parent / "sampler.py"

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
        "omegaconf==2.3.0",
        "opencv-python-headless==4.10.0.84",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "safetensors==0.4.4",
        "huggingface-hub==0.24.6",
        "tqdm==4.66.5",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"PYTHONPATH": ONEDM_ROOT, "HF_HOME": "/tmp/hf"})
    .add_local_dir(str(VENDOR_ONEDM), remote_path=ONEDM_ROOT)
    .add_local_file(str(SAMPLER_LOCAL), remote_path="/opt/sampler.py")
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=["T4", "L4"],
    timeout=1800,
    scaledown_window=60,
    volumes={WEIGHTS_MOUNT: volume},
)
class OneDMService:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, ONEDM_ROOT)
        sys.path.insert(0, "/opt")
        os.chdir(ONEDM_ROOT)

        from sampler import OneDMSampler

        ckpt = f"{WEIGHTS_MOUNT}/One-DM-ckpt.pt"
        vae_root = f"{WEIGHTS_MOUNT}/stable-diffusion-v1-5"
        unifont = f"{ONEDM_ROOT}/data/unifont.pickle"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Missing {ckpt}. Run scripts/populate_modal_onedm_volume.py first."
            )
        if not os.path.isdir(os.path.join(vae_root, "vae")):
            raise FileNotFoundError(
                f"Missing VAE at {vae_root}/vae. Run scripts/populate_modal_onedm_volume.py first."
            )
        self.sampler = OneDMSampler(
            ckpt_path=ckpt,
            vae_path=vae_root,
            unifont_path=unifont,
            device="cuda",
        )

    @modal.method()
    def generate_lines(
        self,
        texts: List[str],
        style_png: bytes,
        steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[bytes]:
        return self.sampler.generate_lines(texts, style_png, steps=steps, seed=seed)


@app.local_entrypoint()
def main(text: str = "Pass", steps: int = 50):
    """Smoke: modal run services/modal_onedm/app.py --text 'Pass' """
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

    svc = OneDMService()
    pngs = svc.generate_lines.remote([text], style_png, steps=steps, seed=0)
    out = REPO_ROOT / "data" / "outputs" / "smoke_onedm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pngs[0])
    print(f"Wrote {out} ({len(pngs[0])} bytes)")
