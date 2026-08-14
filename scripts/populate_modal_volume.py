#!/usr/bin/env python3
"""Upload DiffBrush-ckpt.pt and download SD1.5 VAE into Modal Volume diffbrush-weights."""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = REPO_ROOT / "DiffBrush-ckpt.pt"
VOLUME_NAME = "diffbrush-weights"
WEIGHTS_MOUNT = "/weights"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

populate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub==0.24.6",
        "diffusers==0.30.3",
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
)

app = modal.App("diffbrush-populate-weights", image=populate_image)


@app.function(volumes={WEIGHTS_MOUNT: volume}, timeout=1800)
def download_vae():
    """Pull SD1.5 VAE into the volume (CPU is fine)."""
    from diffusers import AutoencoderKL

    target = f"{WEIGHTS_MOUNT}/stable-diffusion-v1-5"
    os.makedirs(target, exist_ok=True)
    print("Downloading runwayml/stable-diffusion-v1-5 VAE…")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
    )
    vae.save_pretrained(f"{target}/vae")
    volume.commit()
    print("VAE saved to", f"{target}/vae")


@app.local_entrypoint()
def main():
    if not CKPT.exists():
        raise SystemExit(f"Missing checkpoint at {CKPT}")

    print(f"Uploading {CKPT} ({CKPT.stat().st_size / 1e9:.2f} GB) to volume {VOLUME_NAME}…")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(CKPT), "DiffBrush-ckpt.pt")
    print("Checkpoint uploaded.")

    print("Downloading VAE into volume…")
    download_vae.remote()
    print("Done. Volume ready.")
