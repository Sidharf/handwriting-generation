#!/usr/bin/env python3
"""Download Emuru weights from Hugging Face into Modal Volume emuru-weights."""

from __future__ import annotations

import json
import os
from glob import glob

import modal

VOLUME_NAME = "emuru-weights"
WEIGHTS_MOUNT = "/weights"
HF_REPO = "blowing-up-groundhogs/emuru"
HF_VAE_REPO = "blowing-up-groundhogs/emuru_vae"
HF_TOKENIZER = "google/byt5-small"
HF_T5_CONFIG = "google-t5/t5-large"

# Remote-code modules required for trust_remote_code offline load
REMOTE_CODE_FILES = (
    "configuration_emuru.py",
    "modeling_emuru.py",
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

populate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub==0.24.6",
        "tqdm==4.66.5",
    )
)

app = modal.App("emuru-populate-weights", image=populate_image)


def _assert_emuru_snapshot(emuru_dir: str) -> None:
    missing = [
        name
        for name in ("config.json", *REMOTE_CODE_FILES)
        if not os.path.isfile(os.path.join(emuru_dir, name))
    ]
    weights = (
        glob(os.path.join(emuru_dir, "*.safetensors"))
        + glob(os.path.join(emuru_dir, "*.bin"))
        + glob(os.path.join(emuru_dir, "pytorch_model*.bin"))
        + glob(os.path.join(emuru_dir, "model.safetensors*"))
    )
    listing = sorted(os.listdir(emuru_dir)) if os.path.isdir(emuru_dir) else []
    print(f"/weights/emuru contents ({len(listing)}): {listing[:40]}")
    if missing:
        raise FileNotFoundError(
            f"Emuru snapshot at {emuru_dir} missing required files: {missing}. "
            f"Directory listing: {listing}"
        )
    if not weights:
        raise FileNotFoundError(
            f"Emuru snapshot at {emuru_dir} has no weight files "
            f"(*.safetensors / *.bin). Listing: {listing}"
        )
    print(f"Emuru snapshot OK — remote code + {len(weights)} weight file(s)")


@app.function(volumes={WEIGHTS_MOUNT: volume}, timeout=7200)
def download_emuru():
    """Snapshot Emuru + VAE + tokenizer/T5 config into the volume (offline-ready)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    def pull(repo_id: str, dest: str) -> str:
        os.makedirs(dest, exist_ok=True)
        print(f"Downloading {repo_id} → {dest} …")
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
        return dest

    emuru_dir = pull(HF_REPO, f"{WEIGHTS_MOUNT}/emuru")

    # Explicitly ensure remote-code modules are present (offline trust_remote_code).
    for fname in REMOTE_CODE_FILES:
        dest = os.path.join(emuru_dir, fname)
        if not os.path.isfile(dest):
            print(f"Missing {fname} after snapshot; pulling explicitly…")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                local_dir=emuru_dir,
                local_dir_use_symlinks=False,
            )

    _assert_emuru_snapshot(emuru_dir)

    vae_dir = pull(HF_VAE_REPO, f"{WEIGHTS_MOUNT}/emuru_vae")
    tok_dir = pull(HF_TOKENIZER, f"{WEIGHTS_MOUNT}/byt5-small")

    # Emuru only needs T5Config from t5-large (weights live in the Emuru checkpoint).
    t5_dir = f"{WEIGHTS_MOUNT}/t5-large"
    os.makedirs(t5_dir, exist_ok=True)
    print(f"Downloading {HF_T5_CONFIG} config.json → {t5_dir} …")
    hf_hub_download(
        repo_id=HF_T5_CONFIG,
        filename="config.json",
        local_dir=t5_dir,
        local_dir_use_symlinks=False,
    )

    config_path = os.path.join(emuru_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Force all nested from_pretrained calls to local absolute paths.
    cfg["vae_name_or_path"] = vae_dir
    cfg["tokenizer_name_or_path"] = tok_dir
    cfg["t5_name_or_path"] = t5_dir
    # Prevent transformers dynamic-module resolution from falling back to the Hub id.
    cfg["_name_or_path"] = emuru_dir
    # Hub ships auto_map as "repo--module.Class" which forces a Hub lookup even for
    # local dirs. Rewrite to plain module refs so offline trust_remote_code works.
    cfg["auto_map"] = {
        "AutoConfig": "configuration_emuru.EmuruConfig",
        "AutoModel": "modeling_emuru.Emuru",
    }

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Patched config.json:")
    print("  _name_or_path =", cfg["_name_or_path"])
    print("  auto_map =", cfg["auto_map"])
    print("  vae_name_or_path =", cfg["vae_name_or_path"])
    print("  tokenizer_name_or_path =", cfg["tokenizer_name_or_path"])
    print("  t5_name_or_path =", cfg["t5_name_or_path"])

    volume.commit()
    print("Done. Volume ready at", WEIGHTS_MOUNT)


@app.local_entrypoint()
def main():
    print(f"Populating Modal volume {VOLUME_NAME} with Emuru weights…")
    download_emuru.remote()
    print("Done.")


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call(["modal", "run", __file__, *sys.argv[1:]]))
