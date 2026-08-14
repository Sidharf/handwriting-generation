# Handwriting Generation

Localhost UI + FastAPI pipeline that diffs **template** vs **synthetic** PDFs, generates styled handwriting with [Emuru](https://github.com/aimagelab/Emuru-autoregressive-text-img) ([CVPR 2025](https://arxiv.org/abs/2503.17074)) on **Modal GPUs (L4/A10G)**, and stamps **`#2563EB`** ink onto the template.

Emuru is an **autoregressive** T5 + VAE styled text-image model: style PNG + **style transcription** + generation text → clean ink PNG (any length, no 9-char chunking).

## Prerequisites

- Python 3.10+
- Node 18+
- Modal account
- Hugging Face access to download `blowing-up-groundhogs/emuru` (public)

## One-time setup

```bash
# Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Modal auth (opens browser)
python3 -m modal setup

# Download Emuru (+ VAE + ByT5 tokenizer) into Modal Volume `emuru-weights`
python scripts/populate_modal_emuru_volume.py

# Deploy the Modal service
modal deploy services/modal_emuru/app.py

# Optional: smoke generate short strings on Modal
python scripts/smoke_modal_emuru.py
# or: modal run services/modal_emuru/app.py --text "Pass" --style-text "An erratic some-CAPS"
```

## Run locally

Terminal 1 — API:

```bash
source .venv/bin/activate
./scripts/run_api.sh
```

Terminal 2 — UI:

```bash
cd apps/web
npm install
npm run dev
```

Open http://127.0.0.1:5173

After code changes to the Modal service, redeploy:

```bash
modal deploy services/modal_emuru/app.py
```

## Workflow

1. Pick a pair from `reference-pdfs/` or drop TEMPLATE + synthetic PDFs.
2. **Detect fills** — finds every blank→filled cell; edit/uncheck as needed.
3. Drag-drop a handwriting **PNG**, type the **exact transcription** of that sample (`style_text`), then upload. Set active style.
4. **Run batch** — Modal Emuru generates all lines, then stamps blue ink onto the template PDF.
5. Download the result when status is `done`.

## Layout

```
apps/web/                 Vite + React UI
services/api/             FastAPI (diff, styles, jobs, stamp)
services/modal_emuru/     Modal Emuru service
services/modal_onedm/     Legacy One-DM (unused; hard-cutover to Emuru)
vendor/One-DM/            Legacy vendored One-DM source
scripts/                  populate volume, smoke, run API
data/                     pairs, styles, outputs
```

## Notes

- Batch-per-document (not per-cell) to avoid Modal cold-start multiplication.
- Emuru requires `style_text` (transcription of the style PNG) in addition to the image.
- Generation length is controlled by `max_new_tokens` (default 128).
- Text is sanitized to printable ASCII (unicode punctuation normalized).
- Legacy One-DM Modal app / volume are no longer on the live job path.
