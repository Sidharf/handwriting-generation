"""Synthesize CMC-style handwriting PNG samples via OpenAI."""

from __future__ import annotations

import base64
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

STYLE_TEXT = "biotech is cool"
NUM_SAMPLES = 4
IMAGE_SIZE = "1024x1024"

# Images API models (when the project has image access)
IMAGE_MODELS = ("gpt-image-1", "gpt-image-1-mini", "dall-e-3", "dall-e-2")
# Chat/Responses models that may expose the image_generation tool
RESPONSE_MODELS = ("gpt-5.4", "gpt-5.5", "gpt-5-chat-latest", "gpt-5-pro")

_FLAVORS = (
    "slightly upright block print",
    "soft natural print with tiny irregularities",
    "neat QA form print, modest right slant",
    "clean medium ballpoint block letters",
)


def _prompt_for(index: int) -> str:
    flavor = _FLAVORS[index % len(_FLAVORS)]
    return (
        "Photorealistic close-up of a single line of neat pharmaceutical CMC / QA "
        "employee handwriting on plain white paper. Medium black ballpoint or felt-tip "
        f"pen, even stroke weight (not thin), {flavor}. "
        "No second line, no UI chrome, no shadows, no logos, no ruled paper. "
        f'Write exactly this text and nothing else: "{STYLE_TEXT}". '
        "Centered horizontally, generous white margins, high contrast black ink."
    )


def openai_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _split_override() -> tuple[str, Sequence[str]]:
    """Return ('images'|'responses', models) from OPENAI_IMAGE_MODEL if set."""
    override = (os.environ.get("OPENAI_IMAGE_MODEL") or "").strip()
    if not override:
        return "", ()
    if override.startswith("gpt-5") or override.endswith("-chat-latest"):
        return "responses", (override,)
    return "images", (override,)


def _extract_b64_from_image_item(item: Any) -> Optional[str]:
    b64 = getattr(item, "b64_json", None)
    if b64:
        return b64
    url = getattr(item, "url", None)
    if not url:
        return None
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as resp:
        return base64.b64encode(resp.read()).decode("ascii")


def _generate_via_images_api(client: Any, model: str, prompt: str) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": IMAGE_SIZE,
        "n": 1,
    }
    if model.startswith("dall-e"):
        try:
            result = client.images.generate(**kwargs, response_format="b64_json")
        except Exception as e:
            if "response_format" in str(e):
                result = client.images.generate(**kwargs)
            else:
                raise
    else:
        result = client.images.generate(**kwargs)
    b64 = _extract_b64_from_image_item(result.data[0])
    if not b64:
        raise RuntimeError(f"Images API model {model} returned no PNG data")
    return b64


def _generate_via_responses_api(client: Any, model: str, prompt: str) -> str:
    """Use Responses API + image_generation tool (for chat-only projects)."""
    result = client.responses.create(
        model=model,
        input=(
            "Create exactly one image (no prose). "
            + prompt
        ),
        tools=[{"type": "image_generation"}],
    )
    out = getattr(result, "output", None) or []
    for item in out:
        if getattr(item, "type", None) != "image_generation_call":
            continue
        res = getattr(item, "result", None)
        if isinstance(res, str) and len(res) > 100:
            return res
        # Some SDK shapes nest b64 under result dict
        if isinstance(res, dict):
            b64 = res.get("b64_json") or res.get("image_base64")
            if b64:
                return b64
    # Fallback: scan serialized output for large base64-looking payloads
    raise RuntimeError(f"Responses model {model} returned no image_generation result")


def _generate_one(index: int) -> Dict[str, Any]:
    from openai import OpenAI

    key = openai_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=key, timeout=180.0)
    prompt = _prompt_for(index)
    errors: List[str] = []

    kind, override_models = _split_override()
    image_models: Sequence[str] = override_models if kind == "images" else IMAGE_MODELS
    response_models: Sequence[str] = (
        override_models if kind == "responses" else RESPONSE_MODELS
    )

    if kind != "responses":
        for model in image_models:
            try:
                b64 = _generate_via_images_api(client, model, prompt)
                return {
                    "id": f"synth-{uuid.uuid4().hex[:8]}-{index}",
                    "png_base64": b64,
                    "style_text": STYLE_TEXT,
                    "index": index,
                    "model": model,
                }
            except Exception as e:
                errors.append(f"images/{model}: {e}")

    if kind != "images":
        for model in response_models:
            try:
                b64 = _generate_via_responses_api(client, model, prompt)
                return {
                    "id": f"synth-{uuid.uuid4().hex[:8]}-{index}",
                    "png_base64": b64,
                    "style_text": STYLE_TEXT,
                    "index": index,
                    "model": model,
                }
            except Exception as e:
                errors.append(f"responses/{model}: {e}")

    raise RuntimeError(
        "No usable OpenAI image path for this API key/project. "
        "Enable Images (gpt-image-1) or image_generation on a gpt-5 model, "
        "or set OPENAI_IMAGE_MODEL. Tried: " + " | ".join(errors)
    )


def _demo_png_b64(index: int) -> str:
    """Local placeholder handwriting PNG when OPENAI_SYNTH_DEMO=1."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    w, h = 1024, 256
    im = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for candidate in (
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 64)
            break
        except Exception:
            continue
    x = 48 + index * 7
    y = 90 + (index % 3) * 4
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        draw.text((x + dx, y + dy), STYLE_TEXT, fill=(20, 20, 20), font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def synthesize_handwriting_samples(count: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Run `count` OpenAI image generations in parallel; return list of sample dicts."""
    n = max(1, min(int(count), 8))

    # Explicit local demo path (no OpenAI Images) for UI/Emuru wiring tests
    if (os.environ.get("OPENAI_SYNTH_DEMO") or "").strip() in ("1", "true", "yes"):
        return [
            {
                "id": f"synth-demo-{uuid.uuid4().hex[:8]}-{i}",
                "png_base64": _demo_png_b64(i),
                "style_text": STYLE_TEXT,
                "index": i,
                "model": "local-demo",
            }
            for i in range(n)
        ]

    if not openai_api_key():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or add it to a gitignored .env "
            "at the repo root, then restart the API."
        )
    out: List[Optional[Dict[str, Any]]] = [None] * n
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_generate_one, i): i for i in range(n)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                errors.append(f"sample {i}: {e}")

    if errors:
        raise RuntimeError("OpenAI synthesize failed: " + " | ".join(errors))
    return [x for x in out if x is not None]
