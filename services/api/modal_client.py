"""Thin client that calls the deployed Modal EmuruService (parallel GPU chunks)."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import modal

APP_NAME = "handwriting-emuru"
CLS_NAME = "EmuruService"
# Starter plan GPU concurrency
MAX_GPU_WORKERS = 10


def _chunk_texts(texts: List[str], max_workers: int) -> List[Tuple[int, List[str]]]:
    """Return (start_index, chunk) pairs covering texts in order."""
    n = len(texts)
    if n == 0:
        return []
    workers = max(1, min(max_workers, n))
    chunk_size = int(math.ceil(n / workers))
    chunks: List[Tuple[int, List[str]]] = []
    for start in range(0, n, chunk_size):
        chunks.append((start, texts[start : start + chunk_size]))
    return chunks


def generate_lines_remote(
    texts: List[str],
    style_png: bytes,
    style_text: str,
    max_new_tokens: int = 128,
    seed: Optional[int] = None,
    max_workers: int = MAX_GPU_WORKERS,
    seed_stride: int = 17,
    thicken: int = 1,
) -> List[bytes]:
    """Invoke EmuruService across up to max_workers GPU containers in parallel."""
    if not texts:
        return []
    if not (style_text or "").strip():
        raise ValueError("style_text is required for Emuru generation")

    EmuruService = modal.Cls.from_name(APP_NAME, CLS_NAME)
    svc = EmuruService()
    chunks = _chunk_texts(texts, max_workers)
    stride = max(1, int(seed_stride))
    thick = max(0, int(thicken))

    def _call(start: int, chunk: List[str]) -> List[bytes]:
        # Global line i uses seed + i * stride (start_index makes chunks consistent)
        return svc.generate_lines.remote(
            chunk,
            style_png,
            style_text,
            max_new_tokens=max_new_tokens,
            seed=seed,
            seed_stride=stride,
            thicken=thick,
            start_index=start,
        )

    if len(chunks) == 1:
        start, chunk = chunks[0]
        return _call(start, chunk)

    results: List[Optional[List[bytes]]] = [None] * len(chunks)

    def _run(idx: int, start: int, chunk: List[str]) -> Tuple[int, List[bytes]]:
        return idx, _call(start, chunk)

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futs = [
            pool.submit(_run, i, start, chunk)
            for i, (start, chunk) in enumerate(chunks)
        ]
        for fut in as_completed(futs):
            idx, pngs = fut.result()
            results[idx] = pngs

    out: List[bytes] = []
    for part in results:
        if part is None:
            raise RuntimeError("Modal chunk returned no result")
        out.extend(part)
    if len(out) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} PNGs from Modal, got {len(out)}"
        )
    return out
