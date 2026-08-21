from __future__ import annotations

import io

import fitz
import numpy as np
from PIL import Image

from pdf_stamp import (
    compute_stamp_rect,
    ensure_minimum_ink_visibility,
    stamp_pdf,
)
from variation import resolve_variation


def test_compute_stamp_rect_contains_wide_images_for_all_jitter() -> None:
    target = fitz.Rect(10, 20, 90, 42)
    axes = resolve_variation(
        {
            "master": 100,
            "diversity": 100,
            "size": 100,
            "placement": 100,
            "stroke": 100,
        }
    )

    for seed in range(20):
        place = compute_stamp_rect(target, 900, 64, 0.97, axes, seed, seed)
        assert place.x0 >= target.x0
        assert place.y0 >= target.y0
        assert place.x1 <= target.x1
        assert place.y1 <= target.y1
        assert place.width <= target.width
        assert place.height <= target.height


def test_visibility_guard_only_strengthens_low_opacity_hairlines() -> None:
    faint = np.zeros((24, 80, 4), dtype=np.uint8)
    faint[:, :, :3] = (37, 99, 235)
    faint[6:18, 20:22, 3] = 70
    faint_im = Image.fromarray(faint)

    strengthened = np.array(ensure_minimum_ink_visibility(faint_im))

    assert strengthened[:, :, 3].max() >= 180
    assert int((strengthened[:, :, 3] > 10).sum()) > int(
        (faint[:, :, 3] > 10).sum()
    )

    opaque = faint.copy()
    opaque[6:18, 20:22, 3] = 220
    opaque_im = Image.fromarray(opaque)
    unchanged = np.array(ensure_minimum_ink_visibility(opaque_im))
    assert np.array_equal(unchanged, opaque)


def test_stamp_clamps_stale_bbox_to_vector_table_rule(tmp_path) -> None:
    template = tmp_path / "template.pdf"
    output = tmp_path / "output.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=120)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(20, 20, 160, 70))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    doc.save(template)
    doc.close()

    gray = np.full((64, 500), 255, dtype=np.uint8)
    gray[18:45, 8:492] = 25
    line = Image.fromarray(gray)
    line_buf = io.BytesIO()
    line.save(line_buf, format="PNG")

    stamp_pdf(
        template,
        [
            {
                "page": 0,
                "bbox": [100, 30, 190, 52],
                "text": "KL 21 JUL 26",
                "enabled": True,
                "kind": "text",
            }
        ],
        [line_buf.getvalue()],
        output,
        variation={
            "master": 100,
            "diversity": 100,
            "size": 100,
            "placement": 100,
            "stroke": 100,
        },
        seed=7,
    )

    rendered = fitz.open(output)
    pix = rendered[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, :3]
    rendered.close()
    rgb_i = rgb.astype(np.int16)
    blue = (
        (rgb_i[:, :, 2] > 140)
        & (rgb_i[:, :, 2] > rgb_i[:, :, 0] + 40)
        & (rgb_i[:, :, 2] > rgb_i[:, :, 1] + 30)
    )

    assert blue.any()
    assert not blue[:, int(160 * 4) :].any()
