"""Pure quality analysis for generated grayscale handwriting lines.

The Modal sampler and local API both load this file so retry and fallback
decisions use identical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


INK_SOFT_THRESH = 245
INK_DARK_THRESH = 200
INK_CORE_THRESH = 160
MIN_INK_DENSITY = 0.04
LEFT_INK_FRAC = 0.15
MIN_WIDTH_PER_CHAR = 10


@dataclass(frozen=True)
class LineQuality:
    ok: bool
    reason: str
    score: float
    metrics: Dict[str, float]


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _true_runs(values: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    runs = []
    start: Optional[int] = None
    for index, value in enumerate(values):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return tuple(runs)


def _longest_true_run(values: np.ndarray) -> int:
    return max((end - start for start, end in _true_runs(values)), default=0)


def _result(
    ok: bool,
    reason: str,
    metrics: Dict[str, float],
) -> LineQuality:
    expected_w = max(metrics.get("expected_w", 1.0), 1.0)
    ink_w = metrics.get("ink_w", 0.0)
    width_score = 50.0 * min(1.0, ink_w / expected_w)
    core_score = 50.0 * min(1.0, metrics.get("core_ratio", 0.0) / 0.35)
    density_score = 20.0 * min(1.0, metrics.get("dark_density", 0.0) / 0.05)
    thickness = metrics.get("stroke_p99", 0.0)
    dense_run = metrics.get("dense_run", 0.0)
    smear_penalty = max(0.0, thickness - 3.0) * 20.0 + max(0.0, dense_run - 4.0) * 2.0
    reason_penalty = {
        "smear": 300.0,
        "local_smear": 300.0,
        "faint": 180.0,
        "weak_glyph": 180.0,
        "weak_stroke": 180.0,
        "missing_glyphs": 180.0,
        "near_empty": 180.0,
        "no_ink": 200.0,
        "too_narrow": 140.0,
        "left_truncated": 120.0,
        "sparse_wide": 100.0,
    }.get(reason.split(" ", 1)[0], 0.0)
    score = width_score + core_score + density_score - smear_penalty - reason_penalty
    if ok:
        score += 1000.0
    return LineQuality(ok=ok, reason=reason, score=float(score), metrics=metrics)


def analyze_line(arr: np.ndarray, gen_text: str) -> LineQuality:
    """Analyze one generated line without modifying it.

    Faint detection compares softly visible ink with its dark core. Local-smear
    detection requires both an abnormally thick stroke and a sustained dense
    horizontal run, which avoids rejecting ordinary connected cursive writing.
    """
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = np.asarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    nchar = max(1, len((gen_text or "").replace(" ", "")))
    expected_w = float(MIN_WIDTH_PER_CHAR * nchar)

    soft = arr < INK_SOFT_THRESH
    dark = arr < INK_DARK_THRESH
    core = arr < INK_CORE_THRESH
    soft_count = int(soft.sum())
    dark_count = int(dark.sum())
    core_count = int(core.sum())
    area = max(1, h * w)
    soft_density = soft_count / area
    dark_density = dark_count / area
    core_ratio = core_count / max(1, soft_count)
    soft_bbox = _mask_bbox(soft)
    dark_bbox = _mask_bbox(dark)
    bbox = dark_bbox or soft_bbox
    ink_w = 0 if bbox is None else bbox[2] - bbox[0]

    visible_darkness = (255 - arr[soft]).astype(np.float32)
    darkness_p75 = (
        float(np.percentile(visible_darkness, 75)) if visible_darkness.size else 0.0
    )
    metrics: Dict[str, float] = {
        "height": float(h),
        "width": float(w),
        "nchar": float(nchar),
        "expected_w": expected_w,
        "ink_w": float(ink_w),
        "soft_density": float(soft_density),
        "dark_density": float(dark_density),
        "core_ratio": float(core_ratio),
        "darkness_p75": darkness_p75,
        "stroke_p99": 0.0,
        "dense_run": 0.0,
        "min_glyph_core_ratio": 1.0,
        "min_glyph_stroke": 0.0,
        "reference_glyph_stroke": 0.0,
        "significant_components": 0.0,
    }

    if soft_bbox is None or soft_density < 0.002:
        return _result(False, f"near_empty dens={soft_density:.4f}", metrics)

    # A visible line without enough genuinely dark pen core becomes translucent
    # after alpha conversion, even when its overall width and density look valid.
    if (
        darkness_p75 < 80.0
        and core_ratio < 0.08
    ):
        return _result(
            False,
            (
                f"faint core={core_ratio:.3f} "
                f"p75={darkness_p75:.1f} dens={dark_density:.4f}"
            ),
            metrics,
        )

    if dark_bbox is None or dark_density < 0.008:
        return _result(False, f"near_empty dens={dark_density:.4f}", metrics)

    if ink_w < max(24, MIN_WIDTH_PER_CHAR * min(nchar, 8) * 0.35):
        return _result(
            False, f"too_narrow ink_w={ink_w} nchar={nchar}", metrics
        )

    glyph_core_ratios = []
    glyph_strokes = []
    for gx0, gx1 in _true_runs(soft.any(axis=0)):
        glyph_soft = int(soft[:, gx0:gx1].sum())
        if gx1 - gx0 < 12 or glyph_soft < 100:
            continue
        glyph_core = int(core[:, gx0:gx1].sum())
        glyph_core_ratios.append(glyph_core / glyph_soft)
        glyph_distance = cv2.distanceTransform(
            dark[:, gx0:gx1].astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        glyph_positive = glyph_distance[glyph_distance > 0]
        if glyph_positive.size:
            glyph_strokes.append(float(np.percentile(glyph_positive, 50)))
    if glyph_core_ratios:
        min_glyph_core_ratio = float(min(glyph_core_ratios))
        metrics["min_glyph_core_ratio"] = min_glyph_core_ratio
        if min_glyph_core_ratio < 0.50:
            return _result(
                False,
                f"weak_glyph core={min_glyph_core_ratio:.3f}",
                metrics,
            )

    if len(glyph_strokes) >= 2:
        min_glyph_stroke = float(min(glyph_strokes))
        reference_glyph_stroke = float(max(glyph_strokes))
        metrics["min_glyph_stroke"] = min_glyph_stroke
        metrics["reference_glyph_stroke"] = reference_glyph_stroke
        if (
            min_glyph_stroke <= 1.4
            and reference_glyph_stroke >= 2.0
            and min_glyph_stroke / reference_glyph_stroke <= 0.72
        ):
            return _result(
                False,
                (
                    f"weak_stroke min={min_glyph_stroke:.2f} "
                    f"reference={reference_glyph_stroke:.2f}"
                ),
                metrics,
            )

    compact = [char for char in (gen_text or "") if not char.isspace()]
    alnum = [char for char in compact if char.isalnum()]
    digit_fraction = (
        sum(char.isdigit() for char in alnum) / len(alnum) if alnum else 0.0
    )
    if digit_fraction >= 0.50 and len(compact) <= 12:
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            dark.astype(np.uint8),
            connectivity=8,
        )
        significant = (
            int((stats[1:, cv2.CC_STAT_AREA] >= 8).sum()) if n_labels > 1 else 0
        )
        minimum_components = len(compact)
        metrics["significant_components"] = float(significant)
        if significant < minimum_components:
            return _result(
                False,
                (
                    f"missing_glyphs n={significant} "
                    f"expected={minimum_components}"
                ),
                metrics,
            )

    x0, y0, x1, y1 = dark_bbox
    crop = dark[y0:y1, x0:x1].astype(np.uint8)
    distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    positive_distance = distance[distance > 0]
    stroke_p99 = (
        float(np.percentile(positive_distance, 99))
        if positive_distance.size
        else 0.0
    )
    col_density = crop.mean(axis=0).astype(np.float32)
    active_cols = col_density[col_density > 0]
    active_median = float(np.median(active_cols)) if active_cols.size else 0.0
    dense_threshold = max(0.22, active_median * 2.0)
    dense_run = _longest_true_run(col_density >= dense_threshold)
    metrics["stroke_p99"] = stroke_p99
    metrics["dense_run"] = float(dense_run)
    metrics["active_col_median"] = active_median

    # A single VAE blob is locally thick and remains dense over several adjacent
    # columns. Requiring both properties preserves ordinary tall/connected glyphs.
    min_dense_run = max(10, int(round(h * 0.15)))
    min_blob_radius = max(8.0, h * 0.125)
    if stroke_p99 >= min_blob_radius and dense_run >= min_dense_run:
        return _result(
            False,
            (
                f"local_smear stroke={stroke_p99:.2f} "
                f"run={dense_run} ink_w={ink_w}"
            ),
            metrics,
        )

    if ink_w > max(400, expected_w * 8):
        full_col = dark.mean(axis=0).astype(np.float32)
        if float(full_col.std()) < 0.09 and float((full_col > 0.02).mean()) > 0.35:
            return _result(
                False, f"smear ink_w={ink_w} nchar={nchar}", metrics
            )

    if w >= 400 and dark_density < MIN_INK_DENSITY:
        left = dark[:, : max(1, int(w * LEFT_INK_FRAC))].sum()
        total = max(1, dark_count)
        if left / total > 0.85:
            return _result(
                False,
                f"left_truncated dens={dark_density:.4f} w={w}",
                metrics,
            )

    if w >= 700 and dark_density < 0.035:
        return _result(
            False, f"sparse_wide dens={dark_density:.4f} w={w}", metrics
        )

    return _result(True, "ok", metrics)


def qa_line_ok(arr: np.ndarray, gen_text: str) -> Tuple[bool, str]:
    result = analyze_line(arr, gen_text)
    return result.ok, result.reason


def candidate_quality_key(arr: np.ndarray, gen_text: str) -> Tuple[int, float]:
    """Ordering key where every valid candidate outranks every invalid one."""
    result = analyze_line(arr, gen_text)
    return int(result.ok), result.score
