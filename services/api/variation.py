"""Cell-to-cell handwriting variation: resolve axes + deterministic jitter maps."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


AXES = ("diversity", "size", "placement", "stroke")
DEFAULT_MASTER = 35


def _clamp100(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(100.0, x))


@dataclass(frozen=True)
class VariationAxes:
    master: float
    diversity: float
    size: float
    placement: float
    stroke: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "master": self.master,
            "diversity": self.diversity,
            "size": self.size,
            "placement": self.placement,
            "stroke": self.stroke,
        }


def resolve_variation(raw: Optional[Mapping[str, Any]]) -> VariationAxes:
    """Resolve master + optional per-axis overrides into concrete 0..100 values.

    Missing / empty payload → all zeros (legacy stamp/gen identity).
    Missing individual axes track master.
    """
    if not raw:
        return VariationAxes(master=0.0, diversity=0.0, size=0.0, placement=0.0, stroke=0.0)

    master = _clamp100(raw.get("master", 0), 0.0)
    vals: Dict[str, float] = {}
    for axis in AXES:
        if axis in raw and raw[axis] is not None and raw[axis] != "":
            vals[axis] = _clamp100(raw[axis], master)
        else:
            vals[axis] = master
    return VariationAxes(master=master, **vals)


def rng_unit(seed: Optional[int], cell_index: int, channel: int) -> float:
    """Deterministic float in [-1, 1] from (seed, cell, channel)."""
    base = 0 if seed is None else int(seed)
    rng = random.Random((base + 1) * 1_000_003 + cell_index * 97 + channel * 13)
    return rng.random() * 2.0 - 1.0


def seed_stride(diversity: float) -> int:
    """Per-line seed stride. diversity=0 → 17 (legacy)."""
    d = _clamp100(diversity)
    return int(17 + round(0.8 * d))


def thicken_iterations(stroke: float) -> int:
    """0=off, 1=default, 2=stronger. stroke=0 → 1 (legacy identity)."""
    s = _clamp100(stroke)
    if s <= 0:
        return 1
    if s < 40:
        return 1
    if s < 75:
        return 1
    return 2


def height_fit_frac(size: float, seed: Optional[int], cell_index: int) -> float:
    """Height fit vs cell. size=0 → 0.85 exact."""
    s = _clamp100(size)
    if s <= 0:
        return 0.85
    amp = 0.12 * (s / 100.0)
    return float(max(0.72, min(0.97, 0.85 + amp * rng_unit(seed, cell_index, 1))))


def placement_offsets(
    placement: float,
    seed: Optional[int],
    cell_index: int,
    leftover_x: float,
    leftover_y: float,
) -> tuple[float, float, float]:
    """Return (dx, dy, rotation_deg). placement=0 → (0, 0, 0) with caller centering."""
    p = _clamp100(placement)
    if p <= 0:
        return 0.0, 0.0, 0.0
    frac = 0.12 * (p / 100.0)
    dx = leftover_x * frac * rng_unit(seed, cell_index, 2)
    dy = leftover_y * frac * rng_unit(seed, cell_index, 3)
    rot = 3.0 * (p / 100.0) * rng_unit(seed, cell_index, 4)
    return float(dx), float(dy), float(rot)


def stroke_ink_params(
    stroke: float, seed: Optional[int], cell_index: int
) -> tuple[float, float, float]:
    """Return (alpha_floor, darkness_power, alpha_scale). stroke=0 → (0.05, 0.85, 1.0)."""
    s = _clamp100(stroke)
    if s <= 0:
        return 0.05, 0.85, 1.0
    power = 0.85 + 0.15 * (s / 100.0) * rng_unit(seed, cell_index, 5)
    power = float(max(0.70, min(1.0, power)))
    alpha_scale = 1.0 + 0.12 * (s / 100.0) * rng_unit(seed, cell_index, 6)
    alpha_scale = float(max(0.88, min(1.12, alpha_scale)))
    floor = max(0.04, 0.05 - 0.01 * (s / 100.0))
    return float(floor), power, alpha_scale


def line_weight_params(line_weight: int) -> tuple[int, int]:
    """Map UI line_weight (-2..+2) to (thicken_for_modal, thin_for_stamp).

    Negative values skip Modal dilation and apply local erosion instead.
    Zero preserves current default behavior (thicken=1, no erosion).
    Positive values increase dilation strength.
    """
    lw = max(-2, min(2, int(line_weight)))
    if lw < 0:
        return 0, abs(lw)
    if lw == 0:
        return 1, 0
    return lw, 0


def band_label(master: float) -> str:
    m = _clamp100(master)
    if m <= 0:
        return "None"
    if m < 30:
        return "Subtle"
    if m < 65:
        return "Natural"
    return "Wild"
