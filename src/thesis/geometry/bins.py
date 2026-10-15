"""Locked signed-distance intervals and log-uniform targets (D09--D13)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import cache

from thesis.config import load_config

MIN_ABS_DISTANCE_M = 0.3


@dataclass(frozen=True, slots=True)
class DistanceBin:
    bin_id: int
    lower_m: float
    upper_m: float
    include_upper: bool = False

    def contains(self, value: float) -> bool:
        upper_ok = value <= self.upper_m if self.include_upper else value < self.upper_m
        return self.lower_m <= value and upper_ok and abs(value) >= MIN_ABS_DISTANCE_M

    @property
    def abs_bounds_m(self) -> tuple[float, float]:
        return tuple(sorted((abs(self.lower_m), abs(self.upper_m))))


_SQRT_3 = math.sqrt(3)
_SQRT_15 = math.sqrt(15)
_SQRT_50 = math.sqrt(50)
_SQRT_200 = math.sqrt(200)
_SQRT_800 = math.sqrt(800)
_SQRT_3200 = math.sqrt(3200)

_DISTANCE_BINS = (
    DistanceBin(-5, -5.0, -_SQRT_15),
    DistanceBin(-3, -_SQRT_15, -_SQRT_3),
    DistanceBin(-1, -_SQRT_3, -MIN_ABS_DISTANCE_M, include_upper=True),
    DistanceBin(1, MIN_ABS_DISTANCE_M, _SQRT_3),
    DistanceBin(3, _SQRT_3, _SQRT_15),
    DistanceBin(5, _SQRT_15, _SQRT_50),
    DistanceBin(10, _SQRT_50, _SQRT_200),
    DistanceBin(20, _SQRT_200, _SQRT_800),
    DistanceBin(40, _SQRT_800, _SQRT_3200),
    DistanceBin(80, _SQRT_3200, 80.0, include_upper=True),
)


def distance_bins() -> tuple[DistanceBin, ...]:
    return _DISTANCE_BINS


def sample_log_uniform_target(distance_bin: DistanceBin, rng: random.Random) -> float:
    low_m, high_m = distance_bin.abs_bounds_m
    magnitude = math.exp(rng.uniform(math.log(low_m), math.log(high_m)))
    return -magnitude if distance_bin.bin_id < 0 else magnitude


@cache
def _configured_repeats() -> dict[int, int]:
    return {int(entry["id"]): int(entry["repeats"]) for entry in load_config("bins")["bins"]}


def repeats_for_bin(bin_id: int) -> int:
    if bin_id not in {distance_bin.bin_id for distance_bin in _DISTANCE_BINS}:
        raise ValueError(f"unknown distance bin: {bin_id}")
    return _configured_repeats()[bin_id]
