"""Finite-size diagnostics shared by all phase-continuation domains."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def susceptibility_peaks(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["size"])].append(row)
    peaks = []
    for size, items in sorted(grouped.items()):
        best = max(items, key=lambda item: float(item["susceptibility_mean"]))
        peaks.append(
            {
                "size": float(size),
                "critical_control": float(best["control"]),
                "susceptibility_peak": float(best["susceptibility_mean"]),
            }
        )
    return peaks


def peak_growth_exponent(peaks: Iterable[Mapping[str, float]]) -> float:
    points = list(peaks)
    if len(points) < 3:
        raise ValueError("at least three sizes are required")
    sizes = np.asarray([point["size"] for point in points], dtype=float)
    height = np.asarray([point["susceptibility_peak"] for point in points], dtype=float)
    valid = (sizes > 0) & (height > 0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("peak fit requires three positive points")
    return float(np.polyfit(np.log(sizes[valid]), np.log(height[valid]), 1)[0])


def binder_crossing_spread(rows: Iterable[Mapping[str, Any]]) -> float:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["size"])].append(row)
    crossings = []
    sizes = sorted(grouped)
    for left_size, right_size in zip(sizes[:-1], sizes[1:], strict=True):
        left = {float(row["control"]): float(row["binder_cumulant_mean"]) for row in grouped[left_size]}
        right = {float(row["control"]): float(row["binder_cumulant_mean"]) for row in grouped[right_size]}
        controls = sorted(set(left) & set(right))
        if not controls:
            continue
        crossings.append(min(controls, key=lambda control: abs(left[control] - right[control])))
    return float(np.std(crossings)) if crossings else float("nan")
