"""Finite-size diagnostics shared by all phase-continuation domains."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def susceptibility_peaks(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["size"])].append(row)
    peaks = []
    for size, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: float(item["control"]))
        controls = np.asarray([float(item["control"]) for item in ordered], dtype=float)
        values = np.asarray([float(item["susceptibility_mean"]) for item in ordered], dtype=float)
        index = int(np.argmax(values))
        if index == 0:
            location = float(controls[index])
            status = "left_censored"
        elif index == len(controls) - 1:
            location = float(controls[index])
            status = "right_censored"
        else:
            quadratic, linear, _ = np.polyfit(
                controls[index - 1 : index + 2], values[index - 1 : index + 2], 2
            )
            if quadratic < 0:
                location = float(
                    np.clip(-linear / (2.0 * quadratic), controls[index - 1], controls[index + 1])
                )
                status = "crossed"
            else:
                location = float(controls[index])
                status = "unresolved"
        peaks.append(
            {
                "size": float(size),
                "critical_control": location,
                "susceptibility_peak": float(values[index]),
                "status": status,
                "grid_index": index,
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


def binder_crossings(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["size"])].append(row)
    crossings: list[dict[str, float]] = []
    sizes = sorted(grouped)
    for left_size, right_size in zip(sizes[:-1], sizes[1:], strict=True):
        left = {
            float(row["control"]): float(row["binder_cumulant_mean"]) for row in grouped[left_size]
        }
        right = {
            float(row["control"]): float(row["binder_cumulant_mean"]) for row in grouped[right_size]
        }
        controls = sorted(set(left) & set(right))
        if not controls:
            continue
        differences = np.asarray([left[control] - right[control] for control in controls])
        for index in range(len(controls) - 1):
            left_difference = float(differences[index])
            right_difference = float(differences[index + 1])
            if left_difference == 0.0:
                location = controls[index]
            elif left_difference * right_difference < 0.0:
                fraction = -left_difference / (right_difference - left_difference)
                location = controls[index] + fraction * (controls[index + 1] - controls[index])
            else:
                continue
            crossings.append(
                {
                    "size_a": float(left_size),
                    "size_b": float(right_size),
                    "control": float(location),
                }
            )
    return crossings


def binder_crossing_spread(rows: Iterable[Mapping[str, Any]]) -> float:
    locations = np.asarray([item["control"] for item in binder_crossings(rows)], dtype=float)
    return float(np.std(locations)) if locations.size else float("nan")
