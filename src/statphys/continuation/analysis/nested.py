"""Nested quenched/annealed uncertainty and finite-size summaries."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Any

import numpy as np

T95_DF4 = 2.7764451051977987


def five_seed_interval(values: Iterable[float]) -> dict[str, float]:
    sample = np.asarray(list(values), dtype=float)
    if sample.shape != (5,) or not np.isfinite(sample).all():
        raise ValueError("a confirmatory interval requires exactly five finite outer-seed values")
    mean = float(np.mean(sample))
    standard_deviation = float(np.std(sample, ddof=1))
    standard_error = standard_deviation / math.sqrt(5.0)
    return {
        "mean": mean,
        "std": standard_deviation,
        "se": standard_error,
        "ci95_low": mean - T95_DF4 * standard_error,
        "ci95_high": mean + T95_DF4 * standard_error,
    }


def nested_variance(
    rows: Iterable[Mapping[str, Any]], value: str, outer: str = "seed", inner: str = "inner_seed"
) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row[outer]), []).append(float(row[value]))
    if len(grouped) != 5:
        raise ValueError("nested variance requires exactly five outer seeds")
    outer_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    within = np.asarray([np.var(grouped[key], ddof=1) if len(grouped[key]) > 1 else 0.0 for key in sorted(grouped)])
    interval = five_seed_interval(outer_means)
    interval.update(
        between_seed_variance=float(np.var(outer_means, ddof=1)),
        within_seed_variance=float(np.mean(within)),
        n_inner=float(sum(len(values) for values in grouped.values())),
    )
    return interval
