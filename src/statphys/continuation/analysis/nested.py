"""Nested quenched/annealed uncertainty and finite-size summaries."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Any

import numpy as np
from scipy.stats import t as student_t

T95_DF4 = 2.7764451051977987


def registered_seed_interval(values: Iterable[float]) -> dict[str, float]:
    sample = np.asarray(list(values), dtype=float)
    if sample.ndim != 1 or sample.size < 5 or not np.isfinite(sample).all():
        raise ValueError("a confirmatory interval requires at least five finite outer-seed values")
    mean = float(np.mean(sample))
    standard_deviation = float(np.std(sample, ddof=1))
    standard_error = standard_deviation / math.sqrt(sample.size)
    critical = float(student_t.ppf(0.975, df=sample.size - 1))
    return {
        "mean": mean,
        "std": standard_deviation,
        "se": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "n_outer": float(sample.size),
    }


def five_seed_interval(values: Iterable[float]) -> dict[str, float]:
    """Backward-compatible name for an interval over at least five seeds."""
    return registered_seed_interval(values)


def nested_variance(
    rows: Iterable[Mapping[str, Any]], value: str, outer: str = "seed", inner: str = "inner_seed"
) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row[outer]), []).append(float(row[value]))
    if len(grouped) < 5:
        raise ValueError("nested variance requires at least five outer seeds")
    outer_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    within = np.asarray([np.var(grouped[key], ddof=1) if len(grouped[key]) > 1 else 0.0 for key in sorted(grouped)])
    interval = registered_seed_interval(outer_means)
    interval.update(
        between_seed_variance=float(np.var(outer_means, ddof=1)),
        within_seed_variance=float(np.mean(within)),
        n_inner=float(sum(len(values) for values in grouped.values())),
    )
    return interval
