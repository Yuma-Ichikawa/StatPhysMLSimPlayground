"""Exploratory distribution shifts and change-point candidate generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array


def adjacent_js_divergence(
    probabilities: ArrayLike,
    *,
    base: float = np.e,
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute Jensen--Shannon divergence between adjacent distributions.

    Rows are normalized internally. Natural-log divergence lies in
    ``[0, log(2)]``; choose ``base=2`` for the conventional ``[0, 1]`` scale.
    """

    distributions = as_float_array(probabilities, name="probabilities", ndim=2)
    if distributions.shape[0] < 2 or distributions.shape[1] == 0:
        raise ValueError("at least two non-empty distributions are required")
    if np.any(distributions < 0):
        raise ValueError("probabilities must be non-negative")
    if base <= 0 or np.isclose(base, 1.0):
        raise ValueError("logarithm base must be positive and different from one")
    totals = distributions.sum(axis=1, keepdims=True)
    if np.any(totals <= eps):
        raise ValueError("each distribution must have positive mass")
    normalized = distributions / totals
    divergences = np.empty(normalized.shape[0] - 1, dtype=float)
    log_base = np.log(base)
    for index in range(divergences.size):
        first, second = normalized[index], normalized[index + 1]
        mixture = 0.5 * (first + second)

        def kl(left: np.ndarray) -> float:
            positive = left > 0
            return float(np.sum(left[positive] * np.log(left[positive] / mixture[positive])))

        divergences[index] = max(0.0, 0.5 * (kl(first) + kl(second)) / log_base)
    return divergences


def adjacent_histogram_js(
    samples_by_control: Sequence[ArrayLike],
    *,
    bins: int | ArrayLike = 51,
    value_range: tuple[float, float] | None = None,
    base: float = np.e,
) -> dict[str, Any]:
    """Histogram multiple controls on shared edges before adjacent JS analysis."""

    samples = [as_float_array(value, name=f"samples_by_control[{index}]").ravel() for index, value in enumerate(samples_by_control)]
    if len(samples) < 2 or any(sample.size == 0 for sample in samples):
        raise ValueError("at least two non-empty sample groups are required")
    if isinstance(bins, int):
        if bins < 1:
            raise ValueError("bins must be positive")
        if value_range is None:
            lower = min(float(sample.min()) for sample in samples)
            upper = max(float(sample.max()) for sample in samples)
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            value_range = (lower, upper)
        edges = np.linspace(value_range[0], value_range[1], bins + 1)
    else:
        edges = as_float_array(bins, name="bins", ndim=1)
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be strictly increasing")
    counts = np.vstack([np.histogram(sample, bins=edges)[0] for sample in samples])
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return {
        "divergence": adjacent_js_divergence(probabilities, base=base),
        "counts": counts,
        "probabilities": probabilities,
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
    }


def change_point_candidates(
    control: ArrayLike,
    observables: ArrayLike,
    *,
    z_threshold: float = 3.0,
    top_k: int | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Rank adjacent control intervals by standardized multivariate jumps.

    This is a discovery diagnostic, not a transition test. Features are scaled
    by their across-control standard deviation; interval jump norms are then
    robustly standardized by median/MAD. Returned candidates should be verified
    on an independently refined control grid.
    """

    x = as_float_array(control, name="control", ndim=1)
    values = as_float_array(observables, name="observables")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != x.size:
        raise ValueError("observables must have shape (n_control, n_features)")
    if x.size < 2:
        raise ValueError("at least two control points are required")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    order = np.argsort(x)
    x, values = x[order], values[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("control values must be unique")
    feature_scale = np.std(values, axis=0)
    feature_scale = np.where(feature_scale > eps, feature_scale, 1.0)
    jumps = np.diff(values, axis=0) / feature_scale
    interval_width = np.diff(x)
    scores = np.linalg.norm(jumps, axis=1) / np.sqrt(values.shape[1])
    # We score discontinuities rather than derivatives: width is retained for
    # reporting but does not reward an irregularly dense local control grid.
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad > eps:
        robust_z = (scores - median) / (1.4826 * mad)
    else:
        spread = float(np.std(scores))
        robust_z = (scores - median) / spread if spread > eps else np.zeros_like(scores)
    ranked = np.argsort(scores)[::-1]
    selected = [int(index) for index in ranked if robust_z[index] >= z_threshold]
    if top_k is not None:
        selected = selected[:top_k]
    candidates = [
        {
            "interval_index": index,
            "control_left": float(x[index]),
            "control_right": float(x[index + 1]),
            "control_midpoint": float(0.5 * (x[index] + x[index + 1])),
            "interval_width": float(interval_width[index]),
            "jump_score": float(scores[index]),
            "robust_z": float(robust_z[index]),
        }
        for index in selected
    ]
    return {
        "candidates": candidates,
        "jump_scores": scores,
        "robust_z": robust_z,
        "control_midpoints": 0.5 * (x[:-1] + x[1:]),
        "feature_scale": feature_scale,
        "z_threshold": float(z_threshold),
    }

