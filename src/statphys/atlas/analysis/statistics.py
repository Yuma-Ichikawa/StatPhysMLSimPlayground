"""Nested uncertainty estimators and thermodynamic fluctuation statistics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array, scalar_or_array


def _ordered_nested_array(
    values: ArrayLike,
    teacher_axis: int,
    data_axis: int,
    optimizer_axis: int,
) -> np.ndarray:
    array = as_float_array(values, name="values")
    if array.ndim < 3:
        raise ValueError("values must contain teacher, data, and optimizer axes")
    normalized = tuple(axis % array.ndim for axis in (teacher_axis, data_axis, optimizer_axis))
    if len(set(normalized)) != 3:
        raise ValueError("teacher, data, and optimizer axes must be distinct")
    ordered = np.moveaxis(array, normalized, (0, 1, 2))
    if min(ordered.shape[:3]) == 0:
        raise ValueError("nested sampling axes must be non-empty")
    return ordered


def nested_estimator(
    values: ArrayLike,
    *,
    teacher_axis: int = 0,
    data_axis: int = 1,
    optimizer_axis: int = 2,
    ddof: int = 1,
) -> dict[str, Any]:
    """Average in the order optimizer → dataset → teacher.

    The outer teacher replicas are the independent units used for the reported
    standard error. ``variance_components`` are descriptive between-group
    variances rather than unbiased random-effects ANOVA estimates; their labels
    make this distinction explicit.
    """

    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    ordered = _ordered_nested_array(values, teacher_axis, data_axis, optimizer_axis)
    optimizer_means = ordered.mean(axis=2)
    teacher_means = optimizer_means.mean(axis=1)
    estimate = teacher_means.mean(axis=0)
    n_teacher, n_data, n_optimizer = ordered.shape[:3]

    def variance(array: np.ndarray, axis: int, count: int) -> np.ndarray:
        if count <= ddof:
            return np.full(array.shape[:axis] + array.shape[axis + 1 :], np.nan)
        return np.var(array, axis=axis, ddof=ddof)

    teacher_variance = variance(teacher_means, 0, n_teacher)
    data_variance = variance(optimizer_means, 1, n_data).mean(axis=0)
    optimizer_variance = variance(ordered, 2, n_optimizer).mean(axis=(0, 1))
    standard_error = (
        np.sqrt(teacher_variance / n_teacher)
        if n_teacher > ddof
        else np.full_like(np.asarray(estimate), np.nan, dtype=float)
    )
    return {
        "estimate": scalar_or_array(estimate),
        "standard_error": scalar_or_array(standard_error),
        "teacher_means": teacher_means,
        "dataset_means": optimizer_means,
        "variance_components": {
            "between_teacher_means": scalar_or_array(teacher_variance),
            "within_teacher_dataset_means": scalar_or_array(data_variance),
            "within_dataset_optimizer": scalar_or_array(optimizer_variance),
        },
        "counts": {
            "teacher": int(n_teacher),
            "dataset_per_teacher": int(n_data),
            "optimizer_per_dataset": int(n_optimizer),
        },
    }


def hierarchical_bootstrap(
    values: ArrayLike,
    *,
    teacher_axis: int = 0,
    data_axis: int = 1,
    optimizer_axis: int = 2,
    statistic: Callable[[np.ndarray], ArrayLike] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap every level of a balanced nested experiment.

    Teachers are sampled first; datasets are sampled independently within each
    selected teacher; optimizer seeds are then sampled within every selected
    teacher-dataset cell. The callable receives the resampled array in canonical
    ``(teacher, data, optimizer, ...)`` order. Its default is the nested mean.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    ordered = _ordered_nested_array(values, teacher_axis, data_axis, optimizer_axis)
    n_teacher, n_data, n_optimizer = ordered.shape[:3]
    reducer = statistic or (lambda sample: sample.mean(axis=(0, 1, 2)))
    estimate = as_float_array(reducer(ordered), name="statistic result")
    distribution = np.empty((n_bootstrap,) + estimate.shape, dtype=float)
    rng = np.random.default_rng(seed)
    for bootstrap_index in range(n_bootstrap):
        resampled = np.empty_like(ordered)
        teacher_indices = rng.integers(0, n_teacher, size=n_teacher)
        for new_teacher, source_teacher in enumerate(teacher_indices):
            data_indices = rng.integers(0, n_data, size=n_data)
            for new_data, source_data in enumerate(data_indices):
                optimizer_indices = rng.integers(0, n_optimizer, size=n_optimizer)
                resampled[new_teacher, new_data] = ordered[
                    source_teacher, source_data, optimizer_indices
                ]
        result = as_float_array(reducer(resampled), name="statistic result")
        if result.shape != estimate.shape:
            raise ValueError("statistic returned inconsistent shapes across resamples")
        distribution[bootstrap_index] = result
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha], axis=0)
    return {
        "estimate": scalar_or_array(estimate),
        "distribution": distribution,
        "confidence_interval": (scalar_or_array(lower), scalar_or_array(upper)),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }


def susceptibility(
    samples: ArrayLike,
    *,
    n_eff: float,
    sample_axis: int = 0,
    ddof: int = 0,
) -> Any:
    """Return ``N_eff * Var(order parameter)`` with explicit system size.

    Requiring ``n_eff`` prevents silently substituting parameter count, token
    count, or dataset size when those notions differ between experiments.
    """

    if not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError("n_eff must be finite and strictly positive")
    array = as_float_array(samples, name="samples")
    axis = sample_axis % array.ndim
    if array.shape[axis] <= ddof:
        raise ValueError("ddof must be smaller than the number of samples")
    return scalar_or_array(n_eff * np.var(array, axis=axis, ddof=ddof))


def binder_cumulant(
    samples: ArrayLike,
    *,
    sample_axis: int = 0,
    centered: bool = False,
    vector_axis: int | None = None,
    eps: float = 1e-12,
) -> Any:
    """Compute raw or centered scalar/O(n) Binder cumulants.

    For a scalar, ``U = 1 - <m^4> / (3 <m^2>^2)``. For an ``n``-component
    vector, ``U = 1 - n/(n+2) * <|m|^4>/<|m|^2>^2``. This normalization gives
    zero for an isotropic Gaussian and ``2/(n+2)`` for fixed vector magnitude.
    """

    array = as_float_array(samples, name="samples")
    sample = sample_axis % array.ndim
    if vector_axis is None:
        if centered:
            array = array - array.mean(axis=sample, keepdims=True)
        second = np.mean(np.square(array), axis=sample)
        fourth = np.mean(np.power(array, 4), axis=sample)
        cumulant = np.divide(
            fourth,
            3.0 * np.square(second),
            out=np.full_like(np.asarray(second), np.nan, dtype=float),
            where=np.abs(second) > eps,
        )
        return scalar_or_array(1.0 - cumulant)

    vector = vector_axis % array.ndim
    if vector == sample:
        raise ValueError("sample_axis and vector_axis must be distinct")
    if centered:
        array = array - array.mean(axis=sample, keepdims=True)
    n_components = array.shape[vector]
    if n_components == 0:
        raise ValueError("vector axis must be non-empty")
    squared_radius = np.sum(np.square(array), axis=vector)
    adjusted_sample_axis = sample - 1 if vector < sample else sample
    second = np.mean(squared_radius, axis=adjusted_sample_axis)
    fourth = np.mean(np.square(squared_radius), axis=adjusted_sample_axis)
    ratio = np.divide(
        fourth,
        np.square(second),
        out=np.full_like(np.asarray(second), np.nan, dtype=float),
        where=np.abs(second) > eps,
    )
    return scalar_or_array(1.0 - (n_components / (n_components + 2.0)) * ratio)


def binder_summary(
    samples: ArrayLike,
    *,
    sample_axis: int = 0,
    vector_axis: int | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Return both raw and centered Binder conventions side by side."""

    return {
        "raw": binder_cumulant(
            samples, sample_axis=sample_axis, centered=False, vector_axis=vector_axis, eps=eps
        ),
        "centered": binder_cumulant(
            samples, sample_axis=sample_axis, centered=True, vector_axis=vector_axis, eps=eps
        ),
    }


def order_parameter_histogram(
    samples: ArrayLike,
    *,
    bins: int | ArrayLike = 51,
    value_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Create shared plotting primitives for an empirical order distribution."""

    values = as_float_array(samples, name="samples").ravel()
    if values.size == 0:
        raise ValueError("samples must be non-empty")
    bin_spec = bins if isinstance(bins, int) else as_float_array(bins, name="bins", ndim=1)
    counts, edges = np.histogram(values, bins=bin_spec, range=value_range)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)
    return {
        "counts": counts,
        "density": density,
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
    }


def rate_function(
    probabilities: ArrayLike,
    *,
    n_eff: float = 1.0,
    eps: float = 1e-300,
    shift_minimum: bool = True,
) -> np.ndarray:
    """Convert histogram probabilities/densities to ``-log(P)/N_eff``."""

    if not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError("n_eff must be finite and strictly positive")
    probability = as_float_array(probabilities, name="probabilities")
    if np.any(probability < 0):
        raise ValueError("probabilities must be non-negative")
    result = -np.log(np.maximum(probability, eps)) / n_eff
    if shift_minimum and result.size:
        result = result - result.min()
    return result

