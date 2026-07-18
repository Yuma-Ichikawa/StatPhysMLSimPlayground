"""Finite-size scaling and reproducible data-collapse searches."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array


def data_collapse_score(
    control: ArrayLike,
    sizes: ArrayLike,
    observables: ArrayLike,
    *,
    critical_control: float,
    observable_exponent: float,
    inverse_nu: float,
    n_grid: int = 128,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Score ``O(N,g) N^y = F((g-g_c) N^(1/nu))`` collapse.

    Curves are interpolated only on the scaled-coordinate interval shared by
    every size, preventing extrapolation from artificially improving a fit. The
    score is mean pointwise across-size variance divided by the mean squared
    collapsed signal; zero denotes exact collapse.
    """

    x = as_float_array(control, name="control", ndim=1)
    system_sizes = as_float_array(sizes, name="sizes", ndim=1)
    curves = as_float_array(observables, name="observables", ndim=2)
    if curves.shape != (system_sizes.size, x.size):
        raise ValueError("observables must have shape (n_sizes, n_control)")
    if x.size < 2 or system_sizes.size < 2:
        raise ValueError("at least two controls and two sizes are required")
    if np.any(system_sizes <= 0):
        raise ValueError("sizes must be strictly positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two")
    order = np.argsort(x)
    x, curves = x[order], curves[:, order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("control values must be unique")
    scaled_x = (x[None, :] - critical_control) * np.power(system_sizes[:, None], inverse_nu)
    scaled_y = curves * np.power(system_sizes[:, None], observable_exponent)
    common_left = float(np.max(scaled_x[:, 0]))
    common_right = float(np.min(scaled_x[:, -1]))
    if not common_left < common_right:
        return {
            "score": float("inf"),
            "scaled_grid": np.empty(0),
            "collapsed_curves": np.empty((system_sizes.size, 0)),
            "common_interval": (common_left, common_right),
        }
    grid = np.linspace(common_left, common_right, n_grid)
    collapsed = np.vstack(
        [np.interp(grid, scaled_x[index], scaled_y[index]) for index in range(system_sizes.size)]
    )
    pointwise_variance = np.var(collapsed, axis=0)
    signal_scale = float(np.mean(np.square(collapsed)))
    score = float(pointwise_variance.mean() / max(signal_scale, eps))
    return {
        "score": score,
        "scaled_grid": grid,
        "collapsed_curves": collapsed,
        "collapsed_mean": collapsed.mean(axis=0),
        "pointwise_variance": pointwise_variance,
        "common_interval": (common_left, common_right),
    }


def fit_finite_size_scaling_grid(
    control: ArrayLike,
    sizes: ArrayLike,
    observables: ArrayLike,
    *,
    critical_control_grid: ArrayLike,
    observable_exponent_grid: ArrayLike,
    inverse_nu_grid: ArrayLike,
    n_grid: int = 128,
) -> dict[str, Any]:
    """Fit finite-size scaling parameters by an explicit deterministic grid."""

    critical_values = as_float_array(
        critical_control_grid, name="critical_control_grid", ndim=1
    )
    exponent_values = as_float_array(
        observable_exponent_grid, name="observable_exponent_grid", ndim=1
    )
    inverse_nu_values = as_float_array(inverse_nu_grid, name="inverse_nu_grid", ndim=1)
    if min(critical_values.size, exponent_values.size, inverse_nu_values.size) == 0:
        raise ValueError("all parameter grids must be non-empty")
    scores = np.empty(
        (critical_values.size, exponent_values.size, inverse_nu_values.size), dtype=float
    )
    best_score = float("inf")
    best_indices = (0, 0, 0)
    for critical_index, critical in enumerate(critical_values):
        for exponent_index, exponent in enumerate(exponent_values):
            for nu_index, inverse_nu in enumerate(inverse_nu_values):
                result = data_collapse_score(
                    control,
                    sizes,
                    observables,
                    critical_control=float(critical),
                    observable_exponent=float(exponent),
                    inverse_nu=float(inverse_nu),
                    n_grid=n_grid,
                )
                score = float(result["score"])
                scores[critical_index, exponent_index, nu_index] = score
                if score < best_score:
                    best_score = score
                    best_indices = (critical_index, exponent_index, nu_index)
    critical_index, exponent_index, nu_index = best_indices
    best_critical = float(critical_values[critical_index])
    best_exponent = float(exponent_values[exponent_index])
    best_inverse_nu = float(inverse_nu_values[nu_index])
    collapse = data_collapse_score(
        control,
        sizes,
        observables,
        critical_control=best_critical,
        observable_exponent=best_exponent,
        inverse_nu=best_inverse_nu,
        n_grid=n_grid,
    )
    return {
        "critical_control": best_critical,
        "observable_exponent": best_exponent,
        "inverse_nu": best_inverse_nu,
        "nu": float(1.0 / best_inverse_nu) if best_inverse_nu != 0 else float("inf"),
        "score": best_score,
        "score_grid": scores,
        "best_indices": best_indices,
        "parameter_grids": {
            "critical_control": critical_values,
            "observable_exponent": exponent_values,
            "inverse_nu": inverse_nu_values,
        },
        "collapse": collapse,
    }


def bootstrap_finite_size_scaling(
    control: ArrayLike,
    sizes: ArrayLike,
    observations: ArrayLike,
    *,
    critical_control_grid: ArrayLike,
    observable_exponent_grid: ArrayLike,
    inverse_nu_grid: ArrayLike,
    replica_axis: int = -1,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int | None = None,
    n_grid: int = 128,
) -> dict[str, Any]:
    """Bootstrap replica-level curves and refit the complete collapse grid."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    raw = as_float_array(observations, name="observations")
    if raw.ndim != 3:
        raise ValueError("observations must have size, control, and replica axes")
    replica = replica_axis % raw.ndim
    raw = np.moveaxis(raw, replica, -1)
    x = as_float_array(control, name="control", ndim=1)
    system_sizes = as_float_array(sizes, name="sizes", ndim=1)
    if raw.shape[:2] != (system_sizes.size, x.size):
        raise ValueError("non-replica axes must be ordered as (size, control)")
    n_replica = raw.shape[-1]
    if n_replica < 1:
        raise ValueError("replica axis must be non-empty")
    common_arguments = {
        "critical_control_grid": critical_control_grid,
        "observable_exponent_grid": observable_exponent_grid,
        "inverse_nu_grid": inverse_nu_grid,
        "n_grid": n_grid,
    }
    point = fit_finite_size_scaling_grid(x, system_sizes, raw.mean(axis=-1), **common_arguments)
    distribution = np.empty((n_bootstrap, 4), dtype=float)
    rng = np.random.default_rng(seed)
    for index in range(n_bootstrap):
        replica_indices = rng.integers(0, n_replica, size=raw.shape)
        sampled = np.take_along_axis(raw, replica_indices, axis=-1).mean(axis=-1)
        fit = fit_finite_size_scaling_grid(x, system_sizes, sampled, **common_arguments)
        distribution[index] = (
            fit["critical_control"],
            fit["observable_exponent"],
            fit["inverse_nu"],
            fit["score"],
        )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha], axis=0)
    names = ("critical_control", "observable_exponent", "inverse_nu", "score")
    intervals = {
        name: (float(lower[position]), float(upper[position]))
        for position, name in enumerate(names)
    }
    return {
        "fit": point,
        "parameter_names": names,
        "distribution": distribution,
        "confidence_intervals": intervals,
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }

