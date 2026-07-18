"""Order parameters for hidden representations and positional correlations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._array import ArrayLike, as_float_array


def representation_statistics(
    representations: ArrayLike,
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Summarize the centered representation covariance spectrum.

    All axes except the final feature axis are flattened into observations.
    Participation ratio is ``tr(C)^2 / tr(C^2)``; effective rank is the
    exponential spectral entropy; anisotropy is
    ``d * lambda_max / tr(C)`` (one for an exactly isotropic covariance).
    """

    array = as_float_array(representations, name="representations")
    if array.ndim < 2:
        raise ValueError("representations must have samples and features")
    n_features = array.shape[-1]
    samples = array.reshape(-1, n_features)
    if samples.shape[0] < 2 or n_features == 0:
        raise ValueError("at least two samples and one feature are required")
    centered = samples - samples.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2 / (samples.shape[0] - 1)
    # Include exact zero modes when feature dimension exceeds sample rank.
    if eigenvalues.size < n_features:
        eigenvalues = np.pad(eigenvalues, (0, n_features - eigenvalues.size))
    trace = float(eigenvalues.sum())
    squared_trace = float(np.square(eigenvalues).sum())
    if trace <= eps:
        participation_ratio = 0.0
        effective_rank = 0.0
        anisotropy = float("nan")
        top_fraction = 0.0
        probabilities = np.zeros_like(eigenvalues)
    else:
        participation_ratio = trace**2 / squared_trace
        probabilities = eigenvalues / trace
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        effective_rank = float(np.exp(entropy))
        anisotropy = float(n_features * eigenvalues[0] / trace)
        top_fraction = float(eigenvalues[0] / trace)
    return {
        "covariance_eigenvalues": eigenvalues,
        "spectral_probabilities": probabilities,
        "participation_ratio": float(participation_ratio),
        "effective_rank": float(effective_rank),
        "anisotropy": anisotropy,
        "top_explained_fraction": top_fraction,
        "covariance_trace": trace,
        "mean_squared_norm": float(np.mean(np.sum(np.square(samples), axis=-1))),
        "n_samples": int(samples.shape[0]),
        "n_features": int(n_features),
        "degenerate": bool(trace <= eps),
    }


def position_correlation(
    representations: ArrayLike,
    *,
    max_lag: int | None = None,
    connected: bool = True,
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate a translation-averaged correlation along the token axis.

    Input has shape ``(..., position, feature)``. Leading axes are treated as
    independent sequences. When ``connected`` is true, the global mean feature
    vector is removed before correlating positions.
    """

    array = as_float_array(representations, name="representations")
    if array.ndim < 2:
        raise ValueError("representations must have position and feature axes")
    n_position, n_features = array.shape[-2:]
    if n_position == 0 or n_features == 0:
        raise ValueError("position and feature axes must be non-empty")
    sequences = array.reshape(-1, n_position, n_features)
    if connected:
        sequences = sequences - sequences.mean(axis=(0, 1), keepdims=True)
    if max_lag is None:
        max_lag = n_position - 1
    if max_lag < 0 or max_lag >= n_position:
        raise ValueError("max_lag must lie between zero and n_position - 1")
    correlation = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        left = sequences[:, : n_position - lag]
        right = sequences[:, lag:]
        correlation[lag] = float(np.mean(np.sum(left * right, axis=-1)))
    if normalize:
        if abs(correlation[0]) <= eps:
            return np.full_like(correlation, np.nan)
        correlation = correlation / correlation[0]
    return correlation


def correlation_length(
    correlation: ArrayLike,
    *,
    method: Literal["integral", "exponential"] = "integral",
    eps: float = 1e-12,
) -> float:
    """Estimate a correlation length from a one-dimensional lag curve.

    ``integral`` sums the positive normalized lobe using a half-weight at lag
    zero. ``exponential`` fits ``log(C(r)/C(0)) = const - r/xi`` over the
    positive lobe. The function returns NaN when a length is not identifiable.
    """

    curve = as_float_array(correlation, name="correlation", ndim=1)
    if curve.size == 0 or abs(curve[0]) <= eps:
        return float("nan")
    normalized = curve / curve[0]
    finite_positive = np.isfinite(normalized) & (normalized > 0)
    end = 1
    while end < normalized.size and finite_positive[end]:
        end += 1
    if method == "integral":
        return float(0.5 + normalized[1:end].sum())
    if method == "exponential":
        lags = np.arange(end, dtype=float)
        if end < 2:
            return float("nan")
        slope = float(np.polyfit(lags, np.log(normalized[:end]), 1)[0])
        return float(-1.0 / slope) if slope < -eps else float("nan")
    raise ValueError("method must be 'integral' or 'exponential'")

