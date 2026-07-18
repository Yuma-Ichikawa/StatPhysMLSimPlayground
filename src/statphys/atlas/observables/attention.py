"""Geometry-based observables for causal and non-causal attention maps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array


def _masked_mean(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    """Return a mean over valid entries without emitting all-NaN warnings."""

    numerator = np.where(valid, values, 0.0).sum(axis=axis)
    denominator = valid.sum(axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def attention_geometry(
    attention: ArrayLike,
    *,
    query_positions: ArrayLike | None = None,
    key_positions: ArrayLike | None = None,
    sink_indices: Sequence[int] = (0,),
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Measure entropy, support, span, and local motifs of attention maps.

    Parameters
    ----------
    attention:
        Array whose final two axes are ``(query, key)``. Any preceding axes are
        retained as independent maps (for example layer, head, and batch).
        Rows need not be normalized, but weights must be non-negative.
    query_positions, key_positions:
        Optional physical/token positions. Defaults to integer indices. They
        determine span, diagonal mass, and previous-token mass.
    sink_indices:
        Key *indices* whose total mass is reported as sink mass.
    eps:
        A row with sum at most ``eps`` is treated as undefined and excluded
        from averages.

    Returns
    -------
    dict
        Scalar aggregate observables, per-map observables, and the row-level
        entropy/support/span arrays. Entropy uses natural logarithms.
    """

    weights = as_float_array(attention, name="attention")
    if weights.ndim < 2:
        raise ValueError("attention must have at least query and key axes")
    if np.any(weights < -eps):
        raise ValueError("attention weights must be non-negative")
    weights = np.maximum(weights, 0.0)
    n_query, n_key = weights.shape[-2:]
    if n_query == 0 or n_key == 0:
        raise ValueError("query and key axes must be non-empty")

    q_pos = (
        np.arange(n_query, dtype=float)
        if query_positions is None
        else as_float_array(query_positions, name="query_positions", ndim=1)
    )
    k_pos = (
        np.arange(n_key, dtype=float)
        if key_positions is None
        else as_float_array(key_positions, name="key_positions", ndim=1)
    )
    if q_pos.size != n_query or k_pos.size != n_key:
        raise ValueError("position arrays must match their attention axes")

    sinks = np.asarray(tuple(sink_indices), dtype=int)
    if sinks.ndim != 1:
        raise ValueError("sink_indices must be one-dimensional")
    if sinks.size and (np.any(sinks < 0) or np.any(sinks >= n_key)):
        raise ValueError("sink_indices contains an out-of-range key index")
    sinks = np.unique(sinks)

    row_sum = weights.sum(axis=-1, keepdims=True)
    valid = row_sum[..., 0] > eps
    probabilities = np.divide(
        weights,
        row_sum,
        out=np.zeros_like(weights, dtype=float),
        where=row_sum > eps,
    )

    log_p = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_p[positive] = np.log(probabilities[positive])
    row_entropy = -(probabilities * log_p).sum(axis=-1)
    row_entropy = np.where(valid, row_entropy, np.nan)
    row_support = np.where(valid, np.exp(row_entropy), np.nan)
    distance = np.abs(q_pos[:, None] - k_pos[None, :])
    row_span = np.where(valid, (probabilities * distance).sum(axis=-1), np.nan)

    sink_mass = probabilities[..., sinks].sum(axis=-1) if sinks.size else np.zeros_like(valid, dtype=float)
    diagonal_mask = q_pos[:, None] == k_pos[None, :]
    previous_mask = k_pos[None, :] == (q_pos[:, None] - 1.0)
    diagonal_mass = (probabilities * diagonal_mask).sum(axis=-1)
    previous_mass = (probabilities * previous_mask).sum(axis=-1)

    per_map_entropy = _masked_mean(row_entropy, valid, axis=-1)
    per_map_support = _masked_mean(row_support, valid, axis=-1)
    per_map_span = _masked_mean(row_span, valid, axis=-1)
    per_map_sink = _masked_mean(sink_mass, valid, axis=-1)
    per_map_diagonal = _masked_mean(diagonal_mass, valid, axis=-1)
    per_map_previous = _masked_mean(previous_mass, valid, axis=-1)

    valid_count = int(valid.sum())

    def aggregate(row_values: np.ndarray) -> float:
        if valid_count == 0:
            return float("nan")
        return float(np.where(valid, row_values, 0.0).sum() / valid_count)

    entropy = aggregate(row_entropy)
    entropy_norm = entropy / np.log(n_key) if n_key > 1 else 0.0
    return {
        "entropy": entropy,
        "normalized_entropy": float(entropy_norm),
        "effective_support": aggregate(row_support),
        "span": aggregate(row_span),
        "sink_mass": aggregate(sink_mass),
        "diagonal_mass": aggregate(diagonal_mass),
        "previous_token_mass": aggregate(previous_mass),
        "valid_query_fraction": float(valid.mean()),
        "per_map": {
            "entropy": per_map_entropy,
            "effective_support": per_map_support,
            "span": per_map_span,
            "sink_mass": per_map_sink,
            "diagonal_mass": per_map_diagonal,
            "previous_token_mass": per_map_previous,
        },
        "row_entropy": row_entropy,
        "row_effective_support": row_support,
        "row_span": row_span,
    }

