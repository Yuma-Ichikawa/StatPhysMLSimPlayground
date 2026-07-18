"""Permutation-free functional replica-overlap distributions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array


def functional_replica_overlaps(
    outputs: ArrayLike,
    *,
    center: bool = True,
    normalize: bool = True,
    bins: int | ArrayLike | None = 41,
    value_range: tuple[float, float] | None = (-1.0, 1.0),
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Estimate the functional replica overlap matrix and empirical ``P(q)``.

    The first axis indexes independently trained replicas. All remaining axes
    are flattened evaluation outputs on a *shared* probe set. Centering each
    replica makes the normalized overlap insensitive to additive output bias.
    Undefined overlaps involving a constant replica are represented by NaN and
    omitted from ``P(q)``.
    """

    array = as_float_array(outputs, name="outputs")
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("outputs must contain at least two replicas")
    replicas = array.reshape(array.shape[0], -1)
    if replicas.shape[1] == 0:
        raise ValueError("each replica must contain at least one output")
    if center:
        replicas = replicas - replicas.mean(axis=1, keepdims=True)
    if normalize:
        norms = np.linalg.norm(replicas, axis=1)
        valid = norms > eps
        normalized = np.divide(
            replicas,
            norms[:, None],
            out=np.zeros_like(replicas),
            where=valid[:, None],
        )
        overlap = normalized @ normalized.T
        overlap[~(valid[:, None] & valid[None, :])] = np.nan
        overlap = np.clip(overlap, -1.0, 1.0)
    else:
        valid = np.ones(replicas.shape[0], dtype=bool)
        overlap = replicas @ replicas.T / replicas.shape[1]
    row, column = np.triu_indices(replicas.shape[0], k=1)
    pair_overlaps = overlap[row, column]
    finite_pairs = pair_overlaps[np.isfinite(pair_overlaps)]
    result: dict[str, Any] = {
        "overlap_matrix": overlap,
        "pair_overlaps": finite_pairs,
        "pair_indices": np.column_stack((row, column))[np.isfinite(pair_overlaps)],
        "mean_overlap": float(finite_pairs.mean()) if finite_pairs.size else float("nan"),
        "std_overlap": float(finite_pairs.std()) if finite_pairs.size else float("nan"),
        "n_pairs": int(finite_pairs.size),
        "n_undefined_pairs": int(pair_overlaps.size - finite_pairs.size),
        "valid_replicas": valid,
    }
    if bins is not None:
        bin_spec = bins if isinstance(bins, int) else as_float_array(bins, name="bins", ndim=1)
        counts, edges = np.histogram(finite_pairs, bins=bin_spec, range=value_range)
        widths = np.diff(edges)
        total = int(counts.sum())
        density = counts / (total * widths) if total else np.zeros_like(widths)
        result["histogram"] = {
            "counts": counts,
            "density": density,
            "edges": edges,
            "centers": 0.5 * (edges[:-1] + edges[1:]),
        }
    return result


functional_replica_distribution = functional_replica_overlaps

