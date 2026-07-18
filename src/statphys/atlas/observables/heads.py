"""Permutation-aware observables for multi-head specialization."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array, normalized_rows

__all__ = [
    "head_specialization_metrics",
    "latent_overlap_matrix",
    "match_heads",
    "permutation_invariant_head_spectrum",
]


def latent_overlap_matrix(
    heads: ArrayLike,
    latent_components: ArrayLike,
    *,
    absolute: bool = False,
    center: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return cosine overlaps between ``H`` heads and ``R`` latent signals.

    Inputs have shapes ``(H, ...)`` and ``(R, ...)``; all non-leading
    dimensions are flattened and must contain the same number of values.
    ``center=True`` removes each row's mean before normalization, which is
    useful for functional profiles but is usually inappropriate for weights.
    """
    head_array = as_float_array(heads, name="heads")
    latent_array = as_float_array(latent_components, name="latent_components")
    if head_array.ndim < 2 or latent_array.ndim < 2:
        raise ValueError("heads and latent_components must each have a leading component axis")
    head_flat = head_array.reshape(head_array.shape[0], -1)
    latent_flat = latent_array.reshape(latent_array.shape[0], -1)
    if head_flat.shape[1] != latent_flat.shape[1]:
        raise ValueError(
            "heads and latent components must have matching flattened dimensions: "
            f"{head_flat.shape[1]} != {latent_flat.shape[1]}"
        )
    if center:
        head_flat = head_flat - head_flat.mean(axis=1, keepdims=True)
        latent_flat = latent_flat - latent_flat.mean(axis=1, keepdims=True)
    head_unit, _ = normalized_rows(head_flat, eps=eps)
    latent_unit, _ = normalized_rows(latent_flat, eps=eps)
    overlap = head_unit @ latent_unit.T
    if absolute:
        overlap = np.abs(overlap)
    return np.clip(overlap, 0.0 if absolute else -1.0, 1.0)


def _exact_assignment(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = scores.shape
    best_score = -float("inf")
    best_rows: tuple[int, ...] = ()
    best_columns: tuple[int, ...] = ()
    if rows <= columns:
        fixed_rows = tuple(range(rows))
        for candidate_columns in itertools.permutations(range(columns), rows):
            score = float(scores[fixed_rows, candidate_columns].sum())
            if score > best_score:
                best_score = score
                best_rows, best_columns = fixed_rows, candidate_columns
    else:
        fixed_columns = tuple(range(columns))
        for candidate_rows in itertools.permutations(range(rows), columns):
            score = float(scores[candidate_rows, fixed_columns].sum())
            if score > best_score:
                best_score = score
                best_rows, best_columns = candidate_rows, fixed_columns
    return np.asarray(best_rows, dtype=int), np.asarray(best_columns, dtype=int)


def _greedy_assignment(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available_rows = set(range(scores.shape[0]))
    available_columns = set(range(scores.shape[1]))
    matched_rows: list[int] = []
    matched_columns: list[int] = []
    while available_rows and available_columns:
        best = max(
            ((scores[row, column], row, column) for row in available_rows for column in available_columns),
            key=lambda item: item[0],
        )
        _, row, column = best
        matched_rows.append(row)
        matched_columns.append(column)
        available_rows.remove(row)
        available_columns.remove(column)
    return np.asarray(matched_rows, dtype=int), np.asarray(matched_columns, dtype=int)


def match_heads(
    overlap_matrix: ArrayLike,
    *,
    maximize_absolute: bool = True,
    method: str = "auto",
    exact_max_dim: int = 8,
) -> dict[str, Any]:
    """Match heads to latent components using a maximum-weight assignment.

    ``method="auto"`` uses SciPy's Hungarian algorithm.  If SciPy is not
    importable, it performs an exact permutation search when both matrix
    dimensions are at most ``exact_max_dim`` and otherwise uses a documented
    greedy fallback.  The returned ``method`` field is always one of
    ``hungarian_scipy``, ``exact_permutation``, or ``greedy_fallback`` so an
    approximate assignment can never be mistaken for an exact one.
    """
    overlap = as_float_array(overlap_matrix, name="overlap_matrix", ndim=2)
    scores = np.abs(overlap) if maximize_absolute else overlap
    allowed = {"auto", "scipy", "exact", "greedy"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}, got {method!r}")

    rows: np.ndarray
    columns: np.ndarray
    used_method: str
    if method in {"auto", "scipy"}:
        try:
            from scipy.optimize import linear_sum_assignment

            rows, columns = linear_sum_assignment(scores, maximize=True)
            used_method = "hungarian_scipy"
        except (ImportError, ModuleNotFoundError):
            if method == "scipy":
                raise
            if max(scores.shape) <= exact_max_dim:
                rows, columns = _exact_assignment(scores)
                used_method = "exact_permutation"
            else:
                rows, columns = _greedy_assignment(scores)
                used_method = "greedy_fallback"
    elif method == "exact":
        if max(scores.shape) > exact_max_dim:
            raise ValueError(
                f"exact assignment is limited to max dimension {exact_max_dim}; got {scores.shape}"
            )
        rows, columns = _exact_assignment(scores)
        used_method = "exact_permutation"
    else:
        rows, columns = _greedy_assignment(scores)
        used_method = "greedy_fallback"

    matched = overlap[rows, columns]
    return {
        "row_indices": rows,
        "column_indices": columns,
        "matched_overlaps": matched,
        "matched_scores": scores[rows, columns],
        "mean_matched_score": float(scores[rows, columns].mean()) if rows.size else float("nan"),
        "total_matched_score": float(scores[rows, columns].sum()),
        "method": used_method,
        "is_exact": used_method != "greedy_fallback",
    }


def permutation_invariant_head_spectrum(overlap_matrix: ArrayLike) -> dict[str, Any]:
    """Return singular/eigen spectra invariant to head and latent permutations."""
    overlap = as_float_array(overlap_matrix, name="overlap_matrix", ndim=2)
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    gram_eigenvalues = np.square(singular_values)
    energy = float(gram_eigenvalues.sum())
    normalized = (
        gram_eigenvalues / energy if energy > 0 else np.zeros_like(gram_eigenvalues)
    )
    effective_rank = (
        float(1.0 / np.square(normalized).sum()) if np.square(normalized).sum() > 0 else 0.0
    )
    return {
        "singular_values": singular_values,
        "gram_eigenvalues": gram_eigenvalues,
        "normalized_energy": normalized,
        "effective_rank": effective_rank,
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
        "frobenius_norm": float(np.sqrt(energy)),
    }


def head_specialization_metrics(
    overlap_matrix: ArrayLike,
    *,
    dead_relative_threshold: float = 0.05,
    redundancy_cosine_threshold: float = 0.95,
    eps: float = 1e-12,
    assignment_method: str = "auto",
) -> dict[str, Any]:
    """Summarize specialization, redundancy, and inactivity of attention heads.

    Specialization strength is the mean squared distance of each head's
    latent-overlap vector from the mean head.  Per-head specialization
    entropy uses squared overlaps as assignment probabilities.  Effective
    head count is the participation ratio of head activities.  A head is
    dead when its squared-overlap activity is below a user-visible fraction
    of the most active head; a head is redundant when its absolute overlap
    profile has cosine similarity above the requested threshold with any
    other non-dead head.
    """
    overlap = as_float_array(overlap_matrix, name="overlap_matrix", ndim=2)
    if not 0.0 <= dead_relative_threshold <= 1.0:
        raise ValueError("dead_relative_threshold must lie in [0, 1]")
    if not 0.0 <= redundancy_cosine_threshold <= 1.0:
        raise ValueError("redundancy_cosine_threshold must lie in [0, 1]")

    n_heads, n_latents = overlap.shape
    row_mean = overlap.mean(axis=0, keepdims=True)
    strength = float(np.mean(np.sum(np.square(overlap - row_mean), axis=1)))
    squared = np.square(overlap)
    activities = squared.sum(axis=1)
    max_activity = float(activities.max(initial=0.0))
    if max_activity <= eps:
        dead = np.ones(n_heads, dtype=bool)
    else:
        dead = activities <= dead_relative_threshold * max_activity

    probabilities = np.zeros_like(squared)
    active_rows = activities > eps
    probabilities[active_rows] = squared[active_rows] / activities[active_rows, None]
    entropies = np.zeros(n_heads, dtype=np.float64)
    positive = probabilities > 0
    entropies = -np.sum(np.where(positive, probabilities * np.log(probabilities + eps), 0.0), axis=1)
    entropy_normalizer = np.log(n_latents) if n_latents > 1 else 1.0
    normalized_entropies = entropies / entropy_normalizer

    activity_sum = float(activities.sum())
    effective_heads = (
        float(activity_sum**2 / np.square(activities).sum())
        if np.square(activities).sum() > eps
        else 0.0
    )

    profiles, profile_norms = normalized_rows(np.abs(overlap), eps=eps)
    similarity = profiles @ profiles.T
    redundant = np.zeros(n_heads, dtype=bool)
    for head in range(n_heads):
        if dead[head] or profile_norms[head] <= eps:
            continue
        peers = np.ones(n_heads, dtype=bool)
        peers[head] = False
        peers &= ~dead
        redundant[head] = bool(np.any(similarity[head, peers] >= redundancy_cosine_threshold))

    assignment = match_heads(overlap, method=assignment_method)
    spectrum = permutation_invariant_head_spectrum(overlap)
    return {
        "specialization_strength": strength,
        "specialization_entropy": float(entropies[active_rows].mean()) if active_rows.any() else 0.0,
        "normalized_specialization_entropy": (
            float(normalized_entropies[active_rows].mean()) if active_rows.any() else 0.0
        ),
        "per_head_entropy": entropies,
        "per_head_normalized_entropy": normalized_entropies,
        "head_activity": activities,
        "effective_heads": effective_heads,
        "dead_head_fraction": float(dead.mean()),
        "dead_heads": dead,
        "redundant_head_fraction": float(redundant.mean()),
        "redundant_heads": redundant,
        "head_profile_similarity": similarity,
        "assignment": assignment,
        "permutation_invariant_spectrum": spectrum,
    }
