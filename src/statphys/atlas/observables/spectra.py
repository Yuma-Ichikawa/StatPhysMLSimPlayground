"""Spectral observables for attention operators and learned subspaces."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._array import ArrayLike, as_float_array


def matrix_spectrum(
    matrix: ArrayLike,
    *,
    top_k: int | None = None,
    explained_rank: int = 1,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Return singular-value outliers and scale-invariant rank statistics.

    ``stable_rank`` is :math:`||M||_F^2/||M||_2^2`, while explained fraction
    uses squared singular values. Batched matrices are deliberately rejected:
    callers should retain the semantic identity (layer/head) while looping.
    """

    array = as_float_array(matrix, name="matrix", ndim=2)
    if min(array.shape) == 0:
        raise ValueError("matrix axes must be non-empty")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if explained_rank < 1:
        raise ValueError("explained_rank must be positive")

    singular_values = np.linalg.svd(array, compute_uv=False)
    energy = singular_values**2
    total_energy = float(energy.sum())
    leading = float(singular_values[0])
    second = float(singular_values[1]) if singular_values.size > 1 else 0.0
    stable_rank = total_energy / (leading**2) if leading > eps else 0.0
    explained = (
        float(energy[: min(explained_rank, energy.size)].sum() / total_energy)
        if total_energy > eps
        else 0.0
    )
    if second > eps:
        outlier_ratio = leading / second
    elif leading > eps:
        outlier_ratio = float("inf")
    else:
        outlier_ratio = 1.0
    retained = singular_values if top_k is None else singular_values[:top_k]
    return {
        "singular_values": singular_values,
        "top_singular_values": retained,
        "spectral_norm": leading,
        "frobenius_norm": float(np.sqrt(total_energy)),
        "stable_rank": float(stable_rank),
        "outlier_gap": float(leading - second),
        "outlier_ratio": float(outlier_ratio),
        "explained_fraction": explained,
        "explained_rank": int(min(explained_rank, energy.size)),
    }


def effective_qk_matrix(
    query_weight: ArrayLike,
    key_weight: ArrayLike,
    *,
    convention: Literal["pytorch", "columns"] = "pytorch",
) -> np.ndarray:
    """Construct the input-space bilinear QK operator.

    Under the default PyTorch ``Linear`` convention, weights have shape
    ``(d_head, d_model)`` and the returned operator is ``W_Q.T @ W_K``.
    With ``convention='columns'``, weights have shape ``(d_model, d_head)``.
    """

    query = as_float_array(query_weight, name="query_weight", ndim=2)
    key = as_float_array(key_weight, name="key_weight", ndim=2)
    if convention == "pytorch":
        if query.shape[0] != key.shape[0]:
            raise ValueError("Q and K output dimensions must match")
        return query.T @ key
    if convention == "columns":
        if query.shape[1] != key.shape[1]:
            raise ValueError("Q and K head dimensions must match")
        return query @ key.T
    raise ValueError("convention must be 'pytorch' or 'columns'")


def effective_ov_matrix(
    value_weight: ArrayLike,
    output_weight: ArrayLike,
    *,
    convention: Literal["pytorch", "columns"] = "pytorch",
) -> np.ndarray:
    """Construct the input-to-output OV operator for one attention head.

    For PyTorch weights, ``W_V`` has shape ``(d_head, d_model)`` and the
    head slice of ``W_O`` has shape ``(d_model, d_head)``, hence ``W_O @ W_V``.
    In the column convention the corresponding product is ``W_V @ W_O``.
    """

    value = as_float_array(value_weight, name="value_weight", ndim=2)
    output = as_float_array(output_weight, name="output_weight", ndim=2)
    if convention == "pytorch":
        if output.shape[1] != value.shape[0]:
            raise ValueError("OV head dimensions must match")
        return output @ value
    if convention == "columns":
        if value.shape[1] != output.shape[0]:
            raise ValueError("OV head dimensions must match")
        return value @ output
    raise ValueError("convention must be 'pytorch' or 'columns'")


def qk_ov_spectra(
    query_weight: ArrayLike,
    key_weight: ArrayLike,
    value_weight: ArrayLike,
    output_weight: ArrayLike,
    *,
    convention: Literal["pytorch", "columns"] = "pytorch",
    top_k: int | None = None,
    explained_rank: int = 1,
) -> dict[str, Any]:
    """Construct effective QK/OV matrices and summarize their spectra."""

    qk = effective_qk_matrix(query_weight, key_weight, convention=convention)
    ov = effective_ov_matrix(value_weight, output_weight, convention=convention)
    return {
        "qk_matrix": qk,
        "ov_matrix": ov,
        "qk": matrix_spectrum(qk, top_k=top_k, explained_rank=explained_rank),
        "ov": matrix_spectrum(ov, top_k=top_k, explained_rank=explained_rank),
    }


def subspace_principal_angles(
    student_subspace: ArrayLike,
    teacher_subspace: ArrayLike,
    *,
    vectors_as_rows: bool = True,
    rank_tol: float | None = None,
) -> dict[str, Any]:
    """Measure alignment between student and teacher spans.

    The result is invariant to rotations and rescalings within either span.
    If the spans have unequal rank, principal angles cover the smaller rank and
    ``unmatched_dimensions`` records the difference.
    """

    student = as_float_array(student_subspace, name="student_subspace", ndim=2)
    teacher = as_float_array(teacher_subspace, name="teacher_subspace", ndim=2)
    if vectors_as_rows:
        student = student.T
        teacher = teacher.T
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("student and teacher vectors must share ambient dimension")

    def orthonormal_basis(vectors: np.ndarray) -> tuple[np.ndarray, int]:
        u, singular, _ = np.linalg.svd(vectors, full_matrices=False)
        if singular.size == 0:
            return u[:, :0], 0
        tolerance = (
            float(rank_tol)
            if rank_tol is not None
            else max(vectors.shape) * np.finfo(float).eps * float(singular[0])
        )
        rank = int(np.count_nonzero(singular > tolerance))
        return u[:, :rank], rank

    student_basis, student_rank = orthonormal_basis(student)
    teacher_basis, teacher_rank = orthonormal_basis(teacher)
    if student_rank == 0 or teacher_rank == 0:
        raise ValueError("both subspaces must have positive numerical rank")
    cosines = np.linalg.svd(student_basis.T @ teacher_basis, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    angles = np.arccos(cosines)
    squared_cosine = cosines**2
    return {
        "cosines": cosines,
        "angles_radians": angles,
        "angles_degrees": np.degrees(angles),
        "mean_cosine": float(cosines.mean()),
        "mean_squared_cosine": float(squared_cosine.mean()),
        "chordal_distance": float(np.sqrt(np.sum(1.0 - squared_cosine))),
        "student_rank": student_rank,
        "teacher_rank": teacher_rank,
        "unmatched_dimensions": abs(student_rank - teacher_rank),
    }

