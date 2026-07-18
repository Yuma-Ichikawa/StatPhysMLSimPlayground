"""Functional order parameters for positional--semantic decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array, safe_rms

__all__ = ["centered_functional_overlap", "two_template_decomposition"]


def _paired_flatten(left: ArrayLike, right: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = as_float_array(left, name="left").reshape(-1)
    b = as_float_array(right, name="right").reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"left and right must have the same number of values: {a.size} != {b.size}")
    if a.size < 2:
        raise ValueError("functional overlaps require at least two probe values")
    return a, b


def centered_functional_overlap(
    outputs: ArrayLike,
    template: ArrayLike,
    *,
    eps: float = 1e-12,
    zero_variance: str = "nan",
) -> float:
    """Return the Pearson/cosine overlap after centering both functions.

    The observable is invariant to an additive constant and to positive
    rescaling of either function.  A constant function has no centered
    direction; by default this returns ``nan``.  Set ``zero_variance`` to
    ``"zero"`` to obtain a conservative zero instead.
    """
    y, t = _paired_flatten(outputs, template)
    yc = y - y.mean()
    tc = t - t.mean()
    denominator = float(np.linalg.norm(yc) * np.linalg.norm(tc))
    if denominator <= eps:
        if zero_variance == "nan":
            return float("nan")
        if zero_variance == "zero":
            return 0.0
        raise ValueError("zero_variance must be 'nan' or 'zero'")
    return float(np.clip(np.dot(yc, tc) / denominator, -1.0, 1.0))


def two_template_decomposition(
    outputs: ArrayLike,
    positional_template: ArrayLike,
    semantic_template: ArrayLike,
    *,
    eps: float = 1e-12,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Fit centered outputs to centered positional and semantic templates.

    Each centered template and the centered output are normalized to unit
    L2 norm before least squares.  Consequently ``m_pos`` and ``m_sem`` are
    dimensionless *partial* template coefficients.  They equal ordinary
    centered overlaps when the two templates are orthogonal.  Correlated
    templates can yield coefficients outside ``[-1, 1]``; the returned
    marginal overlaps, matrix rank, and condition number make this
    identifiability issue explicit.

    ``coexistence_strength`` is ``min(abs(m_pos), abs(m_sem))`` and is large
    only when both partial components are present.  ``coexistence_balance``
    lies in ``[0, 1]`` and measures balance independent of amplitude.
    ``residual_fraction`` is ``SSE/SST = 1-R2`` up to numerical tolerance.
    Constant outputs have undefined R2 and return zero coefficients with
    ``degenerate_output=True``.
    """
    y = as_float_array(outputs, name="outputs").reshape(-1)
    p = as_float_array(positional_template, name="positional_template").reshape(-1)
    s = as_float_array(semantic_template, name="semantic_template").reshape(-1)
    if y.size < 3:
        raise ValueError("two-template decomposition requires at least three probe values")
    if p.shape != y.shape or s.shape != y.shape:
        raise ValueError("outputs and both templates must have identical shapes after flattening")

    yc, pc, sc = y - y.mean(), p - p.mean(), s - s.mean()
    y_norm = float(np.linalg.norm(yc))
    p_norm = float(np.linalg.norm(pc))
    s_norm = float(np.linalg.norm(sc))
    if p_norm <= eps or s_norm <= eps:
        which = "positional" if p_norm <= eps else "semantic"
        raise ValueError(f"{which} template is constant after centering")

    design = np.column_stack((pc / p_norm, sc / s_norm))
    singular_values = np.linalg.svd(design, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > eps
        else float("inf")
    )
    rank = int(np.linalg.matrix_rank(design, tol=eps))
    marginal_pos = centered_functional_overlap(y, p, eps=eps)
    marginal_sem = centered_functional_overlap(y, s, eps=eps)

    if y_norm <= eps:
        return {
            "m_pos": 0.0,
            "m_sem": 0.0,
            "positional_overlap": marginal_pos,
            "semantic_overlap": marginal_sem,
            "coexistence_strength": 0.0,
            "coexistence_balance": 0.0,
            "r2": float("nan"),
            "residual_fraction": float("nan"),
            "residual_rms": 0.0,
            "rank": rank,
            "condition_number": condition,
            "degenerate_output": True,
        }

    target = yc / y_norm
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=rcond)
    fitted = design @ coefficients
    residual = target - fitted
    residual_fraction = float(np.dot(residual, residual))
    # Projection cannot increase norm in exact arithmetic.  Clamp tiny drift.
    residual_fraction = float(np.clip(residual_fraction, 0.0, 1.0 + 1e-10))
    r2 = float(np.clip(1.0 - residual_fraction, -1e-10, 1.0))
    m_pos, m_sem = map(float, coefficients)
    magnitudes = np.abs(coefficients)
    magnitude_sum = float(magnitudes.sum())
    balance = 0.0 if magnitude_sum <= eps else float(2.0 * magnitudes.min() / magnitude_sum)

    return {
        "m_pos": m_pos,
        "m_sem": m_sem,
        "positional_overlap": marginal_pos,
        "semantic_overlap": marginal_sem,
        "coexistence_strength": float(magnitudes.min()),
        "coexistence_balance": balance,
        "r2": r2,
        "residual_fraction": residual_fraction,
        "residual_rms": safe_rms(yc - (fitted * y_norm)),
        "rank": rank,
        "condition_number": condition,
        "degenerate_output": False,
    }
