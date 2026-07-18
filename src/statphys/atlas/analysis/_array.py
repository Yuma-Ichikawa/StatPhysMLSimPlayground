"""Small NumPy conversion helpers local to atlas analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

ArrayLike = Any


def as_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    """Convert NumPy/Torch-like input to a validated floating array."""

    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    array = np.asarray(candidate, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def scalar_or_array(value: np.ndarray | float) -> Any:
    """Convert a zero-dimensional array to float and preserve other arrays."""

    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array

