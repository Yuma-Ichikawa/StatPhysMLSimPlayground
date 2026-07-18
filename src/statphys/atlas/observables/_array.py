"""Array conversion and validation helpers for atlas observables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


ArrayLike = Any


def as_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | Sequence[int] | None = None,
    finite: bool = True,
    copy: bool = False,
) -> np.ndarray:
    """Convert NumPy/Torch-like input to a validated float64 array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    # NumPy >=2 rejects ``np.array(scalar, copy=False)`` because a scalar must
    # be materialized. ``asarray`` preserves the intended no-unnecessary-copy
    # behavior and an explicit copy remains available to callers.
    array = np.asarray(value, dtype=np.float64)
    if copy:
        array = array.copy()
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if array.ndim not in allowed:
            expected = " or ".join(str(v) for v in allowed)
            raise ValueError(f"{name} must have {expected} dimensions, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def safe_rms(array: np.ndarray) -> float:
    """Return root mean square, including a stable zero result."""
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def normalized_rows(array: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize matrix rows and return normalized rows and original norms."""
    norms = np.linalg.norm(array, axis=1)
    normalized = np.zeros_like(array, dtype=np.float64)
    valid = norms > eps
    normalized[valid] = array[valid] / norms[valid, None]
    return normalized, norms
