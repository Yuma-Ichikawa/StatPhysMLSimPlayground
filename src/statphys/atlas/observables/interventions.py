"""Causal observables from ablation and intervention experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._array import ArrayLike, as_float_array


def intervention_loss_deltas(
    baseline_loss: ArrayLike,
    intervention_losses: Mapping[str, ArrayLike],
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute absolute and relative loss increases for named interventions.

    Scalar losses and identically-shaped arrays are supported. A positive delta
    means that the intervention harmed performance. Relative deltas are NaN
    wherever the baseline magnitude is at most ``eps``.
    """

    baseline = as_float_array(baseline_loss, name="baseline_loss")
    results: dict[str, Any] = {"baseline_loss": float(baseline) if baseline.ndim == 0 else baseline.copy()}
    deltas: dict[str, dict[str, Any]] = {}
    for name, value in intervention_losses.items():
        intervened = as_float_array(value, name=f"intervention_losses[{name!r}]")
        try:
            delta = intervened - baseline
        except ValueError as exc:
            raise ValueError(f"intervention {name!r} is not broadcast-compatible with baseline") from exc
        relative = np.divide(
            delta,
            np.abs(baseline),
            out=np.full(np.broadcast_shapes(delta.shape, baseline.shape), np.nan, dtype=float),
            where=np.abs(baseline) > eps,
        )

        def scalar_or_array(array: np.ndarray) -> Any:
            return float(array) if array.ndim == 0 else array

        deltas[str(name)] = {
            "loss": scalar_or_array(intervened),
            "delta": scalar_or_array(np.asarray(delta)),
            "relative_delta": scalar_or_array(np.asarray(relative)),
        }
    results["interventions"] = deltas
    return results

