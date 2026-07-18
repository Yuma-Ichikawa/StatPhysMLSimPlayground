"""Conservative phase labels and transition-location estimators."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal

import numpy as np

from ._array import ArrayLike, as_float_array


def binder_crossing(
    control: ArrayLike,
    sizes: ArrayLike,
    binders: ArrayLike,
    *,
    adjacent_sizes: bool = True,
) -> dict[str, Any]:
    """Locate pairwise Binder-curve crossings by linear interpolation.

    Every sign change is retained rather than silently choosing one. The median
    crossing is a convenience summary; a broad or multi-modal set should be
    treated as weak evidence, as reflected by ``crossing_spread``.
    """

    x = as_float_array(control, name="control", ndim=1)
    system_sizes = as_float_array(sizes, name="sizes", ndim=1)
    curves = as_float_array(binders, name="binders", ndim=2)
    if curves.shape != (system_sizes.size, x.size):
        raise ValueError("binders must have shape (n_sizes, n_control)")
    if x.size < 2 or system_sizes.size < 2:
        raise ValueError("at least two controls and two sizes are required")
    order = np.argsort(x)
    x = x[order]
    curves = curves[:, order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("control values must be unique")

    sorted_sizes = np.argsort(system_sizes)
    if adjacent_sizes:
        pairs = list(zip(sorted_sizes[:-1], sorted_sizes[1:]))
    else:
        pairs = list(combinations(sorted_sizes.tolist(), 2))
    crossings: list[dict[str, float | int]] = []
    for first, second in pairs:
        difference = curves[first] - curves[second]
        for index in range(x.size - 1):
            left, right = float(difference[index]), float(difference[index + 1])
            if left == 0.0:
                crossing = float(x[index])
            elif left * right < 0.0:
                fraction = -left / (right - left)
                crossing = float(x[index] + fraction * (x[index + 1] - x[index]))
            else:
                continue
            if crossings and (
                crossings[-1]["size_index_a"] == int(first)
                and crossings[-1]["size_index_b"] == int(second)
                and np.isclose(crossings[-1]["control"], crossing)
            ):
                continue
            crossings.append(
                {
                    "size_index_a": int(first),
                    "size_index_b": int(second),
                    "size_a": float(system_sizes[first]),
                    "size_b": float(system_sizes[second]),
                    "control": crossing,
                }
            )
    locations = np.asarray([entry["control"] for entry in crossings], dtype=float)
    estimate = float(np.median(locations)) if locations.size else float("nan")
    spread = (
        float(1.4826 * np.median(np.abs(locations - estimate)))
        if locations.size > 1
        else float("nan")
    )
    return {
        "crossings": crossings,
        "crossing_locations": locations,
        "crossing_estimate": estimate,
        "crossing_spread": spread,
        "n_crossings": int(locations.size),
    }


def finite_difference_response(
    control: ArrayLike,
    observable: ArrayLike,
) -> dict[str, Any]:
    """Estimate a response curve and its largest absolute slope."""

    x = as_float_array(control, name="control", ndim=1)
    y = as_float_array(observable, name="observable", ndim=1)
    if x.size != y.size or x.size < 2:
        raise ValueError("control and observable must have equal length >= 2")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("control values must be unique")
    edge_order = 2 if x.size >= 3 else 1
    response = np.gradient(y, x, edge_order=edge_order)
    peak_index = int(np.argmax(np.abs(response)))
    return {
        "control": x,
        "response": response,
        "absolute_response": np.abs(response),
        "peak_index": peak_index,
        "peak_control": float(x[peak_index]),
        "peak_response": float(response[peak_index]),
    }


def estimate_transition_boundary(
    control: ArrayLike,
    signal: ArrayLike,
    *,
    method: Literal["peak", "max_gradient", "threshold"] = "max_gradient",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Estimate a one-dimensional boundary from a diagnostic curve."""

    x = as_float_array(control, name="control", ndim=1)
    y = as_float_array(signal, name="signal", ndim=1)
    if x.size != y.size or x.size < 2:
        raise ValueError("control and signal must have equal length >= 2")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("control values must be unique")
    if method == "peak":
        index = int(np.argmax(y))
        return {"boundary": float(x[index]), "index": index, "method": method}
    if method == "max_gradient":
        response = finite_difference_response(x, y)
        return {
            "boundary": response["peak_control"],
            "index": response["peak_index"],
            "method": method,
            "response": response,
        }
    if method == "threshold":
        if threshold is None or not np.isfinite(threshold):
            raise ValueError("a finite threshold is required for threshold method")
        difference = y - threshold
        candidates: list[float] = []
        for index in range(x.size - 1):
            left, right = difference[index], difference[index + 1]
            if left == 0:
                candidates.append(float(x[index]))
            elif left * right < 0:
                fraction = -left / (right - left)
                candidates.append(float(x[index] + fraction * (x[index + 1] - x[index])))
        return {
            "boundary": candidates[0] if candidates else float("nan"),
            "candidates": np.asarray(candidates),
            "method": method,
        }
    raise ValueError("unknown boundary method")


def classify_phase(
    m_pos: float,
    m_sem: float,
    *,
    order_threshold: float = 0.25,
    dominance_ratio: float = 1.5,
) -> dict[str, Any]:
    """Conservatively label positional, semantic, coexistence, or unresolved order.

    Signs are preserved in the returned values, but phase strength uses their
    magnitudes because template orientation is conventional. A phase is called
    dominant only when it exceeds both the absolute threshold and the requested
    ratio over the other component.
    """

    if not np.isfinite(m_pos) or not np.isfinite(m_sem):
        return {"label": "unresolved", "reason": "non-finite order parameter"}
    if order_threshold < 0 or dominance_ratio < 1:
        raise ValueError("order_threshold must be non-negative and dominance_ratio >= 1")
    positional, semantic = abs(float(m_pos)), abs(float(m_sem))
    pos_ordered = positional >= order_threshold
    sem_ordered = semantic >= order_threshold
    if pos_ordered and sem_ordered:
        if positional >= dominance_ratio * semantic:
            label = "positional"
        elif semantic >= dominance_ratio * positional:
            label = "semantic"
        else:
            label = "coexistence"
    elif pos_ordered and positional >= dominance_ratio * max(semantic, np.finfo(float).eps):
        label = "positional"
    elif sem_ordered and semantic >= dominance_ratio * max(positional, np.finfo(float).eps):
        label = "semantic"
    else:
        label = "disordered_or_unresolved"
    return {
        "label": label,
        "m_pos": float(m_pos),
        "m_sem": float(m_sem),
        "order_threshold": float(order_threshold),
        "dominance_ratio": float(dominance_ratio),
    }


def classify_transition_evidence(
    *,
    n_sizes: int,
    susceptibility_peak_growth: float | None = None,
    binder_crossing_spread: float | None = None,
    data_collapse_score: float | None = None,
    bimodal: bool = False,
    barrier_growth: bool = False,
    hysteresis: bool = False,
    response_peak_growth: float | None = None,
    maximum_crossing_spread: float = 0.1,
    maximum_collapse_score: float = 0.2,
) -> dict[str, Any]:
    """Classify transition evidence without over-claiming universality.

    A continuous-transition label requires at least five sizes and three
    mutually consistent finite-size diagnostics. A first-order candidate needs
    two direct coexistence diagnostics. Everything else remains a crossover
    candidate or insufficient evidence.
    """

    if n_sizes < 1:
        raise ValueError("n_sizes must be positive")
    first_order_signs = {
        "bimodal_distribution": bool(bimodal),
        "barrier_growth": bool(barrier_growth),
        "hysteresis": bool(hysteresis),
    }
    n_first_order = sum(first_order_signs.values())
    continuous_signs = {
        "susceptibility_peak_grows": susceptibility_peak_growth is not None
        and susceptibility_peak_growth > 0,
        "stable_binder_crossing": binder_crossing_spread is not None
        and np.isfinite(binder_crossing_spread)
        and binder_crossing_spread <= maximum_crossing_spread,
        "acceptable_data_collapse": data_collapse_score is not None
        and np.isfinite(data_collapse_score)
        and data_collapse_score <= maximum_collapse_score,
    }
    if n_sizes >= 3 and n_first_order >= 2:
        label = "first_order_candidate"
    elif n_sizes >= 5 and all(continuous_signs.values()) and n_first_order == 0:
        label = "continuous_transition_candidate"
    elif (
        n_sizes >= 3
        and n_first_order == 0
        and response_peak_growth is not None
        and response_peak_growth <= 0
        and not continuous_signs["susceptibility_peak_grows"]
    ):
        label = "crossover_candidate"
    else:
        label = "insufficient_evidence"
    return {
        "label": label,
        "n_sizes": int(n_sizes),
        "first_order_signs": first_order_signs,
        "continuous_signs": continuous_signs,
        "thresholds": {
            "maximum_crossing_spread": float(maximum_crossing_spread),
            "maximum_collapse_score": float(maximum_collapse_score),
        },
    }

