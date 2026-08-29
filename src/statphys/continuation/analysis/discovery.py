"""Automatic phase-boundary proposals from JS, Fisher sensitivity, and change points."""

from __future__ import annotations

import numpy as np


def adjacent_js(histograms: np.ndarray) -> np.ndarray:
    histograms = np.asarray(histograms, dtype=float)
    histograms = histograms / np.maximum(histograms.sum(axis=1, keepdims=True), 1e-12)
    output = []
    for left, right in zip(histograms[:-1], histograms[1:], strict=True):
        middle = 0.5 * (left + right)
        terms = []
        for probability in (left, right):
            mask = probability > 0
            terms.append(
                np.sum(
                    probability[mask] * np.log(probability[mask] / np.maximum(middle[mask], 1e-12))
                )
            )
        output.append(0.5 * sum(terms))
    return np.asarray(output)


def fisher_sensitivity(histograms: np.ndarray, controls: np.ndarray) -> np.ndarray:
    probability = np.asarray(histograms, dtype=float)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    score = np.gradient(np.log(np.maximum(probability, 1e-12)), np.asarray(controls), axis=0)
    return np.sum(probability * score**2, axis=1)


def _z_score(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    mean = null.mean(axis=0)
    scale = null.std(axis=0, ddof=1)
    return (observed - mean) / np.maximum(scale, 1e-12)


def discover_boundaries(
    histograms: np.ndarray,
    controls: np.ndarray,
    *,
    count: int = 3,
    permutations: int = 256,
    seed: int = 0,
    minimum_separation: float | None = None,
) -> list[dict[str, float | str]]:
    """Standardize heterogeneous discovery scores against a permutation null.

    ``histograms`` may be one profile or an outer-seed stack with shape
    ``(seed, control, common_bin)``. Candidate generation is exploratory; a
    fresh-seed confirmation step must update ``status`` outside this function.
    """
    profiles = np.asarray(histograms, dtype=float)
    x = np.asarray(controls, dtype=float)
    if profiles.ndim == 2:
        profiles = profiles[None, ...]
    if profiles.ndim != 3 or profiles.shape[1] != len(x):
        raise ValueError("histograms must have shape (seed, control, common_bin)")
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("controls must contain at least three increasing values")
    if count < 1 or permutations < 2:
        raise ValueError("count must be positive and permutations must exceed one")
    rng = np.random.default_rng(seed)
    js_by_seed = np.stack([adjacent_js(profile) for profile in profiles])
    fisher_by_seed = np.stack([fisher_sensitivity(profile, x) for profile in profiles])
    observed_js = js_by_seed.mean(axis=0)
    observed_fisher = fisher_by_seed.mean(axis=0)
    null_js = np.empty((permutations, len(x) - 1), dtype=float)
    null_fisher = np.empty((permutations, len(x)), dtype=float)
    for index in range(permutations):
        permuted = np.stack([profile[rng.permutation(len(x))] for profile in profiles])
        null_js[index] = np.stack([adjacent_js(profile) for profile in permuted]).mean(axis=0)
        null_fisher[index] = np.stack(
            [fisher_sensitivity(profile, x) for profile in permuted]
        ).mean(axis=0)
    midpoint = 0.5 * (x[:-1] + x[1:])
    candidates = [
        {
            "control": float(control),
            "method": "adjacent_js",
            "raw_score": float(raw),
            "standardized_score": float(score),
            "status": "candidate",
        }
        for control, raw, score in zip(
            midpoint, observed_js, _z_score(observed_js, null_js), strict=True
        )
    ]
    candidates.extend(
        {
            "control": float(control),
            "method": "fisher_sensitivity",
            "raw_score": float(raw),
            "standardized_score": float(score),
            "status": "candidate",
        }
        for control, raw, score in zip(
            x, observed_fisher, _z_score(observed_fisher, null_fisher), strict=True
        )
    )
    separation = (
        float(minimum_separation)
        if minimum_separation is not None
        else 0.5 * float(np.median(np.diff(x)))
    )
    selected: list[dict[str, float | str]] = []
    for candidate in sorted(
        candidates, key=lambda item: float(item["standardized_score"]), reverse=True
    ):
        if all(
            abs(float(candidate["control"]) - float(previous["control"])) >= separation
            for previous in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def propose_boundaries(histograms: np.ndarray, controls: np.ndarray, count: int = 3) -> list[float]:
    """Compatibility wrapper returning exploratory candidate locations only."""
    return [
        float(item["control"]) for item in discover_boundaries(histograms, controls, count=count)
    ]
