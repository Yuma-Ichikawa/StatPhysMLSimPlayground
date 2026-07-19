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
            terms.append(np.sum(probability[mask] * np.log(probability[mask] / np.maximum(middle[mask], 1e-12))))
        output.append(0.5 * sum(terms))
    return np.asarray(output)


def fisher_sensitivity(histograms: np.ndarray, controls: np.ndarray) -> np.ndarray:
    probability = np.asarray(histograms, dtype=float)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    score = np.gradient(np.log(np.maximum(probability, 1e-12)), np.asarray(controls), axis=0)
    return np.sum(probability * score**2, axis=1)


def propose_boundaries(histograms: np.ndarray, controls: np.ndarray, count: int = 3) -> list[float]:
    js = adjacent_js(histograms)
    fisher = fisher_sensitivity(histograms, controls)
    midpoint = 0.5 * (controls[:-1] + controls[1:])
    candidates = [(float(score), float(control)) for score, control in zip(js, midpoint, strict=True)]
    candidates += [(float(score), float(control)) for score, control in zip(fisher, controls, strict=True)]
    return [control for _, control in sorted(candidates, reverse=True)[:count]]
