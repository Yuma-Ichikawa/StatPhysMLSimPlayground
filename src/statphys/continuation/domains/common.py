"""Shared NumPy utilities for compact, reproducible numerical anchors."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping

import numpy as np

from ..core.schema import TaskSpec


def task_rng(task: TaskSpec, namespace: str, inner: int = 0) -> np.random.Generator:
    token = f"{task.seed}:{task.domain.value}:{task.family}:{namespace}:{inner}".encode()
    seed = int.from_bytes(sha256(token).digest()[:8], "big")
    return np.random.default_rng(seed)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(np.clip(shifted, -700.0, 0.0))
    return weights / np.maximum(weights.sum(axis=axis, keepdims=True), np.finfo(float).tiny)


def entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / np.maximum(p.sum(axis=axis, keepdims=True), np.finfo(float).tiny)
    return -np.sum(np.where(p > 0.0, p * np.log(np.maximum(p, np.finfo(float).tiny)), 0.0), axis=axis)


def effective_count(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(entropy(probabilities, axis=axis))


def normalized_overlap(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def common_coordinates(
    signed_samples: np.ndarray,
    *,
    size: int,
    generalization_error: float,
    ood_generalization_error: float | None = None,
    effective_multiplicity: float = 1.0,
    interaction_range: float = 0.0,
    oracle_gap: float = 0.0,
    intervention_response: float = 0.0,
    extras: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    samples = np.asarray(signed_samples, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("signed samples must be finite and non-empty")
    second = float(np.mean(samples**2))
    fourth = float(np.mean(samples**4))
    binder = 0.0 if second <= np.finfo(float).eps else 1.0 - fourth / (3.0 * second**2)
    bins = min(16, max(2, int(np.sqrt(samples.size))))
    scale = max(1.0, float(np.max(np.abs(samples))))
    if float(np.ptp(samples)) <= bins * np.finfo(np.float64).eps * scale:
        histogram = np.asarray([samples.size], dtype=np.float64)
    else:
        histogram, _ = np.histogram(samples, bins=bins)
        histogram = histogram.astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    macro_entropy = float(entropy(histogram))
    metrics: dict[str, float] = {
        "order_parameter": float(np.mean(np.abs(samples))),
        "signed_order_parameter": float(np.mean(samples)),
        "susceptibility": float(int(size) * np.var(samples)),
        "binder_cumulant": float(binder),
        "generalization_error": float(generalization_error),
        "ood_generalization_error": float(
            generalization_error if ood_generalization_error is None else ood_generalization_error
        ),
        "effective_multiplicity": float(effective_multiplicity),
        "interaction_range": float(interaction_range),
        "macrostate_entropy": macro_entropy,
        "oracle_gap": float(oracle_gap),
        "intervention_response": float(intervention_response),
    }
    if extras:
        metrics.update({str(key): float(value) for key, value in extras.items()})
    bad = {name: value for name, value in metrics.items() if not math.isfinite(value)}
    if bad:
        raise ValueError(f"non-finite metrics: {bad}")
    return metrics, {"signed_order_samples": samples.astype(np.float32)}


def ridge(features: np.ndarray, targets: np.ndarray, regularization: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    gram = x.T @ x + max(float(regularization), 1e-10) * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)
