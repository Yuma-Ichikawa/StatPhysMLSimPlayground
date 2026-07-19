"""Numerically safe common phase coordinates shared by every domain."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

EPS = 1e-12

COMMON_METRICS = (
    "order_parameter",
    "susceptibility",
    "binder_cumulant",
    "generalization_error",
    "ood_generalization_error",
    "effective_multiplicity",
    "interaction_range",
    "macrostate_entropy",
    "oracle_gap",
    "intervention_response",
)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _floating(probability: torch.Tensor) -> torch.Tensor:
    return probability if probability.is_floating_point() else probability.float()


def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = _floating(probability)
    eps = torch.finfo(probability.dtype).eps
    p = probability.clamp(min=eps, max=1.0 - eps)
    return -(p * p.log() + (1.0 - p) * torch.log1p(-p))


def categorical_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probabilities = _floating(probabilities)
    eps = torch.finfo(probabilities.dtype).eps
    p = probabilities.clamp_min(0.0)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def effective_multiplicity(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return categorical_entropy(probabilities, dim=dim).exp()


def susceptibility(samples: torch.Tensor, size: int) -> float:
    x = _floating(samples).double().reshape(-1)
    return float(int(size) * x.var(unbiased=False).item())


def binder_cumulant(samples: torch.Tensor) -> float:
    x = _floating(samples).double().reshape(-1)
    second = x.square().mean()
    if second <= torch.finfo(second.dtype).eps:
        return 0.0
    fourth = x.pow(4).mean()
    return float((1.0 - fourth / (3.0 * second.square())).item())


def phase_statistics(
    signed_samples: torch.Tensor, *, size: int, order_absolute: bool = True
) -> dict[str, float]:
    x = _floating(signed_samples).double().reshape(-1)
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("signed phase samples must be non-empty and finite")
    order = x.abs().mean() if order_absolute else x.mean()
    positive = (x > 0).double().mean()
    return {
        "order_parameter": float(order.item()),
        "signed_order_parameter": float(x.mean().item()),
        "susceptibility": susceptibility(x, size),
        "binder_cumulant": binder_cumulant(x),
        "macrostate_entropy": float(binary_entropy(positive).item()),
    }


def finite_scalar(value: Any, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"metric {name!r} is not finite: {scalar}")
    return scalar


def validate_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in COMMON_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"missing common metrics: {', '.join(missing)}")
    return {str(name): finite_scalar(value, str(name)) for name, value in metrics.items()}


__all__ = [
    "COMMON_METRICS",
    "binary_entropy",
    "binder_cumulant",
    "categorical_entropy",
    "effective_multiplicity",
    "finite_scalar",
    "phase_statistics",
    "resolve_device",
    "seed_everything",
    "susceptibility",
    "validate_metrics",
]
