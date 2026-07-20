"""Dimensionless O(1) losses and component-level order parameters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def intensive_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    objective_lambda: float,
) -> dict[str, torch.Tensor]:
    vocabulary = logits.shape[-1]
    selected = mask.reshape(-1) > 0
    flat_logits = logits.reshape(-1, vocabulary)[selected]
    flat_targets = targets.reshape(-1)[selected]
    ce_nats = F.cross_entropy(flat_logits, flat_targets)
    ce_normalized = ce_nats / math.log(vocabulary)
    probabilities = flat_logits.softmax(dim=-1)
    truth = F.one_hot(flat_targets, vocabulary).to(probabilities.dtype)
    brier = (probabilities - truth).square().sum(dim=-1).mean()
    brier_normalized = brier / (1.0 - 1.0 / vocabulary)
    objective = (1.0 - objective_lambda) * brier_normalized + objective_lambda * ce_normalized
    accuracy = flat_logits.argmax(dim=-1).eq(flat_targets).float().mean()
    return {
        "objective": objective,
        "ce_normalized": ce_normalized,
        "ce_nats": ce_nats,
        "bits_per_byte": ce_nats / math.log(2.0),
        "brier_normalized": brier_normalized,
        "accuracy": accuracy,
    }


def normalized_participation_ratio(activations: torch.Tensor) -> float:
    matrix = activations.detach().float().reshape(-1, activations.shape[-1])
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    eigenvalues = singular.square()
    pr = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-20)
    return float(pr / max(matrix.shape[-1], 1))


def normalized_activation_entropy(activations: torch.Tensor) -> float:
    mass = activations.detach().float().abs().mean(dim=tuple(range(activations.ndim - 1)))
    mass = mass / mass.sum().clamp_min(1e-20)
    entropy = -(mass * mass.clamp_min(1e-20).log()).sum()
    return float(entropy / max(math.log(max(mass.numel(), 2)), 1e-20))


def normalized_attention_entropy(attention: torch.Tensor) -> float:
    probabilities = attention.detach().float().clamp_min(1e-20)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    return float(entropy / max(math.log(max(attention.shape[-1], 2)), 1e-20))


def block_gradient_statistics(model: torch.nn.Module) -> dict[str, float]:
    rms_values = []
    for _, parameter in model.named_parameters():
        if parameter.grad is not None:
            rms_values.append(float(parameter.grad.detach().float().square().mean().sqrt()))
    positive = np.asarray([value for value in rms_values if value > 0], dtype=float)
    if positive.size == 0:
        return {"gradient_rms": 0.0, "gradient_log_cv": 0.0, "gradient_gini": 0.0}
    logs = np.log(positive)
    sorted_values = np.sort(positive)
    count = sorted_values.size
    gini = float(
        (2.0 * np.sum((np.arange(count) + 1) * sorted_values) / (count * sorted_values.sum()))
        - (count + 1.0) / count
    )
    return {
        "gradient_rms": float(np.sqrt(np.mean(positive**2))),
        "gradient_log_cv": float(np.std(logs) / max(abs(np.mean(logs)), 1e-12)),
        "gradient_gini": gini,
    }


def relative_update_statistics(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> dict[str, float]:
    update_square_sum = 0.0
    weight_square_sum = 0.0
    block_ratios = []
    for name, parameter in model.named_parameters():
        reference = initial[name].to(parameter.device).float()
        difference = parameter.detach().float() - reference
        update_square_sum += float(difference.square().sum())
        weight_square_sum += float(reference.square().sum())
        block_weight_rms = float(reference.square().mean().sqrt())
        if block_weight_rms > 1e-12:
            block_ratios.append(float(difference.square().mean().sqrt()) / block_weight_rms)
    return {
        "update_to_weight_rms": math.sqrt(update_square_sum / max(weight_square_sum, 1e-20)),
        "update_to_weight_max": float(np.max(block_ratios, initial=0.0)),
    }


def causal_contributions(
    full: float,
    no_attention: float,
    no_mlp: float,
    neither: float,
) -> dict[str, float]:
    scale = max(1.0 - full, 1e-8)
    return {
        "attention_contribution": (no_attention - full) / scale,
        "mlp_contribution": (no_mlp - full) / scale,
        "attention_mlp_synergy": (no_attention + no_mlp - neither - full) / scale,
    }
