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


def mlp_mechanism_statistics(
    activations: torch.Tensor,
    gates: torch.Tensor,
    *,
    gated: bool,
) -> dict[str, float]:
    """Return bounded MLP occupancy and gate observables from a fixed probe set."""
    values = activations.detach().float()
    scale = values.square().mean().sqrt().clamp_min(1e-12)
    sparsity = (values.abs() <= 0.05 * scale).float().mean()
    if gated:
        gate_values = gates.detach().float()
        gate_scale = gate_values.square().mean().sqrt().clamp_min(1e-12)
        saturation = (gate_values.abs() >= 2.0 * gate_scale).float().mean()
    else:
        saturation = values.new_zeros(())
    return {
        "mlp_activation_sparsity": float(sparsity),
        "mlp_gate_saturation": float(saturation),
    }


def local_mlp_jacobian_participation(mlp: torch.nn.Module, inputs: torch.Tensor) -> float:
    """Estimate a local MLP Jacobian effective rank from one registered token."""
    probe = inputs.detach().float().reshape(-1).requires_grad_(True)

    def mapping(vector: torch.Tensor) -> torch.Tensor:
        output, _, _ = mlp(vector.reshape(1, 1, -1))
        return output.reshape(-1)

    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(mapping, probe, vectorize=True)
    singular = torch.linalg.svdvals(jacobian.detach().float())
    eigenvalues = singular.square()
    participation = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-20)
    return float(participation / max(eigenvalues.numel(), 1))


def residual_stream_statistics(representations: torch.Tensor) -> dict[str, float]:
    """Bounded residual-stream scale, drift, and nearest-layer correlation."""
    values = representations.detach().float()
    rms = values.square().mean().sqrt()
    token_drift = values.mean(dim=-1).abs().mean() / rms.clamp_min(1e-12)
    if values.shape[0] < 2:
        correlation = values.new_ones(())
    else:
        left = values[:-1].reshape(values.shape[0] - 1, -1)
        right = values[1:].reshape(values.shape[0] - 1, -1)
        left = left - left.mean(dim=-1, keepdim=True)
        right = right - right.mean(dim=-1, keepdim=True)
        correlation = (left * right).sum(dim=-1) / (
            left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-12)
        correlation = correlation.mean()
    return {
        "residual_stream_rms": float(rms),
        "residual_token_mean_drift": float(token_drift),
        "residual_depth_correlation": float(correlation),
    }


def gradient_noise_scale(first: torch.Tensor, second: torch.Tensor) -> float:
    """Dimensionless two-minibatch gradient-noise estimate."""
    mean = 0.5 * (first + second)
    noise = 0.5 * (first - second)
    return float(noise.square().sum() / mean.square().sum().clamp_min(1e-20))


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
    """Return auditable raw ablation risks and bounded signed effects.

    The previous normalization by ``1 - full`` diverged when the full-model
    risk approached one.  A risk delta is now mapped to ``[-1, 1]`` using
    ``delta / (abs(delta) + full + eps)`` while retaining every raw quantity.
    """
    risks = (float(full), float(no_attention), float(no_mlp), float(neither))
    if not all(math.isfinite(value) and value >= 0.0 for value in risks):
        raise ValueError("causal ablation risks must be finite and non-negative")
    eps = 1e-12
    attention_delta = no_attention - full
    mlp_delta = no_mlp - full
    joint_delta = neither - full
    synergy_delta = no_attention + no_mlp - neither - full

    def bounded(delta: float) -> float:
        return float(delta / (abs(delta) + full + eps))

    return {
        "full_risk": float(full),
        "attention_ablated_risk": float(no_attention),
        "mlp_ablated_risk": float(no_mlp),
        "attention_mlp_ablated_risk": float(neither),
        "attention_risk_delta": float(attention_delta),
        "mlp_risk_delta": float(mlp_delta),
        "attention_mlp_risk_delta": float(joint_delta),
        "attention_mlp_synergy_risk_delta": float(synergy_delta),
        "attention_causal_effect": bounded(attention_delta),
        "mlp_causal_effect": bounded(mlp_delta),
        "attention_mlp_causal_effect": bounded(joint_delta),
        "attention_mlp_synergy_effect": bounded(synergy_delta),
    }
