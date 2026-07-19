"""Mean-field-to-local multi-agent continuation via parallel Glauber dynamics."""

from __future__ import annotations

from typing import Any

import torch

from ...metrics import (
    EPS,
    binary_entropy,
    phase_statistics,
    seed_everything,
)
from ...schema import TaskSpec


def _field(
    states: torch.Tensor,
    coupling: float,
    variant: str,
    *,
    bias: float,
    heterogeneity: torch.Tensor | None,
) -> torch.Tensor:
    global_mean = states.mean(dim=1, keepdim=True)
    if variant == "mean_field":
        return coupling * global_mean + bias
    local = 0.5 * (torch.roll(states, 1, dims=1) + torch.roll(states, -1, dims=1))
    if variant == "small_world":
        return coupling * (0.75 * local + 0.25 * global_mean) + bias
    if variant == "heterogeneous":
        assert heterogeneity is not None
        return coupling * heterogeneity * (0.5 * local + 0.5 * global_mean) + bias
    raise ValueError(f"unknown multi-agent variant: {variant}")


def _step(
    states: torch.Tensor,
    coupling: float,
    variant: str,
    temperature: float,
    generator: torch.Generator,
    *,
    bias: float = 0.0,
    heterogeneity: torch.Tensor | None = None,
) -> torch.Tensor:
    field = _field(
        states,
        coupling,
        variant,
        bias=bias,
        heterogeneity=heterogeneity,
    )
    probability = torch.sigmoid(2.0 * field / temperature)
    uniforms = torch.rand(
        states.shape, generator=generator, device=states.device
    )
    return torch.where(uniforms < probability, 1.0, -1.0)


def _mean_field_oracle(coupling: float, temperature: float) -> float:
    value = 1e-3
    for _ in range(10000):
        updated = torch.tanh(torch.tensor(coupling * value / temperature)).item()
        if abs(updated - value) < 1e-12:
            break
        value = updated
    return abs(value)


def run_multiagent(
    task: TaskSpec,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    generator = torch.Generator(device=device).manual_seed(task.seed)
    agents = task.size
    worlds = int(task.parameters.get("worlds", 512))
    steps = int(task.parameters.get("steps", 300))
    temperature = float(task.parameters.get("temperature", 1.0))
    coupling = float(task.control)
    variant = task.variant.lower()
    states = torch.where(
        torch.rand((worlds, agents), generator=generator, device=device) < 0.5,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    heterogeneity = None
    if variant == "heterogeneous":
        heterogeneity = torch.exp(
            float(task.parameters.get("heterogeneity", 0.35))
            * torch.randn((1, agents), generator=generator, device=device)
        )
        heterogeneity = heterogeneity / heterogeneity.mean()

    trajectory: list[float] = []
    record_every = max(1, steps // 50)
    for step in range(steps):
        states = _step(
            states,
            coupling,
            variant,
            temperature,
            generator,
            heterogeneity=heterogeneity,
        )
        if step % record_every == 0 or step + 1 == steps:
            trajectory.append(float(states.mean(dim=1).abs().mean()))

    magnetization = states.mean(dim=1)
    fraction_positive = ((magnetization + 1.0) / 2.0).clamp(EPS, 1.0 - EPS)
    entropy = binary_entropy(fraction_positive)
    multiplicity = entropy.exp()

    flip_bias = float(task.parameters.get("flip_bias", 0.2))
    intervention_steps = int(task.parameters.get("intervention_steps", 80))
    flipped = states.clone()
    response_curve: list[float] = []
    for _ in range(intervention_steps):
        flipped = _step(
            flipped,
            coupling,
            variant,
            temperature,
            generator,
            bias=-flip_bias,
            heterogeneity=heterogeneity,
        )
        response_curve.append(float(flipped.mean(dim=1).mean()))
    final_alignment = -flipped.mean(dim=1)
    adaptation_error = 0.5 * (1.0 - final_alignment)
    changed_fraction = 0.5 * (states - flipped).abs().mean(dim=1)

    if variant == "mean_field":
        interaction_range = 1.0
    elif variant == "small_world":
        interaction_range = min(1.0, 2.0 / max(2, agents) + 0.25)
    else:
        interaction_range = 0.5
    oracle = _mean_field_oracle(coupling, temperature)
    oracle_gap = abs(float(magnetization.abs().mean()) - oracle)

    metrics = phase_statistics(magnetization, size=agents)
    metrics.update(
        {
            "generalization_error": float(1.0 - magnetization.abs().mean()),
            "ood_generalization_error": float(adaptation_error.mean()),
            "effective_multiplicity": float(multiplicity.mean()),
            "interaction_range": float(interaction_range),
            "macrostate_entropy": float(entropy.mean()),
            "oracle_gap": float(oracle_gap),
            "intervention_response": float(changed_fraction.mean()),
            "consensus_fraction": float((magnetization.abs() > 0.8).float().mean()),
            "oracle_magnetization": float(oracle),
            "adaptation_error": float(adaptation_error.mean()),
        }
    )
    arrays = {
        "signed_order_samples": magnetization.detach().cpu().numpy(),
        "adaptation_error": adaptation_error.detach().cpu().numpy(),
        "changed_fraction": changed_fraction.detach().cpu().numpy(),
        "order_trajectory": trajectory,
        "flip_response": response_curve,
    }
    return metrics, arrays


__all__ = ["run_multiagent"]
