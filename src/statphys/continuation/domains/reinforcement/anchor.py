"""Finite hierarchical decision problem with oracle and proxy rewards."""

from __future__ import annotations

from typing import Any

import torch

from ...metrics import (
    EPS,
    categorical_entropy,
    effective_multiplicity,
    phase_statistics,
    seed_everything,
)
from ...schema import TaskSpec


def _group_probabilities(policy: torch.Tensor, action_groups: torch.Tensor, groups: int):
    result = torch.zeros((policy.shape[0], groups), device=policy.device)
    result.scatter_add_(1, action_groups.expand(policy.shape[0], -1), policy)
    return result


def run_reinforcement(
    task: TaskSpec,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    generator = torch.Generator(device=device).manual_seed(task.seed)
    contexts = task.size
    actions = int(task.parameters.get("actions", 16))
    groups = int(task.parameters.get("action_groups", 4))
    if actions % groups:
        raise ValueError("actions must be divisible by action_groups")
    action_groups = (torch.arange(actions, device=device) % groups).long()[None, :]
    context_groups = (torch.arange(contexts, device=device) % groups).long()
    matching = (action_groups == context_groups[:, None]).float()
    local_reward = 0.25 * torch.randn(
        (contexts, actions), generator=generator, device=device
    )
    true_reward = matching + local_reward
    true_reward = true_reward - true_reward.mean(dim=1, keepdim=True)

    correlation = float(task.parameters.get("proxy_correlation", 0.75))
    proxy_noise = torch.randn(
        true_reward.shape, generator=generator, device=device
    )
    exploit = torch.zeros_like(true_reward)
    exploit[:, 0] = float(task.parameters.get("exploit_bonus", 1.25))
    proxy_reward = correlation * true_reward + (1.0 - correlation) * proxy_noise + exploit

    pressure = max(float(task.control), 0.0)
    oracle_policy = torch.softmax(pressure * true_reward, dim=1)
    variant = task.variant.lower()
    if variant == "oracle":
        policy = oracle_policy
        optimized_reward = true_reward
    elif variant == "proxy":
        policy = torch.softmax(pressure * proxy_reward, dim=1)
        optimized_reward = proxy_reward
    elif variant in {"kl_regularized", "regularized"}:
        coefficient = float(task.parameters.get("kl_coefficient", 0.5))
        effective_pressure = pressure / (1.0 + coefficient * pressure)
        policy = torch.softmax(effective_pressure * proxy_reward, dim=1)
        optimized_reward = proxy_reward
    else:
        raise ValueError(f"unknown RL variant: {task.variant}")

    optimal_actions = true_reward.argmax(dim=1)
    overlap = policy.gather(1, optimal_actions[:, None]).squeeze(1)
    signed_order = 2.0 * overlap - 1.0
    achieved_true = (policy * true_reward).sum(dim=1)
    oracle_true = (oracle_policy * true_reward).sum(dim=1)
    achieved_proxy = (policy * proxy_reward).sum(dim=1)
    oracle_gap = oracle_true - achieved_true

    ood_true = true_reward.clone()
    nonmatching_exploit = context_groups != 0
    ood_true[nonmatching_exploit, 0] -= float(task.parameters.get("ood_penalty", 1.5))
    ood_oracle = torch.softmax(pressure * ood_true, dim=1)
    ood_gap = (ood_oracle * ood_true).sum(dim=1) - (policy * ood_true).sum(dim=1)

    group_policy = _group_probabilities(policy, action_groups, groups)
    macro_entropy = categorical_entropy(group_policy)
    group_positions = torch.arange(groups, device=device, dtype=torch.float32)
    group_distance = (
        group_positions[:, None] - group_positions[None, :]
    ).abs() / max(1, groups - 1)
    interaction = torch.einsum(
        "bi,ij,bj->b", group_policy, group_distance, group_policy
    )
    collapsed_mass = policy[:, 0]
    low_pressure = max(0.25, min(pressure, 1.0))
    reference_policy = torch.softmax(low_pressure * proxy_reward, dim=1)
    intervention = 0.5 * (policy - reference_policy).abs().sum(dim=1)

    metrics = phase_statistics(signed_order, size=contexts, order_absolute=False)
    metrics.update(
        {
            "generalization_error": float(oracle_gap.mean()),
            "ood_generalization_error": float(ood_gap.mean()),
            "effective_multiplicity": float(effective_multiplicity(policy).mean()),
            "interaction_range": float(interaction.mean()),
            "macrostate_entropy": float(macro_entropy.mean()),
            "oracle_gap": float(oracle_gap.mean()),
            "intervention_response": float(intervention.mean()),
            "true_reward": float(achieved_true.mean()),
            "proxy_reward": float(achieved_proxy.mean()),
            "exploit_mass": float(collapsed_mass.mean()),
            "policy_entropy": float(categorical_entropy(policy).mean()),
        }
    )
    arrays = {
        "signed_order_samples": signed_order.detach().cpu().numpy(),
        "oracle_gap": oracle_gap.detach().cpu().numpy(),
        "ood_gap": ood_gap.detach().cpu().numpy(),
        "policy_entropy": categorical_entropy(policy).detach().cpu().numpy(),
        "exploit_mass": collapsed_mass.detach().cpu().numpy(),
    }
    return metrics, arrays


__all__ = ["run_reinforcement"]
