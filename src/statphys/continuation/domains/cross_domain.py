"""Matched-latent cross-domain continuation experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..core.schema import TaskSpec
from .common import common_coordinates, entropy, softmax, task_rng


def run_cross_domain(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    rng = task_rng(task, "cross_domain")
    states = max(8, min(int(task.size), int(task.parameters.get("state_cap", 256))))
    replicas = max(64, int(task.parameters.get("replicas", 256)))
    latent = rng.normal(size=states)
    oracle = int(np.argmax(latent))
    control = max(float(task.control), 1e-3)
    if task.family == "diffusion_language_rl":
        proposal_logits = latent + rng.normal(scale=control, size=(replicas, states))
        proposals = np.argmax(proposal_logits, axis=1)
        selection = softmax(latent[proposals] / control)
        chosen = rng.choice(proposals, size=replicas, p=selection)
    elif task.family == "diffusion_policy_rl":
        policy_logits = latent[None, :] + control * rng.normal(size=(replicas, states))
        policies = softmax(policy_logits, axis=1)
        chosen = np.array([rng.choice(states, p=policy) for policy in policies])
    elif task.family == "multiagent_rl":
        agents = max(4, int(task.parameters.get("agents", 8)))
        beliefs = latent[None, None, :] + control * rng.normal(size=(replicas, agents, states))
        if task.variant == "debate":
            beliefs = 0.5 * beliefs + 0.5 * beliefs.mean(axis=1, keepdims=True)
        chosen = np.argmax(beliefs.mean(axis=1), axis=1)
    elif task.family == "moe_multiagent":
        experts = max(4, int(task.parameters.get("experts", 8)))
        expert_logits = latent[None, :] + control * rng.normal(size=(experts, states))
        router = softmax(np.max(expert_logits, axis=1))
        selected_expert = rng.choice(experts, size=replicas, p=router)
        chosen = np.argmax(expert_logits[selected_expert], axis=1)
    else:
        raise ValueError(f"unsupported cross-domain family {task.family!r}")
    reward = latent[chosen]
    success = chosen == oracle
    signed = 2.0 * success.astype(float) - 1.0
    load = np.bincount(chosen, minlength=states) / replicas
    best_reward = float(latent[oracle])
    regret = best_reward - float(np.mean(reward))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=regret,
        ood_generalization_error=float(np.std(reward)),
        effective_multiplicity=float(np.exp(entropy(load))),
        interaction_range=float(np.mean(np.abs(chosen[:, None] - chosen[None, :])) / max(states - 1, 1)),
        oracle_gap=regret,
        intervention_response=float(abs(np.mean(reward) - np.median(latent))),
        extras={
            "matched_latent_recovery": float(np.mean(success)),
            "macrostate_entropy": float(entropy(load)),
            "replica_overlap": float(np.mean(chosen[:, None] == chosen[None, :])),
            "generalization_response": float(np.var(reward) / control),
            "information_load_ratio": float(math.log(max(states, 2)) / max(task.size, 1)),
            "cross_domain_regret": regret,
        },
    )
    metrics, arrays = result
    arrays.update(chosen_state=chosen, reward=reward.astype(np.float32), state_load=load.astype(np.float32), latent=latent.astype(np.float32))
    return metrics, arrays
