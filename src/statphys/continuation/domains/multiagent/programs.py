"""Population, debate, critical-minority, influence, and role-specialization programs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, entropy, task_rng


def _network(rng: np.random.Generator, agents: int, variant: str) -> np.ndarray:
    adjacency = np.zeros((agents, agents))
    if variant in {"complete", "all_to_all", "consensus"}:
        adjacency[:] = 1.0
        np.fill_diagonal(adjacency, 0.0)
    elif variant in {"hierarchical", "roles", "leader"}:
        adjacency[:, 0] = 1.0
        adjacency[0, :] = 0.5
        np.fill_diagonal(adjacency, 0.0)
    else:
        degree = min(max(2, int(round(math.sqrt(agents)))), agents - 1)
        for agent in range(agents):
            for offset in range(1, degree // 2 + 1):
                adjacency[agent, (agent + offset) % agents] = 1.0
                adjacency[agent, (agent - offset) % agents] = 1.0
        if variant in {"small_world", "randomized", "debate"}:
            rewired = rng.random((agents, agents)) < min(0.1, 4.0 / agents)
            adjacency = np.maximum(adjacency, rewired.astype(float))
            np.fill_diagonal(adjacency, 0.0)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(row_sum, 1.0)


def run_multiagent_program(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    rng = task_rng(task, "multiagent")
    agents = max(4, min(int(task.size), int(task.parameters.get("agent_cap", 512))))
    worlds = max(64, int(task.parameters.get("worlds", 256)))
    steps = max(3, int(task.parameters.get("steps", 12)))
    influence = _network(rng, agents, task.variant)
    truth = rng.choice((-1.0, 1.0), size=worlds)
    reliability = np.clip(
        rng.normal(float(task.parameters.get("reliability", 0.65)), 0.08, size=agents), 0.5, 0.95
    )
    signal = np.where(
        rng.random((worlds, agents)) < reliability[None, :], truth[:, None], -truth[:, None]
    )
    belief = signal.astype(float)
    control = max(float(task.control), 0.0)
    committed_mask = np.zeros(agents, dtype=bool)
    committed_value = np.zeros(agents)
    if task.family == "minority":
        fraction = np.clip(control, 0.0, 0.5)
        committed = max(1, int(round(fraction * agents)))
        committed_mask[:committed] = True
        committed_value[:committed] = 1.0 if task.variant != "incorrect" else -1.0
    polarization_path, consensus_path, accuracy_path = [], [], []
    initial_majority = np.sign(signal.mean(axis=1))
    for _ in range(steps):
        social = belief @ influence.T
        if task.family == "debate" and task.variant in {"no_communication", "independent"}:
            social[:] = 0.0
        elif task.family == "debate" and task.variant == "one_way":
            social[:, agents // 2 :] = 0.0
        evidence_weight = 1.0 if task.family == "roles" else 0.5
        belief = np.tanh(evidence_weight * signal + control * social)
        if committed_mask.any():
            belief[:, committed_mask] = committed_value[committed_mask]
        group_gap = belief[:, : agents // 2].mean(axis=1) - belief[:, agents // 2 :].mean(axis=1)
        polarization_path.append(float(np.mean(np.abs(group_gap))))
        consensus_path.append(float(np.mean(np.abs(belief.mean(axis=1)))))
        accuracy_path.append(float(np.mean(np.sign(belief.mean(axis=1)) == truth)))
    decision = np.sign(belief.mean(axis=1))
    signed = decision * truth
    baseline_accuracy = float(np.mean(initial_majority == truth))
    final_accuracy = float(np.mean(decision == truth))
    column_influence = influence.mean(axis=0)
    column_influence /= max(column_influence.sum(), 1e-12)
    eigenvalues = np.linalg.eigvals(influence)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    initial_error = float(np.mean(signal != truth[:, None]))
    error_r0 = control * spectral_radius * initial_error
    role_profile = np.mean(np.abs(belief), axis=0)
    role_profile /= max(role_profile.sum(), 1e-12)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - final_accuracy,
        ood_generalization_error=min(1.0, 1.0 - final_accuracy + float(np.std(reliability))),
        effective_multiplicity=float(np.exp(entropy(column_influence))),
        interaction_range=float(np.mean(np.count_nonzero(influence, axis=1)) / agents),
        oracle_gap=1.0 - final_accuracy,
        intervention_response=final_accuracy - baseline_accuracy,
        extras={
            "consensus": consensus_path[-1],
            "polarization": polarization_path[-1],
            "collective_accuracy": final_accuracy,
            "influence_concentration": float(np.sum(column_influence**2)),
            "effective_agents": float(np.exp(entropy(column_influence))),
            "error_reproduction_number": error_r0,
            "role_specialization": float(1.0 - entropy(role_profile) / math.log(agents)),
            "critical_minority_fraction": float(np.mean(committed_mask)),
            "cooperation_gain": final_accuracy - baseline_accuracy,
        },
    )
    metrics, arrays = result
    arrays.update(
        consensus_trajectory=np.asarray(consensus_path, dtype=np.float32),
        polarization_trajectory=np.asarray(polarization_path, dtype=np.float32),
        accuracy_trajectory=np.asarray(accuracy_path, dtype=np.float32),
        influence_matrix=influence.astype(np.float32),
        final_belief=belief.astype(np.float32),
    )
    return metrics, arrays
