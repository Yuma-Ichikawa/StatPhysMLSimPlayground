"""Finite-MDP RL thermodynamics with exact soft-policy oracles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, entropy, softmax, task_rng


def _problem(task: TaskSpec):
    rng = task_rng(task, "mdp")
    states = max(4, min(int(task.size), int(task.parameters.get("state_cap", 128))))
    actions = max(2, int(task.parameters.get("actions", 4)))
    transition = np.zeros((states, actions, states))
    for state in range(states):
        for action in range(actions):
            target = (state + action - actions // 2) % states
            transition[state, action] = 0.1 / states
            transition[state, action, target] += 0.9
    true_reward = rng.normal(scale=0.2, size=(states, actions))
    goal = states - 1
    true_reward[:, -1] += np.linspace(-0.5, 1.0, states)
    true_reward[goal] += 1.0
    feature = np.linspace(-1.0, 1.0, states)[:, None] * np.linspace(0.5, 1.5, actions)[None, :]
    return rng, transition, true_reward, feature


def _soft_value_iteration(transition, reward, temperature, gamma=0.95, iterations=500):
    states, actions = reward.shape
    value = np.zeros(states)
    temperature = max(float(temperature), 1e-4)
    for _ in range(iterations):
        q_value = reward + gamma * np.einsum("sak,k->sa", transition, value)
        updated = temperature * np.log(np.exp((q_value - q_value.max(axis=1, keepdims=True)) / temperature).sum(axis=1)) + q_value.max(axis=1)
        if np.max(np.abs(updated - value)) < 1e-10:
            value = updated
            break
        value = updated
    q_value = reward + gamma * np.einsum("sak,k->sa", transition, value)
    return softmax(q_value / temperature, axis=1), value, q_value


def _occupancy(transition, policy):
    kernel = np.einsum("sa,sak->sk", policy, transition)
    occupancy = np.full(kernel.shape[0], 1.0 / kernel.shape[0])
    for _ in range(10000):
        updated = occupancy @ kernel
        if np.max(np.abs(updated - occupancy)) < 1e-12:
            break
        occupancy = updated
    occupancy = np.maximum(occupancy, 0.0)
    return occupancy / max(occupancy.sum(), 1e-12), kernel


def _rollouts(rng, transition, reward, policy, worlds=128, horizon=64):
    states = rng.integers(0, transition.shape[0], size=worlds)
    returns = np.zeros(worlds)
    path_hash = np.zeros(worlds, dtype=np.uint64)
    entropy_flow = np.empty(horizon)
    for step in range(horizon):
        actions = np.array([rng.choice(policy.shape[1], p=policy[state]) for state in states])
        returns += (0.95**step) * reward[states, actions]
        path_hash = path_hash * np.uint64(1099511628211) ^ (states.astype(np.uint64) * 31 + actions.astype(np.uint64))
        entropy_flow[step] = float(np.mean(entropy(policy[states], axis=1)))
        states = np.array([rng.choice(transition.shape[0], p=transition[state, action]) for state, action in zip(states, actions, strict=True)])
    return returns, path_hash, entropy_flow


def run_reinforcement_program(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    rng, transition, true_reward, proxy_feature = _problem(task)
    pressure = max(float(task.control), 0.0)
    reward_noise = float(task.parameters.get("reward_noise", 0.15))
    proxy_reward = true_reward + pressure * proxy_feature + reward_noise * rng.normal(size=true_reward.shape)
    temperature = float(task.parameters.get("policy_temperature", 0.25))
    oracle_policy, _, _ = _soft_value_iteration(transition, true_reward, temperature)
    if task.family == "entropy_flow":
        learned_reward = true_reward
        learned_temperature = max(pressure, 0.02)
    elif task.family == "goodhart":
        learned_reward = proxy_reward
        learned_temperature = temperature
    elif task.family == "preference":
        comparisons = int(task.parameters.get("comparisons", max(16, task.size * 4)))
        shrinkage = comparisons / (comparisons + task.size * true_reward.shape[1])
        learned_reward = shrinkage * proxy_reward
        learned_temperature = temperature
    else:
        learned_reward = proxy_reward if task.variant not in {"oracle", "exact"} else true_reward
        learned_temperature = temperature
    policy, _, q_value = _soft_value_iteration(transition, learned_reward, learned_temperature)
    if task.family == "optimizer":
        iterations = max(1, int(task.parameters.get("optimization_steps", round(5 + 20 * pressure))))
        approximate = np.full_like(policy, 1.0 / policy.shape[1])
        rate = 0.2 if task.variant in {"policy_gradient", "reinforce"} else 0.5
        for _ in range(iterations):
            approximate = (1.0 - rate) * approximate + rate * policy
        policy = approximate / approximate.sum(axis=1, keepdims=True)
    occupancy, kernel = _occupancy(transition, policy)
    oracle_occupancy, _ = _occupancy(transition, oracle_policy)
    state_action = occupancy[:, None] * policy
    oracle_state_action = oracle_occupancy[:, None] * oracle_policy
    true_return = float(np.sum(state_action * true_reward))
    proxy_return = float(np.sum(state_action * proxy_reward))
    oracle_return = float(np.sum(oracle_state_action * true_reward))
    worlds = max(64, int(task.parameters.get("worlds", 128)))
    horizon = max(16, int(task.parameters.get("horizon", 64)))
    returns, paths, entropy_flow = _rollouts(rng, transition, true_reward, policy, worlds, horizon)
    success_threshold = float(np.median(returns))
    signed = np.where(returns >= success_threshold, 1.0, -1.0)
    occupancy_overlap = float(np.sum(np.sqrt(occupancy * oracle_occupancy)))
    policy_entropy = float(np.sum(occupancy * entropy(policy, axis=1)))
    rollout_multiplicity = float(len(np.unique(paths)))
    response = float(np.mean(np.abs(policy - oracle_policy)))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=max(0.0, oracle_return - true_return),
        ood_generalization_error=float(np.std(returns) / (abs(np.mean(returns)) + 1e-12)),
        effective_multiplicity=float(np.exp(policy_entropy)),
        interaction_range=float(1.0 - np.trace(kernel) / kernel.shape[0]),
        oracle_gap=max(0.0, oracle_return - true_return),
        intervention_response=response,
        extras={
            "true_return": true_return,
            "proxy_return": proxy_return,
            "goodhart_gap": proxy_return - true_return,
            "entropy_rate": policy_entropy,
            "occupancy_overlap": occupancy_overlap,
            "rollout_multiplicity": rollout_multiplicity,
            "entropy_flow": float(entropy_flow[-1] - entropy_flow[0]),
            "policy_susceptibility": float(task.size * np.var(policy.max(axis=1))),
            "q_value_variance": float(np.var(q_value)),
        },
    )
    metrics, arrays = result
    arrays.update(
        policy=policy.astype(np.float32),
        occupancy=occupancy.astype(np.float32),
        oracle_policy=oracle_policy.astype(np.float32),
        rollout_returns=returns.astype(np.float32),
        entropy_trajectory=entropy_flow.astype(np.float32),
    )
    return metrics, arrays
