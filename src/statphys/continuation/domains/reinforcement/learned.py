"""Trainable linear, MLP, and Transformer policies in a verifier-noisy POMDP."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.metrics import seed_everything
from ...core.schema import TaskSpec
from ..common import common_coordinates, task_rng


class _Policy(nn.Module):
    def __init__(self, variant: str, width: int, horizon: int) -> None:
        super().__init__()
        self.variant = variant
        if variant == "linear":
            self.policy = nn.Linear(1, 2)
        elif variant == "mlp":
            self.policy = nn.Sequential(nn.Linear(horizon, width), nn.Tanh(), nn.Linear(width, 2))
        elif variant == "transformer":
            dimension = max(8, width)
            while dimension % 4:
                dimension += 1
            self.input = nn.Linear(1, dimension)
            self.position = nn.Parameter(torch.zeros(1, horizon, dimension))
            layer = nn.TransformerEncoderLayer(
                dimension, 4, dim_feedforward=4 * dimension, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.policy = nn.Linear(dimension, 2)
        else:
            raise ValueError(f"learned policy variant must be linear, mlp, or transformer: {variant}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.variant == "linear":
            return self.policy(observations.mean(dim=1))
        if self.variant == "mlp":
            return self.policy(observations.squeeze(-1))
        hidden = self.input(observations) + self.position[:, : observations.shape[1]]
        return self.policy(self.encoder(hidden).mean(dim=1))


def _pomdp(
    rng: np.random.Generator,
    worlds: int,
    horizon: int,
    observation_noise: float,
    verifier_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = rng.integers(0, 2, size=worlds)
    observations = np.repeat(hidden[:, None], horizon, axis=1)
    flips = rng.random((worlds, horizon)) < observation_noise
    observations = np.where(flips, 1 - observations, observations)
    verifier = hidden.copy()
    verifier_flips = rng.random(worlds) < verifier_noise
    verifier[verifier_flips] = 1 - verifier[verifier_flips]
    signed_observations = 2.0 * observations.astype(np.float32) - 1.0
    return signed_observations[..., None], hidden.astype(np.int64), verifier.astype(np.int64)


def run_learned_policy(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    rng = task_rng(task, "learned_policy")
    width = max(4, min(int(task.size), int(task.parameters.get("width_cap", 96))))
    horizon = max(3, int(task.parameters.get("horizon", 8)))
    worlds = max(16, int(task.parameters.get("worlds", 512)))
    n_probe = max(16, int(task.parameters.get("n_probe", 512)))
    observation_noise = float(task.parameters.get("observation_noise", 0.2))
    verifier_noise = float(np.clip(task.control, 0.0, 0.499))
    train_obs, _, train_verifier = _pomdp(
        rng, worlds, horizon, observation_noise, verifier_noise
    )
    observations = torch.as_tensor(train_obs, device=device)
    verifier = torch.as_tensor(train_verifier, device=device)

    model = _Policy(task.variant, width, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.parameters.get("learning_rate", 3e-3)))
    entropy_coefficient = float(task.parameters.get("entropy_coefficient", 0.01))
    steps = max(1, int(task.parameters.get("steps", 96)))
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(observations)
        log_probabilities = logits.log_softmax(dim=1)
        probabilities = log_probabilities.exp()
        verifier_log_probability = log_probabilities.gather(1, verifier[:, None]).mean()
        policy_entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        loss = -verifier_log_probability - entropy_coefficient * policy_entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    test_obs, hidden, test_verifier = _pomdp(
        rng, n_probe, horizon, observation_noise, verifier_noise
    )
    ood_obs, ood_hidden, _ = _pomdp(
        rng, n_probe, horizon, min(0.49, observation_noise + 0.15), verifier_noise
    )
    test_obs_t = torch.as_tensor(test_obs, device=device)
    hidden_t = torch.as_tensor(hidden, device=device)
    verifier_t = torch.as_tensor(test_verifier, device=device)
    ood_obs_t = torch.as_tensor(ood_obs, device=device)
    ood_hidden_t = torch.as_tensor(ood_hidden, device=device)
    intervened = test_obs_t.clone()
    intervened[:, : horizon // 2] = 0.0

    model.eval()
    with torch.no_grad():
        logits = model(test_obs_t)
        probabilities = logits.softmax(dim=1)
        actions = logits.argmax(dim=1)
        ood_actions = model(ood_obs_t).argmax(dim=1)
        intervened_actions = model(intervened).argmax(dim=1)
    true_success = actions.eq(hidden_t)
    proxy_success = actions.eq(verifier_t)
    ood_success = ood_actions.eq(ood_hidden_t)
    intervention_success = intervened_actions.eq(hidden_t)
    majority = (test_obs_t.squeeze(-1).sum(dim=1) > 0).long()
    oracle_accuracy = float(majority.eq(hidden_t).float().mean())
    true_return = float(true_success.float().mean())
    proxy_return = float(proxy_success.float().mean())
    ood_return = float(ood_success.float().mean())
    intervention_delta = true_return - float(intervention_success.float().mean())
    entropy_per_world = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    signed = 2.0 * true_success.detach().cpu().numpy().astype(np.float64) - 1.0
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - true_return,
        ood_generalization_error=1.0 - ood_return,
        effective_multiplicity=float(entropy_per_world.mean().exp()),
        interaction_range=float(max(0.0, intervention_delta)),
        oracle_gap=float(max(0.0, oracle_accuracy - true_return)),
        intervention_response=float(intervention_delta),
        extras={
            "true_return": true_return,
            "proxy_return": proxy_return,
            "goodhart_gap": proxy_return - true_return,
            "policy_entropy": float(entropy_per_world.mean()),
            "pomdp_accuracy": true_return,
            "oracle_pomdp_accuracy": oracle_accuracy,
            "policy_susceptibility": float(task.size * probabilities[:, 1].var(unbiased=False)),
            "verifier_noise": verifier_noise,
            "training_loss": losses[-1],
            "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        },
    )
    arrays.update(
        loss_curve=np.asarray(losses, dtype=np.float32),
        action_probabilities=probabilities.detach().cpu().numpy().astype(np.float32),
        true_success=true_success.detach().cpu().numpy().astype(np.int8),
    )
    return metrics, arrays


__all__ = ["run_learned_policy"]
