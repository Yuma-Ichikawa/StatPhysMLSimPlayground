"""Trainable independent, message-passing, and attention-based agent populations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.metrics import seed_everything
from ...core.schema import TaskSpec
from ..common import common_coordinates, effective_count, entropy, task_rng


class _AgentPopulation(nn.Module):
    def __init__(self, variant: str, width: int, agents: int) -> None:
        super().__init__()
        self.variant = variant
        self.local = nn.Sequential(nn.Linear(1, width), nn.Tanh())
        if variant == "independent":
            self.readout = nn.Linear(width, 2)
        elif variant == "message_passing":
            self.message = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh())
            self.readout = nn.Linear(width, 2)
        elif variant == "attention_pool":
            heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
            layer = nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=4 * width, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            self.attention = nn.TransformerEncoder(layer, num_layers=2)
            self.position = nn.Parameter(torch.zeros(1, agents, width))
            self.readout = nn.Linear(width, 2)
        else:
            raise ValueError(
                "learned agent variant must be independent, message_passing, or attention_pool"
            )

    def forward(
        self, observations: torch.Tensor, communication_keep: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local(observations)
        if self.variant == "message_passing":
            message = local.mean(dim=1, keepdim=True) * communication_keep
            local = self.message(torch.cat((local, message.expand_as(local)), dim=2))
        elif self.variant == "attention_pool":
            local = self.attention(local + self.position[:, : local.shape[1]] * communication_keep)
            local = communication_keep * local + (1.0 - communication_keep) * self.local(observations)
        agent_logits = self.readout(local)
        collective_logits = agent_logits.mean(dim=1)
        return collective_logits, agent_logits


def _worlds(
    rng: np.random.Generator,
    worlds: int,
    agents: int,
    *,
    reliability_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = rng.integers(0, 2, size=worlds)
    reliability = np.linspace(0.58, 0.92, agents) + reliability_shift
    reliability = np.clip(reliability, 0.51, 0.99)
    correct = rng.random((worlds, agents)) < reliability[None]
    observations = np.where(correct, hidden[:, None], 1 - hidden[:, None])
    return (2.0 * observations.astype(np.float32) - 1.0)[..., None], hidden.astype(np.int64)


def run_learned_agents(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    rng = task_rng(task, "learned_agents")
    agents = max(4, int(task.size))
    width = max(8, min(int(task.parameters.get("model_width", 32)), 96))
    worlds = max(16, int(task.parameters.get("worlds", 256)))
    n_probe = max(16, int(task.parameters.get("n_probe", 256)))
    communication_dropout = float(np.clip(task.control, 0.0, 1.0))
    keep = 0.0 if task.variant == "independent" else 1.0 - communication_dropout
    train_obs, train_hidden = _worlds(rng, worlds, agents)
    observations = torch.as_tensor(train_obs, device=device)
    targets = torch.as_tensor(train_hidden, device=device)

    model = _AgentPopulation(task.variant, width, agents).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.parameters.get("learning_rate", 3e-3)))
    steps = max(1, int(task.parameters.get("steps", 72)))
    individual_weight = float(task.parameters.get("individual_weight", 0.2))
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        collective, individual = model(observations, keep)
        loss = F.cross_entropy(collective, targets)
        repeated_targets = targets[:, None].expand(-1, agents)
        loss = loss + individual_weight * F.cross_entropy(
            individual.flatten(0, 1), repeated_targets.flatten()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    test_obs, hidden = _worlds(rng, n_probe, agents)
    ood_obs, ood_hidden = _worlds(rng, n_probe, agents, reliability_shift=-0.12)
    test_obs_t = torch.as_tensor(test_obs, device=device)
    hidden_t = torch.as_tensor(hidden, device=device)
    ood_obs_t = torch.as_tensor(ood_obs, device=device)
    ood_hidden_t = torch.as_tensor(ood_hidden, device=device)
    intervened = test_obs_t.clone()
    committed = max(1, int(round(0.2 * agents)))
    intervened[:, :committed] *= -1.0

    model.eval()
    with torch.no_grad():
        collective, individual = model(test_obs_t, keep)
        collective_probabilities = collective.softmax(dim=1)
        actions = collective.argmax(dim=1)
        agent_actions = individual.argmax(dim=2)
        ood_actions = model(ood_obs_t, keep)[0].argmax(dim=1)
        intervened_actions = model(intervened, keep)[0].argmax(dim=1)
    success = actions.eq(hidden_t)
    ood_success = ood_actions.eq(ood_hidden_t)
    intervention_success = intervened_actions.eq(hidden_t)
    accuracy = float(success.float().mean())
    ood_accuracy = float(ood_success.float().mean())
    intervention_delta = accuracy - float(intervention_success.float().mean())
    consensus = float(agent_actions.eq(actions[:, None]).float().mean())
    individual_accuracy = float(agent_actions.eq(hidden_t[:, None]).float().mean())
    margins = individual.softmax(dim=2)[:, :, 1].std(dim=0).detach().cpu().numpy()
    influence = margins + 1e-8
    influence /= influence.sum()
    effective_agents = float(effective_count(influence))
    observed_majority = (test_obs_t.squeeze(-1).sum(dim=1) > 0).long()
    oracle_accuracy = float(observed_majority.eq(hidden_t).float().mean())
    initial_error = float((test_obs_t.squeeze(-1).sign() != (2 * hidden_t[:, None] - 1)).float().mean())
    collective_error = 1.0 - accuracy
    signed = 2.0 * success.detach().cpu().numpy().astype(np.float64) - 1.0
    metrics, arrays = common_coordinates(
        signed,
        size=agents,
        generalization_error=1.0 - accuracy,
        ood_generalization_error=1.0 - ood_accuracy,
        effective_multiplicity=effective_agents,
        interaction_range=float(keep),
        oracle_gap=float(max(0.0, oracle_accuracy - accuracy)),
        intervention_response=float(intervention_delta),
        extras={
            "collective_accuracy": accuracy,
            "individual_accuracy": individual_accuracy,
            "consensus": consensus,
            "polarization": 1.0 - consensus,
            "effective_agents": effective_agents,
            "influence_concentration": float(np.sum(influence**2)),
            "error_reproduction_number": collective_error / max(initial_error, 1e-8),
            "global_flip_response": intervention_delta,
            "message_entropy": float(entropy(influence)),
            "training_loss": losses[-1],
            "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        },
    )
    arrays.update(
        loss_curve=np.asarray(losses, dtype=np.float32),
        influence=influence.astype(np.float32),
        collective_probabilities=collective_probabilities.detach().cpu().numpy().astype(np.float32),
        agent_actions=agent_actions.detach().cpu().numpy().astype(np.int8),
    )
    return metrics, arrays


__all__ = ["run_learned_agents"]
