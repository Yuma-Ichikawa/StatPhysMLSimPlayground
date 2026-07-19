"""Trainable causal models bridging reduced-order anchors to decoder Transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.metrics import seed_everything
from ...core.schema import TaskSpec
from ..common import common_coordinates, entropy, task_rng


class _CausalTokenModel(nn.Module):
    def __init__(self, variant: str, vocabulary: int, width: int, length: int) -> None:
        super().__init__()
        self.variant = variant
        self.embedding = nn.Embedding(vocabulary, width)
        self.position = nn.Parameter(torch.zeros(1, length, width))
        heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
        if variant == "attention":
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        elif variant == "decoder":
            layer = nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=4 * width, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=2)
        elif variant != "linear":
            raise ValueError(f"learned Transformer variant must be linear, attention, or decoder: {variant}")
        self.readout = nn.Linear(width, vocabulary)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        if self.variant == "linear":
            hidden = self.embedding(tokens)
        else:
            mask = torch.triu(
                torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool),
                diagonal=1,
            )
            if self.variant == "attention":
                hidden = hidden + self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)[0]
            else:
                hidden = self.decoder(hidden, mask=mask)
        return self.readout(hidden)


def _sequences(
    rng: np.random.Generator,
    count: int,
    length: int,
    vocabulary: int,
    corruption: float,
) -> tuple[np.ndarray, np.ndarray]:
    modes = rng.choice((-1, 1), size=count)
    tokens = np.empty((count, length + 1), dtype=np.int64)
    tokens[:, 0] = rng.integers(0, vocabulary, size=count)
    for position in range(1, length + 1):
        proposed = (tokens[:, position - 1] + modes) % vocabulary
        corrupt = rng.random(count) < corruption
        proposed[corrupt] = rng.integers(0, vocabulary, size=int(corrupt.sum()))
        tokens[:, position] = proposed
    return tokens[:, :-1], tokens[:, 1:]


def _evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(inputs)
    probabilities = logits.softmax(dim=-1)
    correct = logits.argmax(dim=-1).eq(targets)
    losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    return correct, losses, probabilities


def run_learned_transformer(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    rng = task_rng(task, "learned_decoder")
    vocabulary = max(8, int(task.parameters.get("vocabulary", 16)))
    length = max(4, int(task.parameters.get("sequence_length", 12)))
    width = max(8, min(int(task.size), int(task.parameters.get("width_cap", 128))))
    corruption = float(np.clip(task.control, 0.0, 0.95))
    train_count = max(16, int(task.parameters.get("train_examples", 256)))
    n_probe = max(16, int(task.parameters.get("n_probe", 256)))
    train_x, train_y = _sequences(rng, train_count, length, vocabulary, corruption)
    train_inputs = torch.as_tensor(train_x, device=device)
    train_targets = torch.as_tensor(train_y, device=device)

    model = _CausalTokenModel(task.variant, vocabulary, width, length).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task.parameters.get("learning_rate", 3e-3)),
        weight_decay=float(task.parameters.get("weight_decay", 1e-4)),
    )
    steps = max(1, int(task.parameters.get("steps", 64)))
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), train_targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    test_x, test_y = _sequences(rng, n_probe, length, vocabulary, corruption)
    ood_corruption = min(0.95, corruption + float(task.parameters.get("ood_shift", 0.15)))
    ood_x, ood_y = _sequences(rng, n_probe, length, vocabulary, ood_corruption)
    test_inputs = torch.as_tensor(test_x, device=device)
    test_targets = torch.as_tensor(test_y, device=device)
    ood_inputs = torch.as_tensor(ood_x, device=device)
    ood_targets = torch.as_tensor(ood_y, device=device)
    ablated_inputs = test_inputs.clone()
    ablated_inputs[:, :-1] = 0

    model.eval()
    with torch.no_grad():
        correct, token_losses, probabilities = _evaluate(model, test_inputs, test_targets)
        ood_correct, _, _ = _evaluate(model, ood_inputs, ood_targets)
        ablated_correct, _, _ = _evaluate(model, ablated_inputs, test_targets)
    accuracy = float(correct.float().mean())
    ood_accuracy = float(ood_correct.float().mean())
    ablated_accuracy = float(ablated_correct.float().mean())
    signed = 2.0 * correct.detach().cpu().numpy().astype(np.float64).reshape(-1) - 1.0
    marginal = probabilities.mean(dim=(0, 1)).detach().cpu().numpy()
    multiplicity = float(np.exp(entropy(marginal)))
    context_response = accuracy - ablated_accuracy
    oracle_accuracy = (1.0 - corruption) + corruption / vocabulary
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - accuracy,
        ood_generalization_error=1.0 - ood_accuracy,
        effective_multiplicity=multiplicity,
        interaction_range=float(max(0.0, context_response)),
        oracle_gap=float(max(0.0, oracle_accuracy - accuracy)),
        intervention_response=float(context_response),
        extras={
            "token_accuracy": accuracy,
            "ood_token_accuracy": ood_accuracy,
            "context_ablation_delta": context_response,
            "training_loss": losses[-1],
            "perplexity": float(np.exp(min(float(token_losses.mean()), 20.0))),
            "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
            "model_width": float(width),
            "oracle_accuracy": float(oracle_accuracy),
        },
    )
    arrays.update(
        loss_curve=np.asarray(losses, dtype=np.float32),
        token_confidence=probabilities.max(dim=-1).values.detach().cpu().numpy().astype(np.float32),
        token_correct=correct.detach().cpu().numpy().astype(np.int8),
    )
    return metrics, arrays


__all__ = ["run_learned_transformer"]
