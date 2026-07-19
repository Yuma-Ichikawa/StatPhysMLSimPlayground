"""Transformer continuation on the existing M0--M8 attention ladder."""

from __future__ import annotations

from typing import Any

import torch

from statphys.atlas.data import PositionalSemanticDataConfig, PositionalSemanticDataset
from statphys.atlas.models import AttentionLadderConfig, build_attention_ladder

from ...metrics import (
    EPS,
    binary_entropy,
    effective_multiplicity,
    phase_statistics,
    seed_everything,
)
from ...schema import TaskSpec


def _stage_number(variant: str) -> int:
    name = variant.strip().lower().split("_", 1)[0]
    if name not in {f"m{index}" for index in range(9)}:
        raise ValueError("Transformer variants must begin with m0, ..., m8")
    return int(name[1:])


def _loss_pairs(model: Any, predictions: torch.Tensor, targets: torch.Tensor):
    if model.config.is_autoregressive:
        return model.shifted_autoregressive_pairs(predictions, targets)
    return predictions, targets


def _data(
    task: TaskSpec,
    *,
    device: torch.device,
    ensemble: str,
    data_seed: int,
) -> PositionalSemanticDataset:
    stage = _stage_number(task.variant)
    seq_len = 2 if stage <= 1 else int(task.parameters.get("sequence_length", 8))
    rank = 1 if stage == 0 else int(task.parameters.get("teacher_rank", 1 if stage == 1 else 4))
    config = PositionalSemanticDataConfig(
        d_model=task.size,
        seq_len=seq_len,
        signal_rank=min(rank, task.size),
        ensemble=ensemble,
        sigma=float(task.parameters.get("input_noise", 0.5)),
        omega=task.control,
        attention_temperature=float(task.parameters.get("temperature", 1.0)),
        teacher_seed=task.seed,
        data_seed=data_seed,
        init_seed=task.seed + 2,
        student_t_df=float(task.parameters.get("student_t_df", 3.0)),
        device=device,
        dtype=torch.float32,
    )
    return PositionalSemanticDataset(config)


def run_transformer(
    task: TaskSpec,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    stage = _stage_number(task.variant)
    dataset = _data(task, device=device, ensemble="d0", data_seed=task.seed + 1)
    seq_len = dataset.config.seq_len
    rank = dataset.config.signal_rank
    n_heads = 1 if stage <= 2 else min(int(task.parameters.get("n_heads", 4)), task.size)
    while task.size % n_heads:
        n_heads -= 1
    head_dim = task.size if stage <= 2 else task.size // n_heads
    n_layers = int(task.parameters.get("n_layers", 2)) if stage >= 7 else 1
    model = build_attention_ladder(
        AttentionLadderConfig(
            stage=f"m{stage}",
            d_model=task.size,
            seq_len=seq_len,
            signal_rank=rank,
            n_heads=n_heads,
            head_dim=head_dim,
            n_layers=n_layers,
            ffn_dim=int(task.parameters.get("ff_dim_multiplier", 4)) * task.size,
            attention_temperature=dataset.config.attention_temperature,
            init_seed=task.seed + 2,
            device=device,
            dtype=torch.float32,
        )
    )
    strategy = str(task.parameters.get("initialization", "random"))
    model.initialize_from_directions(
        dataset.teacher_vectors,
        dataset.positional_encoding,
        strategy=strategy,
        noise_scale=float(task.parameters.get("initialization_noise", 0.05)),
    )
    model.to(device)

    n_train = max(8, int(float(task.parameters.get("sample_coefficient", 8.0)) * task.size))
    train = dataset.sample(n_train, seed=task.seed + 3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task.parameters.get("learning_rate", 3e-3)),
        weight_decay=float(task.parameters.get("weight_decay", 1e-4)),
    )
    steps = int(task.parameters.get("steps", 1200))
    log_interval = max(1, int(task.parameters.get("log_interval", 25)))
    losses: list[float] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train.inputs)
        prediction, targets = _loss_pairs(model, prediction, train.targets)
        loss = 0.5 * (prediction - targets).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == steps:
            losses.append(float(loss.detach()))

    n_probe = int(task.parameters.get("n_probe", 2048))
    heldout = dataset.sample(n_probe, seed=task.seed + 4)
    ood_dataset = _data(task, device=device, ensemble="d2", data_seed=task.seed + 5)
    ood = ood_dataset.sample(n_probe, seed=task.seed + 6)
    model.eval()
    with torch.no_grad():
        prediction, diagnostics = model(heldout.inputs, return_diagnostics=True)
        ablated = model(heldout.inputs, ablate_attention=True)
        ood_prediction = model(ood.inputs)
        prediction, targets = _loss_pairs(model, prediction, heldout.targets)
        ablated, _ = _loss_pairs(model, ablated, heldout.targets)
        ood_prediction, ood_targets = _loss_pairs(model, ood_prediction, ood.targets)

        errors = 0.5 * (prediction - targets).square().flatten(1).mean(dim=1)
        ood_errors = 0.5 * (ood_prediction - ood_targets).square().flatten(1).mean(dim=1)
        ablated_error = 0.5 * (ablated - targets).square().flatten(1).mean(dim=1)

        attention = diagnostics["attention_maps"][:, -1].mean(dim=1)
        semantic = heldout.semantic_attention
        positional = heldout.positional_attention
        semantic_distance = (attention - semantic).square().flatten(1).mean(dim=1)
        positional_distance = (attention - positional).square().flatten(1).mean(dim=1)
        signed_order = (positional_distance - semantic_distance) / (
            positional_distance + semantic_distance + EPS
        )

        row_multiplicity = effective_multiplicity(attention, dim=-1)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs()
        interaction_range = float(
            (attention * distance).sum(dim=-1).mean() / max(1, seq_len - 1)
        )
        macro_probability = ((signed_order + 1.0) / 2.0).clamp(0.0, 1.0)

        oracle_attention = heldout.mixed_attention
        oracle_sem = (oracle_attention - semantic).square().flatten(1).mean(dim=1)
        oracle_pos = (oracle_attention - positional).square().flatten(1).mean(dim=1)
        oracle_order = (oracle_pos - oracle_sem) / (oracle_pos + oracle_sem + EPS)

    metrics = phase_statistics(signed_order, size=task.size, order_absolute=False)
    metrics.update(
        {
            "generalization_error": float(errors.mean()),
            "ood_generalization_error": float(ood_errors.mean()),
            "effective_multiplicity": float(row_multiplicity.mean()),
            "interaction_range": interaction_range,
            "macrostate_entropy": float(binary_entropy(macro_probability).mean()),
            "oracle_gap": float(errors.mean()),
            "intervention_response": float((ablated_error - errors).mean()),
            "oracle_order_parameter": float(oracle_order.mean()),
            "training_loss": losses[-1],
            "n_train": float(n_train),
        }
    )
    arrays = {
        "signed_order_samples": signed_order.detach().cpu().numpy(),
        "generalization_errors": errors.detach().cpu().numpy(),
        "ood_errors": ood_errors.detach().cpu().numpy(),
        "attention_mean": attention.mean(dim=0).detach().cpu().numpy(),
        "loss_curve": losses,
    }
    return metrics, arrays


__all__ = ["run_transformer"]
