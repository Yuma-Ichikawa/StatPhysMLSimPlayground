"""One-task train/evaluate runner for every phase-tensor Transformer family."""

from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Any

import numpy as np
import torch

from statphys.continuation.core.schema import TaskSpec

from .data import VOCABULARY, TokenDataset, build_token_dataset, token_data_summary
from .model import PhaseTensorTransformer, TransformerConfig
from .observables import (
    block_gradient_statistics,
    causal_contributions,
    gradient_noise_scale,
    intensive_losses,
    local_mlp_jacobian_participation,
    mlp_mechanism_statistics,
    normalized_activation_entropy,
    normalized_attention_entropy,
    normalized_participation_ratio,
    residual_stream_statistics,
    relative_update_statistics,
)
from .optimizers import build_optimizer, set_learning_rate


def _family_configuration(task: TaskSpec) -> dict[str, Any]:
    parameters = dict(task.parameters)
    family = task.family
    if family == "tensor_mlp":
        parameters["activation"] = task.variant
    elif family == "tensor_optimizer":
        parameters["optimizer"] = task.variant
    elif family == "tensor_objective":
        parameters["objective_lambda"] = float(task.variant.replace("lambda_", ""))
    elif family == "tensor_realdata":
        parameters["data_kind"] = task.variant
    elif family == "tensor_residual":
        parameters["normalization"] = task.variant
    elif family == "tensor_scaling":
        parameters["scaling_path"] = task.variant
    else:
        raise ValueError(f"unsupported phase-tensor family: {family}")
    return parameters


def _model_dimensions(task: TaskSpec, parameters: dict[str, Any]) -> tuple[int, int, int]:
    path = str(parameters.get("scaling_path", "width"))
    if path == "depth":
        return int(parameters.get("width", parameters.get("d_model", 64))), int(task.size), int(parameters.get("sequence_length", 64))
    if path == "context":
        return int(parameters.get("width", parameters.get("d_model", 64))), int(parameters.get("depth", parameters.get("layers", 2))), int(task.size)
    return int(task.size), int(parameters.get("depth", parameters.get("layers", 2))), int(parameters.get("sequence_length", 64))


def _heads(width: int, requested: int) -> int:
    candidates = [value for value in range(1, requested + 1) if width % value == 0]
    return max(candidates, default=1)


def _training_example_budget(
    control: float, width: int, sample_exponent: float, train_cap: int
) -> tuple[int, int, bool]:
    requested = max(32, int(round(control * width**sample_exponent)))
    return requested, min(requested, train_cap), requested > train_cap


def _batch(dataset: TokenDataset, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    cpu_indices = indices.cpu()
    return (
        dataset.inputs[cpu_indices].to(device, non_blocking=True),
        dataset.targets[cpu_indices].to(device, non_blocking=True),
        dataset.mask[cpu_indices].to(device, non_blocking=True),
    )


def _evaluate(
    model: PhaseTensorTransformer,
    dataset: TokenDataset,
    objective_lambda: float,
    device: torch.device,
    *,
    ablate_attention: bool = False,
    ablate_mlp: bool = False,
    max_examples: int | None = None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    model.eval()
    with torch.no_grad():
        limit = dataset.size if max_examples is None else min(dataset.size, max_examples)
        inputs = dataset.inputs[:limit].to(device)
        targets = dataset.targets[:limit].to(device)
        mask = dataset.mask[:limit].to(device)
        output = model(
            inputs,
            ablate_attention=ablate_attention,
            ablate_mlp=ablate_mlp,
            return_diagnostics=not ablate_attention and not ablate_mlp,
        )
        if isinstance(output, tuple):
            logits, diagnostics = output
        else:
            logits, diagnostics = output, None
        losses = intensive_losses(logits, targets, mask, objective_lambda)
    return {name: float(value) for name, value in losses.items()}, diagnostics


def run_phase_tensor(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    parameters = _family_configuration(task)
    nested = task.nested_seeds
    torch.manual_seed(nested["initialization"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(nested["initialization"])
    width, layers, sequence_length = _model_dimensions(task, parameters)
    heads = _heads(width, int(parameters.get("heads", 4)))
    activation = str(parameters.get("activation", "gelu"))
    normalization = str(parameters.get("normalization", "pre_rmsnorm"))
    ff_ratio = float(parameters.get("ff_ratio", 4.0))
    sample_exponent = float(parameters.get("sample_exponent", 1.0))
    train_cap = int(parameters.get("train_cap", parameters.get("max_train_examples", 8192)))
    requested_train_examples, train_examples, train_cap_hit = _training_example_budget(
        float(task.control), width, sample_exponent, train_cap
    )
    heldout_examples = int(parameters.get("eval_examples", parameters.get("heldout_examples", 256)))
    noise = float(parameters.get("noise", parameters.get("data_noise", 0.05)))
    data_kind = str(parameters.get("data_kind", "synthetic_retrieval"))
    data_root = os.environ.get("STATPHYS_DATA_ROOT")
    disjoint_corpus_splits = bool(parameters.get("disjoint_corpus_splits", False))
    train_data = build_token_dataset(
        data_kind,
        count=train_examples,
        length=sequence_length,
        seed=nested["data"],
        noise=noise,
        data_root=data_root,
        corpus_split="train" if disjoint_corpus_splits else None,
    )
    test_data = build_token_dataset(
        data_kind,
        count=heldout_examples,
        length=sequence_length,
        seed=nested["evaluation"],
        noise=noise,
        data_root=data_root,
        corpus_split="test" if disjoint_corpus_splits else None,
    )
    ood_kind = str(parameters.get("ood_data_kind", data_kind))
    ood_data = build_token_dataset(
        ood_kind,
        count=heldout_examples,
        length=sequence_length,
        seed=nested["evaluation"] + 1,
        noise=min(0.45, noise + float(parameters.get("ood_shift", parameters.get("ood_noise_shift", 0.15)))),
        data_root=data_root,
        corpus_split="ood" if disjoint_corpus_splits else None,
    )
    model = PhaseTensorTransformer(
        TransformerConfig(
            vocabulary=VOCABULARY,
            width=width,
            sequence_length=sequence_length,
            heads=heads,
            layers=layers,
            ff_ratio=ff_ratio,
            activation=activation,
            normalization=normalization,
            residual_scale=float(parameters.get("residual_scale", 1.0)),
            tie_embeddings=bool(parameters.get("tie_embeddings", True)),
        )
    ).to(device)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer_name = str(parameters.get("optimizer", "adamw"))
    base_lr = float(parameters.get("learning_rate", 3e-4))
    optimizer = build_optimizer(
        optimizer_name,
        model,
        learning_rate=base_lr,
        weight_decay=float(parameters.get("weight_decay", 0.01)),
        momentum=float(parameters.get("momentum", 0.95)),
        rank=int(parameters.get("optimizer_rank", 32)),
    )
    objective_lambda = float(parameters.get("objective_lambda", 1.0))
    steps = int(parameters.get("steps", 300))
    batch_size = min(int(parameters.get("batch_size", 64)), train_data.size)
    warmup = max(1, int(float(parameters.get("warmup_fraction", 0.05)) * steps))
    generator = torch.Generator(device="cpu").manual_seed(nested["minibatch"])
    use_amp = bool(parameters.get("bfloat16", True)) and device.type == "cuda"
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_amp
        else contextlib.nullcontext
    )
    history_step: list[int] = []
    history_train_loss: list[float] = []
    history_test_loss: list[float] = []
    history_generalization_gap: list[float] = []
    history_order: list[float] = []
    tokens_seen = 0
    time_to_order = float(steps)
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        if step < warmup:
            learning_rate = base_lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(steps - warmup - 1, 1)
            learning_rate = 0.1 * base_lr + 0.45 * base_lr * (1.0 + math.cos(math.pi * progress))
        set_learning_rate(optimizer, learning_rate)
        indices = torch.randint(train_data.size, (batch_size,), generator=generator)
        inputs, targets, mask = _batch(train_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(inputs)
            losses = intensive_losses(logits, targets, mask, objective_lambda)
            loss = losses["objective"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(parameters.get("gradient_clip", 1.0)))
        optimizer.step()
        tokens_seen += int(mask.sum().item())
        interval = int(parameters.get("record_interval", max(1, steps // 10)))
        if step == 0 or (step + 1) % interval == 0 or step + 1 == steps:
            train_measured, _ = _evaluate(
                model, train_data, objective_lambda, device, max_examples=heldout_examples
            )
            measured, _ = _evaluate(model, test_data, objective_lambda, device)
            order = 1.0 - measured["objective"]
            history_step.append(step + 1)
            history_train_loss.append(train_measured["objective"])
            history_test_loss.append(measured["objective"])
            history_generalization_gap.append(measured["objective"] - train_measured["objective"])
            history_order.append(order)
            if order >= 0.5 and time_to_order == float(steps):
                time_to_order = float(step + 1)
            model.train()
    elapsed = time.perf_counter() - started

    indices = torch.arange(min(batch_size, train_data.size))
    inputs, targets, mask = _batch(train_data, indices, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    train_probe = intensive_losses(logits, targets, mask, objective_lambda)
    train_probe["objective"].backward()
    gradient_metrics = block_gradient_statistics(model)
    first_gradient = torch.cat(
        [parameter.grad.detach().float().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    )
    second_indices = (torch.arange(batch_size) + batch_size) % train_data.size
    inputs, targets, mask = _batch(train_data, second_indices, device)
    optimizer.zero_grad(set_to_none=True)
    second_logits = model(inputs)
    intensive_losses(second_logits, targets, mask, objective_lambda)["objective"].backward()
    second_gradient = torch.cat(
        [parameter.grad.detach().float().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    )
    gradient_noise = gradient_noise_scale(first_gradient, second_gradient)
    update_metrics = relative_update_statistics(model, initial)
    model.zero_grad(set_to_none=True)

    train_full, _ = _evaluate(
        model, train_data, objective_lambda, device, max_examples=heldout_examples
    )
    full, diagnostics = _evaluate(model, test_data, objective_lambda, device)
    ood, _ = _evaluate(model, ood_data, objective_lambda, device)
    no_attention, _ = _evaluate(
        model, test_data, objective_lambda, device, ablate_attention=True
    )
    no_mlp, _ = _evaluate(model, test_data, objective_lambda, device, ablate_mlp=True)
    neither, _ = _evaluate(
        model,
        test_data,
        objective_lambda,
        device,
        ablate_attention=True,
        ablate_mlp=True,
    )
    contributions = causal_contributions(
        full["objective"], no_attention["objective"], no_mlp["objective"], neither["objective"]
    )
    assert diagnostics is not None
    activation_tensor = diagnostics["mlp_activation"]
    gate_tensor = diagnostics["mlp_gate"]
    attention_tensor = diagnostics["attention"]
    mechanism_metrics = mlp_mechanism_statistics(
        activation_tensor, gate_tensor, gated=activation in {"geglu", "swiglu"}
    )
    jacobian_rank = local_mlp_jacobian_participation(
        model.blocks[0].mlp, diagnostics["mlp_input"][0, 0, 0]
    )
    residual_metrics = residual_stream_statistics(diagnostics["layer_representation"])
    data_metrics = token_data_summary(train_data)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    approximate_flops = 6.0 * parameter_count * tokens_seen
    semantic_order = 1.0 - full["objective"]
    signed_phase_samples = np.clip(2.0 * np.asarray(history_order, dtype=np.float64) - 1.0, -1.0, 1.0)
    phase_second = float(np.mean(signed_phase_samples**2))
    phase_fourth = float(np.mean(signed_phase_samples**4))
    positive_fraction = float(np.mean(signed_phase_samples > 0.0))
    if positive_fraction <= 0.0 or positive_fraction >= 1.0:
        macrostate_entropy = 0.0
    else:
        macrostate_entropy = float(
            -(positive_fraction * math.log(positive_fraction) + (1.0 - positive_fraction) * math.log(1.0 - positive_fraction))
            / math.log(2.0)
        )
    effective_ff_width = max(1.0, ff_ratio * width)
    metrics = {
        "order_parameter": float(abs(np.mean(signed_phase_samples))),
        "susceptibility": float(np.var(signed_phase_samples)),
        "binder_cumulant": float(0.0 if phase_second <= 1e-12 else 1.0 - phase_fourth / (3.0 * phase_second**2)),
        "macrostate_entropy": macrostate_entropy,
        "generalization_error": full["objective"],
        "normalized_generalization_error": full["objective"],
        "normalized_train_risk": train_full["objective"],
        "normalized_test_risk": full["objective"],
        "normalized_ood_risk": ood["objective"],
        "train_ce_nats": train_full["ce_nats"],
        "test_ce_nats": full["ce_nats"],
        "ood_ce_nats": ood["ce_nats"],
        "generalization_gap": float(full["objective"] - train_full["objective"]),
        "normalized_generalization_gap": float(full["objective"] - train_full["objective"]),
        "ood_generalization_gap": float(ood["objective"] - train_full["objective"]),
        "ood_generalization_error": ood["objective"],
        "normalized_ce": full["ce_normalized"],
        "bits_per_byte": full["bits_per_byte"],
        "normalized_brier": full["brier_normalized"],
        "token_accuracy": full["accuracy"],
        "semantic_order": semantic_order,
        "effective_multiplicity": normalized_participation_ratio(activation_tensor),
        "mlp_participation_fraction": normalized_participation_ratio(activation_tensor),
        "mlp_activation_entropy": normalized_activation_entropy(activation_tensor),
        "mlp_jacobian_effective_rank": jacobian_rank,
        "attention_entropy": normalized_attention_entropy(attention_tensor),
        "interaction_range": 1.0 - normalized_attention_entropy(attention_tensor),
        "oracle_gap": full["objective"],
        "intervention_response": contributions["attention_causal_effect"],
        "gradient_rms": gradient_metrics["gradient_rms"],
        "gradient_log_cv": gradient_metrics["gradient_log_cv"],
        "gradient_gini": gradient_metrics["gradient_gini"],
        "gradient_block_gini": gradient_metrics["gradient_gini"],
        "gradient_noise_scale": gradient_noise,
        "update_to_weight_rms": update_metrics["update_to_weight_rms"],
        "update_to_weight_max": update_metrics["update_to_weight_max"],
        "time_to_order": time_to_order,
        "parameter_count": float(parameter_count),
        "model_width": float(width),
        "model_depth": float(layers),
        "context_length": float(sequence_length),
        "ff_ratio": ff_ratio,
        "effective_ff_width": effective_ff_width,
        "sample_coefficient": float(task.control),
        "sample_exponent": sample_exponent,
        "requested_train_examples": float(requested_train_examples),
        "train_examples": float(train_examples),
        "train_cap_hit": float(train_cap_hit),
        "tokens_seen": float(tokens_seen),
        "training_flops_estimate": float(approximate_flops),
        "wall_seconds": float(elapsed),
        "tokens_per_second": float(tokens_seen / max(elapsed, 1e-12)),
        "objective_lambda": objective_lambda,
        "corpus_split_disjoint": float(disjoint_corpus_splits),
    }
    if disjoint_corpus_splits:
        metrics["train_test_byte_overlap_fraction"] = 0.0
    metrics |= contributions | mechanism_metrics | residual_metrics | data_metrics
    arrays: dict[str, Any] = {
        "history_step": np.asarray(history_step, dtype=np.int32),
        "history_train_risk": np.asarray(history_train_loss, dtype=np.float32),
        "history_test_risk": np.asarray(history_test_loss, dtype=np.float32),
        "history_generalization_gap": np.asarray(history_generalization_gap, dtype=np.float32),
        "history_generalization_error": np.asarray(history_test_loss, dtype=np.float32),
        "history_semantic_order": np.asarray(history_order, dtype=np.float32),
        "attention_map_mean": attention_tensor.detach().float().mean(dim=(0, 1, 2)).cpu().numpy(),
        "dataset_sample_hash": np.asarray([int(test_data.metadata["sample_sha256"][:15], 16)]),
        "train_dataset_sample_hash": np.asarray(
            [int(train_data.metadata["sample_sha256"][:15], 16)]
        ),
        "test_dataset_sample_hash": np.asarray(
            [int(test_data.metadata["sample_sha256"][:15], 16)]
        ),
        "ood_dataset_sample_hash": np.asarray(
            [int(ood_data.metadata["sample_sha256"][:15], 16)]
        ),
    }
    return metrics, arrays
