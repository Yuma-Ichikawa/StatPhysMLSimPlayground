"""Transformer programs K--N plus lifecycle and automatic phase discovery."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, effective_count, entropy, ridge, softmax, task_rng


def _finish(result, **arrays):
    metrics, payload = result
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    return metrics, payload


def _moe(task: TaskSpec):
    rng = task_rng(task, "moe")
    experts = max(2, int(task.parameters.get("experts", 8)))
    latent_dim = max(4, min(int(task.parameters.get("latent_dim", 32)), int(task.size)))
    tokens = max(128, int(task.parameters.get("tokens", min(4096, task.size * 8))))
    topk = max(1, min(int(task.variant.replace("top", "")) if task.variant.startswith("top") and task.variant[3:].isdigit() else int(task.parameters.get("topk", 2)), experts))
    centroids = rng.normal(size=(experts, latent_dim))
    labels = rng.integers(0, experts, size=tokens)
    inputs = centroids[labels] + max(float(task.control), 1e-3) * rng.normal(size=(tokens, latent_dim))
    logits = inputs @ centroids.T / math.sqrt(latent_dim)
    selected = np.argpartition(logits, -topk, axis=1)[:, -topk:]
    weights = np.zeros_like(logits)
    selected_weights = softmax(np.take_along_axis(logits, selected, axis=1), axis=1)
    np.put_along_axis(weights, selected, selected_weights, axis=1)
    prediction = np.argmax(weights, axis=1)
    success = prediction == labels
    load = weights.mean(axis=0)
    load /= load.sum()
    signed = 2.0 * success.astype(float) - 1.0
    balanced = np.full(experts, 1.0 / experts)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - float(np.mean(success)),
        ood_generalization_error=min(1.0, 1.0 - float(np.mean(success)) + 0.1 * float(task.control)),
        effective_multiplicity=float(np.exp(entropy(load))),
        interaction_range=float(topk / experts),
        oracle_gap=1.0 - float(np.mean(success)),
        intervention_response=float(np.sum(np.abs(load - balanced)) / 2.0),
        extras={
            "routing_specialization": float(np.mean(success)),
            "load_balance_entropy": float(entropy(load)),
            "expert_utilization": float(np.count_nonzero(load > 1e-4) / experts),
            "router_margin": float(np.mean(np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2])),
        },
    )
    return _finish(result, router_probabilities=weights, expert_load=load)


def _retrieval(task: TaskSpec):
    rng = task_rng(task, "retrieval")
    memory = max(16, int(task.size))
    d = max(4, int(task.parameters.get("dimension", 24)))
    probes = max(64, int(task.parameters.get("n_probe", 256)))
    keys = rng.normal(size=(memory, d))
    targets = np.sin(keys[:, 0]) + keys[:, 1] * keys[:, 2]
    query = rng.normal(size=(probes, d))
    truth = np.sin(query[:, 0]) + query[:, 1] * query[:, 2]
    parametric_weights = ridge(keys, targets, float(task.parameters.get("ridge", 0.1)))
    parametric = query @ parametric_weights
    distances = np.sum((query[:, None, :] - keys[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    retrieved = targets[nearest]
    gate = np.clip(float(task.control), 0.0, 1.0)
    if task.variant in {"parametric", "closed_book"}:
        prediction = parametric
    elif task.variant in {"retrieval", "open_book"}:
        prediction = retrieved
    else:
        prediction = (1.0 - gate) * parametric + gate * retrieved
    error = (prediction - truth) ** 2
    parametric_error = (parametric - truth) ** 2
    retrieved_error = (retrieved - truth) ** 2
    scale = np.var(truth) + 1e-12
    signed = np.clip(1.0 - 2.0 * error / scale, -1.0, 1.0)
    retrieval_load = np.bincount(nearest, minlength=memory) / probes
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(error) / scale),
        ood_generalization_error=float(np.mean((prediction - (truth + 0.2 * query[:, 0])) ** 2) / scale),
        effective_multiplicity=float(np.exp(entropy(retrieval_load))),
        interaction_range=float(np.mean(np.sqrt(np.min(distances, axis=1))) / math.sqrt(d)),
        oracle_gap=float(np.mean(error) / scale),
        intervention_response=float((np.mean(parametric_error) - np.mean(error)) / scale),
        extras={
            "parametric_error": float(np.mean(parametric_error) / scale),
            "retrieval_error": float(np.mean(retrieved_error) / scale),
            "memory_utilization": float(np.count_nonzero(retrieval_load) / memory),
            "retrieval_gate": gate,
        },
    )
    return _finish(result, retrieval_distance=np.sqrt(np.min(distances, axis=1)), memory_load=retrieval_load)


def _multimodal(task: TaskSpec):
    rng = task_rng(task, "multimodal")
    samples = max(32, int(task.size))
    latent_rank = max(2, int(task.parameters.get("latent_rank", 4)))
    width = max(latent_rank, int(task.parameters.get("width", 32)))
    correlation = np.clip(float(task.control), 0.0, 1.0)
    latent = rng.normal(size=(samples, latent_rank))
    left_map = rng.normal(size=(latent_rank, width)) / math.sqrt(latent_rank)
    right_map = rng.normal(size=(latent_rank, width)) / math.sqrt(latent_rank)
    left = latent @ left_map + (1.0 - correlation + 1e-3) * rng.normal(size=(samples, width))
    right = latent @ right_map + (1.0 - correlation + 1e-3) * rng.normal(size=(samples, width))
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    cross = left.T @ right / samples
    singular = np.linalg.svd(cross, compute_uv=False)
    normalized = singular / max(singular.sum(), 1e-12)
    paired = np.sum(left * right, axis=1)
    shuffled = np.sum(left * right[rng.permutation(samples)], axis=1)
    margin = paired - shuffled
    signed = np.tanh(margin / (np.std(margin) + 1e-12))
    collapse = float(singular[0] ** 2 / max(np.sum(singular**2), 1e-12))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(margin < 0)),
        ood_generalization_error=float(np.mean(margin < np.median(margin))),
        effective_multiplicity=float(np.exp(entropy(normalized))),
        interaction_range=float(np.mean(np.abs(cross))),
        oracle_gap=float(1.0 - np.mean(np.abs(signed))),
        intervention_response=float(np.mean(margin) / max(np.std(margin), 1e-12)),
        extras={
            "cross_modal_alignment": float(np.mean(signed)),
            "modality_collapse": collapse,
            "shared_representation_rank": float(np.sum(singular > singular[0] * 1e-3)),
            "canonical_correlation_sum": float(singular.sum()),
        },
    )
    return _finish(result, cross_modal_spectrum=singular, pairing_margin=margin)


def _compression(task: TaskSpec):
    rng = task_rng(task, "compression")
    d = max(16, int(task.size))
    weights = rng.normal(size=d) / math.sqrt(d)
    control = np.clip(float(task.control), 0.0, 0.99)
    if task.variant in {"quantization", "int8", "int4"}:
        bits = int(task.parameters.get("bits", 4 if task.variant == "int4" else 8))
        levels = 2 ** (bits - 1) - 1
        scale = max(np.max(np.abs(weights)) / levels, 1e-12)
        compressed = np.round(weights / scale) * scale
    elif task.variant in {"pruning", "sparse"}:
        threshold = np.quantile(np.abs(weights), control)
        compressed = weights * (np.abs(weights) >= threshold)
    else:
        rank_fraction = max(1.0 - control, 1.0 / d)
        mask = rng.random(d) < rank_fraction
        compressed = weights * mask
    probes = max(128, int(task.parameters.get("n_probe", 512)))
    x = rng.normal(size=(probes, d))
    target = x @ weights
    prediction = x @ compressed
    error = (prediction - target) ** 2
    scale = np.var(target) + 1e-12
    signed = np.clip(1.0 - 2.0 * error / scale, -1.0, 1.0)
    nonzero = np.abs(compressed) > 0
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(error) / scale),
        ood_generalization_error=float(np.mean((1.2 * prediction - target) ** 2) / scale),
        effective_multiplicity=float(np.count_nonzero(nonzero)),
        interaction_range=float(np.count_nonzero(nonzero) / d),
        oracle_gap=float(np.mean(error) / scale),
        intervention_response=float(np.linalg.norm(compressed - weights) / max(np.linalg.norm(weights), 1e-12)),
        extras={
            "compression_ratio": float(1.0 - np.count_nonzero(nonzero) / d),
            "weight_distortion": float(np.mean((compressed - weights) ** 2)),
            "functional_fidelity": float(1.0 - np.mean(error) / scale),
            "compressed_norm_ratio": float(np.linalg.norm(compressed) / max(np.linalg.norm(weights), 1e-12)),
        },
    )
    return _finish(result, original_weights=weights, compressed_weights=compressed, prediction_error=error)


def _lifecycle(task: TaskSpec):
    rng = task_rng(task, "lifecycle")
    width = max(16, min(int(task.size), 512))
    stages = ("pretrain", "sft", "preference", "compression")
    teacher = rng.normal(size=width)
    state = rng.normal(size=width)
    overlaps, drifts, entropies = [], [], []
    previous = state.copy()
    pressure = max(float(task.control), 0.0)
    for index, stage in enumerate(stages):
        step = (0.45 / (index + 1)) * (teacher - state)
        if stage == "preference":
            step += pressure * 0.05 * np.sign(teacher)
        if stage == "compression":
            threshold = np.quantile(np.abs(state), min(0.9, pressure / (1.0 + pressure)))
            state[np.abs(state) < threshold] = 0.0
        state += step + 0.01 * rng.normal(size=width)
        overlaps.append(float(np.dot(state, teacher) / max(np.linalg.norm(state) * np.linalg.norm(teacher), 1e-12)))
        drifts.append(float(np.linalg.norm(state - previous) / math.sqrt(width)))
        mass = np.abs(state) / max(np.abs(state).sum(), 1e-12)
        entropies.append(float(entropy(mass)))
        previous = state.copy()
    signed = np.clip(2.0 * np.abs(state / (np.max(np.abs(state)) + 1e-12)) - 1.0, -1.0, 1.0)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - overlaps[-1] ** 2,
        ood_generalization_error=min(2.0, 1.0 - overlaps[-1] ** 2 + drifts[-1]),
        effective_multiplicity=float(np.exp(entropies[-1])),
        interaction_range=float(np.mean(drifts)),
        oracle_gap=1.0 - overlaps[-1],
        intervention_response=float(max(overlaps) - overlaps[-1]),
        extras={
            "final_overlap": overlaps[-1],
            "preference_drift": drifts[2],
            "compression_drift": drifts[3],
            "lifecycle_entropy_change": entropies[-1] - entropies[0],
        },
    )
    return _finish(result, stage_overlap=np.asarray(overlaps), stage_drift=np.asarray(drifts), stage_entropy=np.asarray(entropies))


def _js(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    middle = 0.5 * (left + right)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))
    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def _discovery(task: TaskSpec):
    rng = task_rng(task, "discovery")
    probes = max(128, int(task.parameters.get("n_probe", 512)))
    delta = float(task.parameters.get("delta", 0.05))
    controls = np.asarray([task.control - delta, task.control, task.control + delta])
    samples = np.stack([
        np.tanh(control * math.sqrt(max(task.size, 1)) + rng.normal(size=probes))
        for control in controls
    ])
    edges = np.linspace(-1, 1, 33)
    histograms = np.stack([np.histogram(row, bins=edges)[0] + 1e-9 for row in samples])
    js_left = _js(histograms[0], histograms[1])
    js_right = _js(histograms[1], histograms[2])
    score = np.gradient(np.log(histograms / histograms.sum(axis=1, keepdims=True)), controls, axis=0)
    fisher = float(np.sum((histograms[1] / histograms[1].sum()) * score[1] ** 2))
    change = float(abs(js_right - js_left) / max(delta, 1e-12))
    result = common_coordinates(
        samples[1],
        size=task.size,
        generalization_error=float(1.0 - np.mean(np.abs(samples[1]))),
        ood_generalization_error=float(1.0 - np.mean(np.abs(samples[2]))),
        effective_multiplicity=float(np.exp(entropy(histograms[1]))),
        interaction_range=float(0.5 * (js_left + js_right)),
        oracle_gap=float(abs(js_right - js_left)),
        intervention_response=change,
        extras={
            "adjacent_js_left": js_left,
            "adjacent_js_right": js_right,
            "fisher_sensitivity": fisher,
            "change_point_score": change,
        },
    )
    return _finish(result, local_controls=controls, local_histograms=histograms, fisher_score=score)


def run_transformer_system(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    runners = {
        "moe": _moe,
        "retrieval": _retrieval,
        "multimodal": _multimodal,
        "compression": _compression,
        "lifecycle": _lifecycle,
        "discovery": _discovery,
    }
    try:
        return runners[task.family](task)
    except KeyError as error:
        raise ValueError(f"unsupported Transformer systems family {task.family!r}") from error
