"""Transformer programs A--J: heads, algorithms, ICL, scaling, and lifecycle precursors."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, effective_count, entropy, ridge, softmax, task_rng


def _bounded_dimension(task: TaskSpec, cap: int = 256) -> int:
    cap = int(task.parameters.get("numerical_cap", cap))
    return max(4, min(int(task.size), cap))


def _with_arrays(
    result: tuple[dict[str, float], dict[str, np.ndarray]], **arrays: np.ndarray
) -> tuple[dict[str, float], dict[str, Any]]:
    metrics, payload = result
    payload.update({name: np.asarray(value) for name, value in arrays.items()})
    return metrics, payload


def _heads(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "heads")
    d = _bounded_dimension(task, 1024)
    rank = max(1, min(int(task.parameters.get("teacher_rank", 4)), d))
    ratio = max(float(task.control), 0.125)
    n_heads = max(1, int(round(ratio * rank)))
    sample_ratio = float(task.parameters.get("sample_ratio", 2.0))
    basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    noise_scale = 1.0 / math.sqrt(max(sample_ratio * task.size, 1.0))
    heads = np.empty((d, n_heads))
    for head in range(n_heads):
        target = basis[:, head % rank]
        if task.variant in {"redundant", "tied"}:
            target = basis[:, 0]
        elif task.variant in {"untied", "qkvo"}:
            target = basis @ rng.normal(size=rank)
            target /= max(np.linalg.norm(target), np.finfo(float).eps)
        vector = target + noise_scale * rng.normal(size=d)
        heads[:, head] = vector / max(np.linalg.norm(vector), np.finfo(float).eps)
    overlap = np.abs(heads.T @ basis)
    top = overlap.max(axis=1)
    second = np.partition(overlap, -2, axis=1)[:, -2] if rank > 1 else np.zeros(n_heads)
    specialization = top - second
    assignments = np.argmax(overlap, axis=1)
    loads = np.bincount(assignments, minlength=rank).astype(float)
    loads /= max(loads.sum(), 1.0)
    recovered = np.max(overlap, axis=0)
    full_recovery = float(np.mean(recovered**2))
    if n_heads > 1:
        ablated = np.delete(overlap, np.argmax(top), axis=0)
        ablated_recovery = float(np.mean(np.max(ablated, axis=0) ** 2))
    else:
        ablated_recovery = 0.0
    signed = np.clip(2.0 * top - 1.0, -1.0, 1.0)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - full_recovery,
        ood_generalization_error=min(2.0, 1.0 - full_recovery + noise_scale),
        effective_multiplicity=float(np.exp(entropy(loads))),
        interaction_range=float(n_heads / rank),
        oracle_gap=1.0 - full_recovery,
        intervention_response=full_recovery - ablated_recovery,
        extras={
            "head_specialization": float(np.mean(specialization)),
            "head_redundancy": float(np.mean(np.max(heads.T @ heads - np.eye(n_heads), axis=1)))
            if n_heads > 1
            else 0.0,
            "effective_heads": float(np.exp(entropy(np.maximum(top, 0.0) / max(top.sum(), 1e-12)))),
            "teacher_rank": float(rank),
            "n_heads": float(n_heads),
        },
    )
    return _with_arrays(result, head_latent_overlap=overlap, head_load=loads)


def _attention_mlp(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "attention_mlp")
    d = _bounded_dimension(task, 64)
    length = max(4, int(task.parameters.get("sequence_length", min(32, max(4, task.size)))))
    probes = max(32, int(task.parameters.get("n_probe", 128)))
    keys = rng.normal(size=(probes, length, d))
    queries = rng.normal(size=(probes, d))
    values = rng.normal(size=(probes, length))
    logits = np.einsum("nld,nd->nl", keys, queries) / math.sqrt(d)
    target_index = np.argmax(logits, axis=1)
    target = values[np.arange(probes), target_index]
    temperature = max(float(task.control), 1e-3)
    weights = softmax(logits / temperature, axis=1)
    attention_prediction = np.sum(weights * values, axis=1)
    mlp_prediction = values.mean(axis=1)
    if task.variant in {"attention", "attention_only", "retrieval"}:
        prediction = attention_prediction
    elif task.variant in {"mlp", "mlp_only", "parity"}:
        prediction = mlp_prediction
    elif task.variant in {"ablated_attention", "no_attention"}:
        prediction = np.zeros_like(target)
    else:
        gate = 1.0 / (1.0 + temperature)
        prediction = gate * attention_prediction + (1.0 - gate) * mlp_prediction
    error = (prediction - target) ** 2
    baseline_error = (mlp_prediction - target) ** 2
    scale = np.var(target) + 1e-12
    signed = np.clip(1.0 - 2.0 * error / scale, -1.0, 1.0)
    distances = (length - 1 - target_index) / max(length - 1, 1)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(error) / scale),
        ood_generalization_error=float(np.mean((0.8 * prediction - 1.2 * target) ** 2) / scale),
        effective_multiplicity=float(np.mean(effective_count(weights, axis=1))),
        interaction_range=float(np.mean(distances)),
        oracle_gap=float(np.mean(error) / scale),
        intervention_response=float((np.mean(baseline_error) - np.mean(error)) / scale),
        extras={
            "attention_entropy": float(np.mean(entropy(weights, axis=1))),
            "attention_success": float(np.mean(np.argmax(weights, axis=1) == target_index)),
            "mlp_error": float(np.mean(baseline_error) / scale),
            "composition_gap": float(np.mean(np.abs(attention_prediction - mlp_prediction))),
        },
    )
    return _with_arrays(result, attention_weights=weights.astype(np.float32), retrieval_distance=distances)


def _icl(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "icl")
    d = _bounded_dimension(task, 48)
    context = max(2, int(round(max(task.control, 0.125) * d)))
    probes = max(24, int(task.parameters.get("n_probe", 64)))
    regularization = float(task.parameters.get("ridge", 1e-3))
    task_pool_size = max(2, int(task.parameters.get("task_pool", 16)))
    prototypes = rng.normal(size=(task_pool_size, d))
    errors, oracle_errors, alignments = [], [], []
    for _ in range(probes):
        teacher = rng.normal(size=d)
        x = rng.normal(size=(context, d))
        y = x @ teacher + float(task.parameters.get("noise_std", 0.05)) * rng.normal(size=context)
        query = rng.normal(size=d)
        target = float(query @ teacher)
        oracle_weights = ridge(x, y, regularization)
        oracle = float(query @ oracle_weights)
        if task.variant in {"finite_pool", "memorizing"}:
            correlations = prototypes @ oracle_weights
            weights = prototypes[int(np.argmax(correlations))]
            prediction = float(query @ weights)
        elif task.variant in {"mean", "no_context"}:
            prediction = float(np.mean(y))
        else:
            prediction = oracle
        errors.append((prediction - target) ** 2)
        oracle_errors.append((oracle - target) ** 2)
        alignments.append(
            float(np.dot(oracle_weights, teacher) / max(np.linalg.norm(oracle_weights) * np.linalg.norm(teacher), 1e-12))
        )
    errors_array = np.asarray(errors)
    oracle_array = np.asarray(oracle_errors)
    scale = float(np.mean(errors_array) + np.var(errors_array) + 1e-12)
    signed = np.clip(np.asarray(alignments), -1.0, 1.0)
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(errors_array)),
        ood_generalization_error=float(np.mean(errors_array) * (1.0 + 1.0 / math.sqrt(context))),
        effective_multiplicity=float(min(task_pool_size, context)),
        interaction_range=float(context / d),
        oracle_gap=float(max(0.0, np.mean(errors_array) - np.mean(oracle_array))),
        intervention_response=float(np.mean(errors_array) - np.mean(oracle_array)),
        extras={
            "icl_alignment": float(np.mean(alignments)),
            "context_length": float(context),
            "ridge_error": float(np.mean(oracle_array)),
            "critical_context_ratio": float(context / d),
            "error_scale": scale,
        },
    )
    return _with_arrays(result, prediction_error=errors_array, functional_overlap=signed)


def _long_context(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "long_context")
    length = max(8, int(task.size))
    depth_match = re.search(r"(\d+)", task.variant)
    depth = int(depth_match.group(1)) if depth_match else int(task.parameters.get("depth", 4))
    correlation = max(float(task.control), 1e-3)
    distance = np.arange(length, dtype=float)
    kernel = np.exp(-distance / max(correlation * length, 1.0))
    if task.variant.startswith("rope"):
        kernel *= np.cos(distance * float(task.parameters.get("rope_frequency", 0.05)))
        kernel = np.abs(kernel)
    kernel /= kernel.sum()
    propagated = kernel.copy()
    for _ in range(max(depth - 1, 0)):
        propagated = np.convolve(propagated, kernel)[:length]
        propagated /= max(propagated.sum(), 1e-12)
    correlation_length = float(np.sum(distance * propagated))
    participation = float(1.0 / np.sum(propagated**2))
    probes = max(64, int(task.parameters.get("n_probe", 128)))
    relevant = rng.integers(0, length, size=probes)
    recall_probability = np.clip(propagated[relevant] * participation, 0.0, 1.0)
    successes = rng.random(probes) < recall_probability
    signed = 2.0 * successes.astype(float) - 1.0
    error = 1.0 - float(np.mean(successes))
    ablated = float(np.sum(propagated[: max(1, length // 8)]))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=error,
        ood_generalization_error=min(1.0, error + depth / max(length, 1)),
        effective_multiplicity=participation,
        interaction_range=correlation_length / max(length - 1, 1),
        oracle_gap=error,
        intervention_response=float(1.0 - ablated),
        extras={
            "correlation_length": correlation_length,
            "context_participation_ratio": participation / length,
            "depth": float(depth),
            "critical_slowing_proxy": float(depth * correlation_length),
        },
    )
    return _with_arrays(result, context_kernel=propagated, recall_probability=recall_probability)


def _lora(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "lora")
    d = max(8, int(task.size))
    true_rank = max(1, min(int(task.parameters.get("teacher_rank", 16)), d))
    match = re.search(r"(\d+)", task.variant)
    adapter_rank = int(match.group(1)) if match else max(1, int(round(task.control * true_rank)))
    adapter_rank = min(adapter_rank, true_rank)
    spectrum = np.exp(-np.arange(true_rank) / max(float(task.parameters.get("spectral_decay", 4.0)), 1e-3))
    recovered_energy = float(np.sum(spectrum[:adapter_rank] ** 2) / np.sum(spectrum**2))
    probes = max(64, int(task.parameters.get("n_probe", 128)))
    fluctuations = rng.normal(recovered_energy, 1.0 / math.sqrt(max(task.size, 1)), size=probes)
    signed = np.clip(2.0 * fluctuations - 1.0, -1.0, 1.0)
    principal_angles = np.arccos(np.clip(np.sqrt(recovered_energy) + 0.03 * rng.normal(size=adapter_rank), 0, 1))
    error = 1.0 - recovered_energy
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=error,
        ood_generalization_error=min(1.5, error + adapter_rank / max(d, 1) * 0.1),
        effective_multiplicity=float(adapter_rank),
        interaction_range=float(adapter_rank / true_rank),
        oracle_gap=error,
        intervention_response=float(spectrum[min(adapter_rank, true_rank - 1)] ** 2 / np.sum(spectrum**2)),
        extras={
            "adapter_rank": float(adapter_rank),
            "target_rank": float(true_rank),
            "subspace_recovery": recovered_energy,
            "mean_principal_angle": float(np.mean(principal_angles)),
        },
    )
    return _with_arrays(result, singular_spectrum=spectrum, principal_angles=principal_angles)


def _glass(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "glass")
    states = max(16, int(task.size))
    replicas = max(32, int(task.parameters.get("replicas", 64)))
    energy = rng.normal(0.0, math.sqrt(max(math.log(states), 1.0)), size=states)
    beta = max(float(task.control), 1e-3)
    probabilities = softmax(-beta * energy)
    samples = rng.choice(states, size=replicas, p=probabilities)
    basins = np.where(samples < states // 2, -1.0, 1.0)
    overlap = (samples[:, None] == samples[None, :]).astype(float)
    off_diagonal = overlap[~np.eye(replicas, dtype=bool)]
    effective = float(np.exp(entropy(probabilities)))
    escape = max(1e-12, 1.0 - float(np.max(probabilities)))
    result = common_coordinates(
        basins,
        size=task.size,
        generalization_error=float(1.0 - np.max(probabilities)),
        ood_generalization_error=float(1.0 - np.sum(np.sort(probabilities)[-2:])),
        effective_multiplicity=effective,
        interaction_range=float(np.mean(off_diagonal)),
        oracle_gap=float(np.mean(energy[samples]) - np.min(energy)),
        intervention_response=float(np.max(probabilities) - 1.0 / states),
        extras={
            "replica_overlap": float(np.mean(off_diagonal)),
            "participation_ratio": effective,
            "barrier_proxy": float(-math.log(escape)),
            "critical_slowing_proxy": float(1.0 / escape),
            "energy_variance": float(np.var(energy[samples])),
        },
    )
    return _with_arrays(result, energies=energy, state_probabilities=probabilities, replica_overlap=overlap)


def _optimizer(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "optimizer")
    d = _bounded_dimension(task, 512)
    steps = max(20, int(task.parameters.get("steps", 200)))
    learning_rate = float(task.parameters.get("learning_rate", max(task.control, 1e-3)))
    eigenvalues = np.geomspace(1.0 / max(task.size, 1), 1.0, d)
    target = rng.normal(size=d)
    weights = rng.normal(size=d)
    velocity = np.zeros(d)
    second = np.zeros(d)
    losses, overlaps, gradient_norms = [], [], []
    noise = float(task.parameters.get("gradient_noise", 0.02))
    for step in range(steps):
        gradient = eigenvalues * (weights - target) + noise * rng.normal(size=d)
        if task.variant in {"momentum", "heavy_ball"}:
            velocity = 0.9 * velocity + gradient
            update = velocity
        elif task.variant in {"adam", "adamw"}:
            velocity = 0.9 * velocity + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            update = velocity / (np.sqrt(second) + 1e-8)
        else:
            update = gradient
        weights -= learning_rate * update
        if task.variant == "adamw":
            weights *= 1.0 - learning_rate * float(task.parameters.get("weight_decay", 1e-3))
        delta = weights - target
        losses.append(0.5 * float(np.sum(eigenvalues * delta**2)))
        overlaps.append(float(np.dot(weights, target) / max(np.linalg.norm(weights) * np.linalg.norm(target), 1e-12)))
        gradient_norms.append(float(np.linalg.norm(gradient)))
    coordinate_order = 1.0 - 2.0 * np.minimum(np.abs(weights - target) / (np.abs(target) + 1e-6), 1.0)
    final_error = float(np.mean((weights - target) ** 2))
    response = float(np.mean(1.0 / (eigenvalues + 1e-8)))
    fluctuation = float(np.var(np.diff(losses))) if len(losses) > 2 else 0.0
    result = common_coordinates(
        coordinate_order,
        size=task.size,
        generalization_error=final_error,
        ood_generalization_error=final_error * (1.0 + float(np.std(eigenvalues))),
        effective_multiplicity=float((eigenvalues.sum() ** 2) / np.sum(eigenvalues**2)),
        interaction_range=float(np.mean(1.0 / (1.0 + eigenvalues))),
        oracle_gap=final_error,
        intervention_response=float(abs(overlaps[-1] - overlaps[max(0, len(overlaps) // 2)])),
        extras={
            "hessian_min": float(eigenvalues.min()),
            "hessian_max": float(eigenvalues.max()),
            "hessian_condition": float(eigenvalues.max() / eigenvalues.min()),
            "gradient_noise_scale": noise,
            "relaxation_time": float(np.argmax(np.asarray(losses) <= losses[0] / math.e) if losses[-1] <= losses[0] / math.e else steps),
            "effective_temperature": fluctuation / max(response, 1e-12),
        },
    )
    return _with_arrays(
        result,
        loss_trajectory=np.asarray(losses),
        overlap_trajectory=np.asarray(overlaps),
        gradient_norm=np.asarray(gradient_norms),
        hessian_eigenvalues=eigenvalues,
    )


def _draw_data(rng: np.random.Generator, stage: str, n: int, d: int) -> np.ndarray:
    key = stage.lower()
    if key in {"d1", "elliptical"}:
        scales = np.geomspace(0.2, 2.0, d)
        return rng.normal(size=(n, d)) * scales
    if key in {"d2", "heavy_tail", "heavy-tail"}:
        return rng.standard_t(df=3.0, size=(n, d)) / math.sqrt(3.0)
    if key in {"d3", "codebook", "discrete"}:
        codebook = rng.choice((-1.0, 1.0), size=(max(8, d // 2), d))
        return codebook[rng.integers(0, len(codebook), size=n)] + 0.05 * rng.normal(size=(n, d))
    if key in {"d4", "hmm"}:
        states = np.empty(n, dtype=int)
        states[0] = rng.integers(0, 2)
        for index in range(1, n):
            states[index] = states[index - 1] if rng.random() < 0.9 else 1 - states[index - 1]
        return rng.normal(size=(n, d)) + (2 * states[:, None] - 1) * 0.5
    if key in {"d5", "grammar"}:
        latent = rng.integers(0, 4, size=n)
        basis = rng.normal(size=(4, d))
        return basis[latent] + 0.2 * rng.normal(size=(n, d))
    return rng.normal(size=(n, d))


def _data_bridge(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "data_bridge")
    d = _bounded_dimension(task, 160)
    exponent = max(float(task.control), 0.25)
    n = max(d + 2, min(int(round(d**exponent)), int(task.parameters.get("max_samples", 4096))))
    stage = task.variant
    x = _draw_data(rng, stage, n, d)
    teacher = rng.normal(size=d) / math.sqrt(d)
    y = x @ teacher + float(task.parameters.get("noise_std", 0.1)) * rng.normal(size=n)
    estimate = ridge(x, y, float(task.parameters.get("ridge", 1e-3)))
    probe = _draw_data(rng, stage, max(128, d), d)
    ood = _draw_data(rng, "d0", max(128, d), d)
    prediction_error = (probe @ (estimate - teacher)) ** 2
    ood_error = (ood @ (estimate - teacher)) ** 2
    functional = (probe @ estimate) * (probe @ teacher)
    signed = np.tanh(functional / (np.std(functional) + 1e-12))
    covariance = x.T @ x / n
    eigenvalues = np.linalg.eigvalsh(covariance)
    alignment = float(np.dot(estimate, teacher) / max(np.linalg.norm(estimate) * np.linalg.norm(teacher), 1e-12))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(prediction_error)),
        ood_generalization_error=float(np.mean(ood_error)),
        effective_multiplicity=float((eigenvalues.sum() ** 2) / max(np.sum(eigenvalues**2), 1e-12)),
        interaction_range=float(np.mean(np.abs(covariance - np.diag(np.diag(covariance))))),
        oracle_gap=float(np.mean(prediction_error)),
        intervention_response=float(abs(np.mean(ood_error) - np.mean(prediction_error))),
        extras={
            "teacher_student_overlap": alignment,
            "sample_exponent": exponent,
            "sample_count": float(n),
            "covariance_anisotropy": float(eigenvalues.max() / max(eigenvalues.mean(), 1e-12)),
            "representation_rank": float(np.sum(eigenvalues > 1e-6)),
        },
    )
    return _with_arrays(result, covariance_eigenvalues=eigenvalues, functional_overlap=signed)


def _cot(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "cot")
    length = max(4, int(task.size))
    worlds = max(64, int(task.parameters.get("worlds", 256)))
    flip = np.clip(float(task.control), 0.001, 0.499)
    state = rng.choice((-1.0, 1.0), size=worlds)
    posterior = np.zeros(worlds)
    direct = np.zeros(worlds)
    trajectory = np.empty((worlds, length))
    for step in range(length):
        transition = rng.random(worlds) < float(task.parameters.get("transition_probability", 0.1))
        state[transition] *= -1.0
        observation = state.copy()
        observation[rng.random(worlds) < flip] *= -1.0
        direct = observation
        likelihood = 0.5 * np.log((1.0 - flip) / flip) * observation
        posterior = 0.8 * posterior + likelihood
        trajectory[:, step] = np.tanh(posterior)
    prediction = direct if task.variant in {"direct", "no_cot"} else np.sign(posterior)
    signed = prediction * state
    direct_accuracy = float(np.mean(direct == state))
    accuracy = float(np.mean(prediction == state))
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=1.0 - accuracy,
        ood_generalization_error=min(1.0, 1.0 - accuracy + 0.5 * flip),
        effective_multiplicity=float(np.exp(entropy(np.bincount((prediction > 0).astype(int), minlength=2) / worlds))),
        interaction_range=float(np.mean(np.abs(trajectory[:, -1] - trajectory[:, 0]))),
        oracle_gap=float(1.0 - accuracy),
        intervention_response=float(accuracy - direct_accuracy),
        extras={
            "state_tracking_accuracy": accuracy,
            "direct_accuracy": direct_accuracy,
            "scratchpad_gain": accuracy - direct_accuracy,
            "trajectory_commitment": float(np.mean(np.abs(trajectory[:, -1]))),
        },
    )
    return _with_arrays(result, state_posterior_trajectory=trajectory, state_tracking=signed)


def _generation(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "generation")
    vocabulary = max(8, int(task.size))
    samples = max(128, int(task.parameters.get("n_probe", 512)))
    ranks = np.arange(1, vocabulary + 1)
    logits = -float(task.parameters.get("zipf_exponent", 1.1)) * np.log(ranks)
    temperature = max(float(task.control), 1e-3)
    probabilities = softmax(logits / temperature)
    tokens = rng.choice(vocabulary, size=samples, p=probabilities)
    semantic = np.where(tokens < vocabulary // 2, 1.0, -1.0)
    counts = np.bincount(tokens, minlength=vocabulary)
    empirical = counts / samples
    train_support = max(1, int(task.parameters.get("memorized_support", vocabulary // 8)))
    memorization = float(np.mean(tokens < train_support))
    result = common_coordinates(
        semantic,
        size=task.size,
        generalization_error=float(np.sum((empirical - probabilities) ** 2)),
        ood_generalization_error=float(np.sum((empirical - 1.0 / vocabulary) ** 2)),
        effective_multiplicity=float(np.exp(entropy(probabilities))),
        interaction_range=float(np.mean(np.abs(np.diff(tokens))) / max(vocabulary - 1, 1)),
        oracle_gap=float(np.sum((empirical - probabilities) ** 2)),
        intervention_response=float(abs(entropy(probabilities) - entropy(softmax(logits)))),
        extras={
            "generation_entropy": float(entropy(probabilities)),
            "distinct_fraction": float(np.count_nonzero(counts) / vocabulary),
            "memorization_fraction": memorization,
            "generation_temperature": temperature,
            "training_temperature": 0.0,
            "attention_temperature": 0.0,
        },
    )
    return _with_arrays(result, token_histogram=empirical, token_samples=tokens)


def run_transformer_algorithm(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    """Dispatch programs B--J while keeping temperature notions explicitly separated."""
    del device
    runners = {
        "heads": _heads,
        "attention_mlp": _attention_mlp,
        "icl": _icl,
        "long_context": _long_context,
        "lora": _lora,
        "glass": _glass,
        "optimizer": _optimizer,
        "data_bridge": _data_bridge,
        "cot": _cot,
        "generation": _generation,
    }
    try:
        return runners[task.family](task)
    except KeyError as error:
        raise ValueError(f"unsupported Transformer algorithm family {task.family!r}") from error
