"""Diffusion continuation: guidance, trajectory commitment, locality, and memorization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, effective_count, entropy, softmax, task_rng


def _centers(rng: np.random.Generator, components: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.where(np.arange(components) < components // 2, -1.0, 1.0)
    centers = rng.normal(size=(components, dimension))
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    centers[:, 0] += 1.5 * semantic
    return centers, semantic


def _guidance(task: TaskSpec):
    rng = task_rng(task, "guidance")
    d = max(2, min(int(task.size), int(task.parameters.get("dimension_cap", 128))))
    components = max(4, int(task.parameters.get("components", 8)))
    probes = max(64, int(task.parameters.get("n_probe", 256)))
    centers, semantic = _centers(rng, components, d)
    labels = rng.integers(0, components, size=probes)
    noise = float(task.parameters.get("forward_noise", 1.0))
    x = centers[labels] + noise * rng.normal(size=(probes, d))
    distance = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    unconditional_logits = -distance / (2.0 * noise**2)
    semantic_logits = unconditional_logits + semantic[None, :] * float(task.control)
    if task.variant in {"unconditional", "unguided"}:
        posterior = softmax(unconditional_logits, axis=1)
    elif task.variant in {"classifier", "cfg", "guided"}:
        posterior = softmax(semantic_logits, axis=1)
    else:
        posterior = softmax(
            unconditional_logits + float(task.control) * (semantic_logits - unconditional_logits),
            axis=1,
        )
    semantic_order = posterior @ semantic
    assignment = np.argmax(posterior, axis=1)
    correct = assignment == labels
    base = softmax(unconditional_logits, axis=1)
    score = posterior @ centers - x
    base_score = base @ centers - x
    nearest = np.sqrt(np.min(distance, axis=1))
    result = common_coordinates(
        semantic_order,
        size=task.size,
        generalization_error=1.0 - float(np.mean(correct)),
        ood_generalization_error=float(np.mean(nearest > np.median(nearest))),
        effective_multiplicity=float(np.mean(effective_count(posterior, axis=1))),
        interaction_range=float(np.mean(np.linalg.norm(score - base_score, axis=1)) / math.sqrt(d)),
        oracle_gap=float(np.mean(np.sum((posterior - base) ** 2, axis=1))),
        intervention_response=float(np.mean(np.abs(semantic_order - base @ semantic))),
        extras={
            "semantic_speciation": float(np.mean(np.abs(semantic_order))),
            "guidance_susceptibility": float(np.var(semantic_order) * task.size),
            "memorization_distance": float(np.mean(nearest)),
            "posterior_entropy": float(np.mean(entropy(posterior, axis=1))),
            "guidance_scale": float(task.control),
        },
    )
    metrics, arrays = result
    arrays.update(
        posterior=posterior.astype(np.float32),
        semantic_order=semantic_order.astype(np.float32),
        memorization_distance=nearest.astype(np.float32),
    )
    return metrics, arrays


def _trajectory(task: TaskSpec):
    rng = task_rng(task, "trajectory")
    d = max(2, min(int(task.size), int(task.parameters.get("dimension_cap", 64))))
    components = max(4, int(task.parameters.get("components", 8)))
    probes = max(32, int(task.parameters.get("n_probe", 128)))
    steps = max(8, int(task.parameters.get("steps", 32)))
    centers, semantic = _centers(rng, components, d)
    x = rng.normal(size=(probes, d)) * 3.0
    semantic_path = np.empty((probes, steps))
    multiplicity_path = np.empty((probes, steps))
    commitment = np.full(probes, steps, dtype=int)
    control = max(float(task.control), 0.0)
    previous_sign = np.zeros(probes)
    for step in range(steps):
        sigma = 2.0 * (1.0 - step / steps) + 0.15
        distance = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        logits = -distance / (2.0 * sigma**2) + control * semantic[None, :]
        posterior = softmax(logits, axis=1)
        score = (posterior @ centers - x) / sigma**2
        dt = 0.05
        x += dt * score + math.sqrt(2.0 * dt) * sigma * rng.normal(size=x.shape)
        order = posterior @ semantic
        semantic_path[:, step] = order
        multiplicity_path[:, step] = effective_count(posterior, axis=1)
        sign = np.sign(order)
        newly_committed = (commitment == steps) & (np.abs(order) > 0.8) & (sign == previous_sign)
        commitment[newly_committed] = step
        previous_sign = sign
    final_order = semantic_path[:, -1]
    final_distance = np.sqrt(np.min(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1))
    replica_overlap = float(np.mean(np.sign(final_order[:, None]) == np.sign(final_order[None, :])))
    result = common_coordinates(
        final_order,
        size=task.size,
        generalization_error=float(np.mean(final_distance**2) / d),
        ood_generalization_error=float(np.mean(commitment == steps)),
        effective_multiplicity=float(np.mean(multiplicity_path[:, -1])),
        interaction_range=float(np.mean(np.linalg.norm(np.diff(x, axis=0), axis=1)) / math.sqrt(d)),
        oracle_gap=float(np.mean(final_distance)),
        intervention_response=float(np.mean(np.abs(final_order - semantic_path[:, steps // 2]))),
        extras={
            "trajectory_commitment_time": float(np.mean(commitment)),
            "trajectory_uncommitted_fraction": float(np.mean(commitment == steps)),
            "trajectory_replica_overlap": replica_overlap,
            "semantic_speciation": float(np.mean(np.abs(final_order))),
            "critical_slowing_proxy": float(np.mean(commitment) / steps),
        },
    )
    metrics, arrays = result
    arrays.update(
        semantic_trajectory=semantic_path.astype(np.float32),
        multiplicity_trajectory=multiplicity_path.astype(np.float32),
        commitment_time=commitment,
        final_distance=final_distance.astype(np.float32),
    )
    return metrics, arrays


def _locality(task: TaskSpec):
    rng = task_rng(task, "locality")
    d = max(8, min(int(task.size), int(task.parameters.get("dimension_cap", 1024))))
    probes = max(32, int(task.parameters.get("n_probe", 128)))
    frequency = np.fft.rfftfreq(d)
    correlation_length = max(float(task.control) * d, 1.0)
    spectrum = 1.0 / (1.0 + (2.0 * math.pi * correlation_length * frequency) ** 2)
    noise = rng.normal(size=(probes, d))
    field = np.fft.irfft(np.fft.rfft(noise, axis=1) * np.sqrt(spectrum), n=d, axis=1)
    exact_score = np.fft.irfft(
        -np.fft.rfft(field, axis=1) / np.maximum(spectrum, 1e-8), n=d, axis=1
    )
    radius = max(1, int(task.parameters.get("local_radius", max(1, round(float(task.control) * d)))))
    kernel = np.zeros(d)
    kernel[0] = 1.0
    for offset in range(1, min(radius + 1, d // 2)):
        kernel[offset] = kernel[-offset] = math.exp(-offset / max(radius, 1))
    kernel /= kernel.sum()
    local_score = -field + np.fft.irfft(
        np.fft.rfft(field, axis=1) * np.fft.rfft(kernel)[None, :], n=d, axis=1
    )
    error = np.mean((local_score - exact_score) ** 2, axis=1)
    scale = np.mean(exact_score**2) + 1e-12
    signed = np.clip(1.0 - 2.0 * error / scale, -1.0, 1.0)
    response = np.roll(local_score, d // 2, axis=1) - local_score
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(error) / scale),
        ood_generalization_error=float(np.quantile(error, 0.9) / scale),
        effective_multiplicity=float((spectrum.sum() ** 2) / np.sum(spectrum**2)),
        interaction_range=float(radius / d),
        oracle_gap=float(np.mean(error) / scale),
        intervention_response=float(np.mean(np.abs(response)) / (np.mean(np.abs(local_score)) + 1e-12)),
        extras={
            "score_nonlocality": float(np.linalg.norm(exact_score - local_score) / max(np.linalg.norm(exact_score), 1e-12)),
            "correlation_length": correlation_length,
            "local_radius": float(radius),
            "distant_response": float(np.mean(np.abs(response))),
        },
    )
    metrics, arrays = result
    arrays.update(score_error=error.astype(np.float32), field_spectrum=spectrum.astype(np.float32))
    return metrics, arrays


def _memorization(task: TaskSpec):
    rng = task_rng(task, "memorization")
    d = max(2, min(int(task.size), int(task.parameters.get("dimension_cap", 64))))
    train_count = max(16, int(task.parameters.get("train_centers", task.size * 4)))
    probes = max(64, int(task.parameters.get("n_probe", 256)))
    train = rng.normal(size=(train_count, d))
    fresh = rng.normal(size=(probes, d))
    mixing = np.clip(float(task.control), 0.0, 1.0)
    copied = rng.integers(0, train_count, size=probes)
    generated = mixing * train[copied] + (1.0 - mixing) * fresh
    distances = np.sqrt(np.min(np.sum((generated[:, None, :] - train[None, :, :]) ** 2, axis=2), axis=1))
    threshold = float(np.quantile(np.sqrt(np.sum((fresh[: min(probes, train_count)] - train[: min(probes, train_count)]) ** 2, axis=1)), 0.1))
    memorized = distances < threshold
    signed = 2.0 * memorized.astype(float) - 1.0
    assignments = np.argmin(np.sum((generated[:, None, :] - train[None, :, :]) ** 2, axis=2), axis=1)
    load = np.bincount(assignments, minlength=train_count) / probes
    result = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean(distances)),
        ood_generalization_error=float(np.mean(np.linalg.norm(generated - fresh, axis=1))),
        effective_multiplicity=float(np.exp(entropy(load))),
        interaction_range=float(np.mean(distances) / math.sqrt(d)),
        oracle_gap=float(np.mean(distances)),
        intervention_response=float(np.mean(memorized) - np.mean(distances < threshold / 2.0)),
        extras={
            "memorization_fraction": float(np.mean(memorized)),
            "memorization_distance": float(np.mean(distances)),
            "training_support_utilization": float(np.count_nonzero(load) / train_count),
            "novelty_fraction": float(1.0 - np.mean(memorized)),
        },
    )
    metrics, arrays = result
    arrays.update(memorization_distance=distances.astype(np.float32), training_load=load.astype(np.float32))
    return metrics, arrays


def run_diffusion_program(task: TaskSpec, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    runners = {
        "guidance": _guidance,
        "trajectory": _trajectory,
        "locality": _locality,
        "memorization": _memorization,
    }
    try:
        return runners[task.family](task)
    except KeyError as error:
        raise ValueError(f"unsupported diffusion family {task.family!r}") from error
