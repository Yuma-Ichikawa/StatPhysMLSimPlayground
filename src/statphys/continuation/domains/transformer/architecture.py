"""M0--M8 x D0--D5 reduced-order state evolution with finite-size fluctuations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...core.schema import TaskSpec
from ..common import common_coordinates, entropy, task_rng


_DATA_NOISE = {
    "d0": 1.0,
    "d1": 1.25,
    "d2": 1.8,
    "d3": 1.1,
    "d4": 1.4,
    "d5": 1.6,
}


def _stage(variant: str) -> int:
    lowered = variant.lower()
    if lowered.startswith("m") and lowered[1:].isdigit():
        value = int(lowered[1:])
        if 0 <= value <= 8:
            return value
    raise ValueError(f"architecture variant must be M0--M8, got {variant!r}")


def run_architecture_ladder(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    rng = task_rng(task, "architecture")
    stage = _stage(task.variant)
    data_stage = str(task.parameters.get("data_stage", "d0")).lower()
    if data_stage not in _DATA_NOISE:
        raise ValueError(f"data_stage must be D0--D5, got {data_stage!r}")
    dimension = int(task.size)
    exponent = max(float(task.control), 0.25)
    sample_coefficient = float(task.parameters.get("sample_coefficient", 1.0))
    log_samples = math.log(max(sample_coefficient, 1e-12)) + exponent * math.log(dimension)
    sample_ratio = math.exp(min(log_samples - math.log(dimension), 50.0))
    teacher_rank = 1 if stage == 0 else int(task.parameters.get("teacher_rank", 4))
    teacher_rank = max(1, teacher_rank)
    capacities = (1, 1, 2, 4, 4, 6, 6, 8, 8)
    capacity = min(capacities[stage], teacher_rank)
    semantic_mixture = np.clip(float(task.parameters.get("semantic_mixture", 0.5)), 0.0, 1.0)
    signal = np.empty(teacher_rank)
    signal[0] = math.sqrt(max(1.0 - semantic_mixture, 0.0))
    if teacher_rank > 1:
        signal[1:] = math.sqrt(semantic_mixture / (teacher_rank - 1))
    noise_scale = _DATA_NOISE[data_stage] / math.sqrt(max(sample_ratio, 1e-12))
    if data_stage == "d2":
        fluctuation = rng.standard_t(df=3.0, size=teacher_rank) / math.sqrt(3.0)
    elif data_stage == "d3":
        fluctuation = rng.choice((-1.0, 1.0), size=teacher_rank)
    elif data_stage == "d4":
        innovations = rng.normal(size=teacher_rank)
        fluctuation = np.empty(teacher_rank)
        fluctuation[0] = innovations[0]
        for index in range(1, teacher_rank):
            fluctuation[index] = 0.8 * fluctuation[index - 1] + 0.6 * innovations[index]
    elif data_stage == "d5":
        fluctuation = rng.normal(size=teacher_rank) + 0.5 * np.sin(np.arange(teacher_rank))
    else:
        fluctuation = rng.normal(size=teacher_rank)
    estimate = signal + noise_scale * fluctuation
    if stage == 1:
        estimate *= 1.0 - 0.05 / math.sqrt(max(dimension, 1))
    if capacity < teacher_rank:
        estimate[capacity:] = 0.0
    if stage >= 4:
        estimate += 0.03 * signal
    if stage >= 6:
        estimate[0] *= 1.0 + 0.05
    if stage >= 7:
        estimate = np.tanh(estimate) + 0.1 * estimate
    direction = str(task.parameters.get("scan_direction", "independent")).lower()
    if direction in {"up", "down"}:
        branch = 1.0 if direction == "up" else -1.0
        metastability = math.exp(-dimension / max(float(task.parameters.get("hysteresis_scale", 256)), 1.0))
        estimate[0] += branch * metastability * 0.1
    denominator = max(np.linalg.norm(signal) * np.linalg.norm(estimate), 1e-12)
    total_overlap = float(np.dot(signal, estimate) / denominator)
    positional_overlap = float(signal[0] * estimate[0] / max(abs(signal[0] * estimate[0]) + noise_scale, 1e-12))
    semantic_overlap = (
        float(np.dot(signal[1:], estimate[1:]) / max(np.linalg.norm(signal[1:]) * np.linalg.norm(estimate[1:]), 1e-12))
        if teacher_rank > 1 and np.linalg.norm(estimate[1:]) > 0
        else 0.0
    )
    probes = max(64, int(task.parameters.get("n_probe", 256)))
    overlap_samples = np.clip(
        total_overlap + noise_scale * rng.normal(size=probes) / math.sqrt(max(teacher_rank, 1)),
        -1.0,
        1.0,
    )
    generalization = float(np.sum((estimate - signal) ** 2) / max(np.sum(signal**2), 1e-12))
    spectrum = np.sort(np.abs(estimate))[::-1]
    normalized_spectrum = spectrum / max(spectrum.sum(), 1e-12)
    result = common_coordinates(
        overlap_samples,
        size=dimension,
        generalization_error=generalization,
        ood_generalization_error=generalization * _DATA_NOISE[data_stage] ** 2,
        effective_multiplicity=float(np.exp(entropy(normalized_spectrum))),
        interaction_range=float((stage + 1) / 9.0),
        oracle_gap=generalization,
        intervention_response=float(abs(positional_overlap - semantic_overlap)),
        extras={
            "teacher_student_overlap": total_overlap,
            "functional_m_pos": positional_overlap,
            "functional_m_sem": semantic_overlap,
            "specialization_strength": abs(positional_overlap - semantic_overlap),
            "sample_exponent": exponent,
            "sample_ratio": sample_ratio,
            "architecture_stage": float(stage),
            "teacher_rank": float(teacher_rank),
            "student_capacity": float(capacity),
            "representation_rank": float(np.count_nonzero(spectrum > noise_scale)),
            "qk_outlier_ratio": float(spectrum[0] / max(np.median(spectrum), 1e-12)),
            "finite_size_noise": noise_scale,
        },
    )
    metrics, arrays = result
    arrays.update(
        order_parameter_samples=overlap_samples.astype(np.float32),
        signal_spectrum=spectrum.astype(np.float32),
        latent_signal=signal.astype(np.float32),
        latent_estimate=estimate.astype(np.float32),
    )
    return metrics, arrays
