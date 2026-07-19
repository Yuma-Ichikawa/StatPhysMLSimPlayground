"""Cross-domain diagnostics for assumption interactions and continuation outcomes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..core.schema import TaskSpec
from .common import common_coordinates, effective_count, entropy, ridge, softmax, task_rng

_COORDINATES = ("data", "architecture", "objective", "optimizer", "dynamics", "population")
_DOMAIN_INDEX = {"transformer": 0, "diffusion": 1, "reinforcement": 2, "multiagent": 3}


def _domain_index(variant: str) -> int:
    key = variant.lower()
    if key == "rl":
        key = "reinforcement"
    if key not in _DOMAIN_INDEX:
        raise ValueError(f"diagnostic variant must name a domain, got {variant!r}")
    return _DOMAIN_INDEX[key]


def _assumption_pairs(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "assumption_pairs")
    pair = str(task.parameters.get("pair", ""))
    pieces = pair.split("__")
    if len(pieces) != 2 or any(item not in _COORDINATES for item in pieces):
        raise ValueError(f"invalid assumption pair {pair!r}")
    first, second = (_COORDINATES.index(item) for item in pieces)
    domain = _domain_index(task.variant)
    probes = max(64, int(task.parameters.get("n_probe", 512)))
    strength = float(np.clip(task.control, 0.0, 2.0))
    latent = rng.normal(size=(probes, len(_COORDINATES)))
    domain_weights = np.roll(np.asarray([0.45, -0.30, 0.25, -0.20, 0.35, -0.15]), domain)
    baseline_field = 0.2 * latent[:, domain % len(_COORDINATES)]
    first_effect = domain_weights[first] * latent[:, first]
    second_effect = domain_weights[second] * latent[:, second]
    interaction_coefficient = 0.18 * math.sin((first + 1) * (second + 2) * (domain + 1))
    additive_field = baseline_field + strength * (first_effect + second_effect)
    full_field = additive_field + interaction_coefficient * strength**2 * latent[:, first] * latent[:, second]
    finite_noise = rng.normal(size=probes) / math.sqrt(max(task.size, 1))
    additive_order = np.tanh(additive_field + finite_noise)
    full_order = np.tanh(full_field + finite_noise)
    pair_delta = float(np.mean(np.abs(full_order - additive_order)))
    generalization = float(np.mean((full_order - np.tanh(full_field)) ** 2))
    metrics, arrays = common_coordinates(
        full_order,
        size=task.size,
        generalization_error=generalization,
        ood_generalization_error=float(
            np.mean((np.tanh(1.1 * full_field) - np.tanh(full_field)) ** 2)
        ),
        effective_multiplicity=float(effective_count(np.histogram(full_order, bins=8)[0] + 1e-8)),
        interaction_range=float(2.0 / len(_COORDINATES)),
        oracle_gap=pair_delta,
        intervention_response=pair_delta,
        extras={
            "pair_interaction_delta": pair_delta,
            "additive_prediction_error": float(np.mean((full_order - additive_order) ** 2)),
            "interaction_coefficient": interaction_coefficient,
            "deformation_strength": strength,
            "assumption_a_code": float(first),
            "assumption_b_code": float(second),
        },
    )
    arrays.update(
        additive_order=additive_order.astype(np.float32),
        full_order=full_order.astype(np.float32),
        latent_coordinates=latent.astype(np.float32),
    )
    return metrics, arrays


def _true_response(domain: int, control: np.ndarray) -> np.ndarray:
    critical = (0.43, 0.50, 0.56, 0.47)[domain]
    slope = (5.0, 4.2, 6.0, 3.8)[domain]
    shifted = np.asarray(control) - critical
    return np.tanh(slope * shifted + (domain - 1.5) * shifted**3)


def _renormalized_bridge(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "renormalized_bridge")
    domain = _domain_index(task.variant)
    controls = np.linspace(0.05, 0.95, 11)
    truth = _true_response(domain, controls)
    noise_scale = 0.35 / math.sqrt(max(task.size, 1))
    observed = truth + noise_scale * rng.normal(size=len(controls))
    calibration_mask = np.abs(controls - task.control) > 0.06
    x = controls[calibration_mask]
    y = observed[calibration_mask]
    bare_features = np.stack((np.ones_like(x), x), axis=1)
    renormalized_features = np.stack((np.ones_like(x), x, x**2, 1.0 / np.sqrt(task.size) * np.ones_like(x)), axis=1)
    bare = ridge(bare_features, y, float(task.parameters.get("ridge", 1e-4)))
    renormalized = ridge(renormalized_features, y, float(task.parameters.get("ridge", 1e-4)))
    heldout_bare = float(np.asarray([1.0, task.control]) @ bare)
    heldout_features = np.asarray([1.0, task.control, task.control**2, 1.0 / math.sqrt(task.size)])
    heldout = float(heldout_features @ renormalized)
    heldout_truth = float(_true_response(domain, np.asarray([task.control]))[0])
    bridge_error = abs(heldout - heldout_truth)
    bare_error = abs(heldout_bare - heldout_truth)
    grid = np.linspace(0.0, 1.0, 1001)
    grid_features = np.stack(
        (np.ones_like(grid), grid, grid**2, np.full_like(grid, 1.0 / math.sqrt(task.size))),
        axis=1,
    )
    inferred_boundary = float(grid[np.argmin(np.abs(grid_features @ renormalized))])
    true_boundary = (0.43, 0.50, 0.56, 0.47)[domain]
    probes = max(64, int(task.parameters.get("n_probe", 512)))
    signed = np.tanh(heldout + rng.normal(size=probes) / math.sqrt(max(task.size, 1)))
    ood_control = min(1.0, task.control + 0.1)
    ood_prediction = float(
        np.asarray([1.0, ood_control, ood_control**2, 1.0 / math.sqrt(task.size)])
        @ renormalized
    )
    ood_truth = float(_true_response(domain, np.asarray([ood_control]))[0])
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=bridge_error**2,
        ood_generalization_error=(ood_prediction - ood_truth) ** 2,
        effective_multiplicity=float(np.exp(entropy(np.histogram(signed, bins=8)[0] + 1e-8))),
        interaction_range=float(len(renormalized)),
        oracle_gap=bridge_error,
        intervention_response=float(bare_error - bridge_error),
        extras={
            "bridge_error": bridge_error,
            "bare_bridge_error": bare_error,
            "heldout_prediction": heldout,
            "heldout_truth": heldout_truth,
            "inferred_boundary": inferred_boundary,
            "true_boundary": true_boundary,
            "boundary_error": abs(inferred_boundary - true_boundary),
            "calibration_residual": float(np.mean((renormalized_features @ renormalized - y) ** 2)),
        },
    )
    arrays.update(
        calibration_controls=controls.astype(np.float32),
        calibration_observed=observed.astype(np.float32),
        renormalized_coefficients=renormalized.astype(np.float32),
    )
    return metrics, arrays


def _critical_window(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "critical_window")
    domain = _domain_index(task.variant)
    kind = str(task.parameters.get("transition_kind", "continuous"))
    probes = max(128, int(task.parameters.get("n_probe", 1024)))
    critical = (0.43, 0.50, 0.56, 0.47)[domain]
    reduced = float(task.control) - critical
    scale = max(task.size, 1) ** 0.35
    noise = rng.normal(size=probes)
    if kind == "continuous":
        signed = np.tanh(scale * reduced + noise)
        hysteresis = 0.0
        relaxation = task.size ** 0.35 / (1.0 + abs(scale * reduced))
    elif kind == "first_order":
        weight = 1.0 / (1.0 + math.exp(-np.clip(9.0 * scale * reduced, -60.0, 60.0)))
        branch = np.where(rng.random(probes) < weight, 1.0, -1.0)
        signed = np.clip(0.82 * branch + 0.12 * noise, -1.0, 1.0)
        hysteresis = float(0.18 * math.exp(-abs(reduced) * scale))
        relaxation = task.size ** 0.6 / (1.0 + abs(scale * reduced))
    elif kind == "crossover":
        signed = np.tanh(4.0 * reduced + noise)
        hysteresis = 0.0
        relaxation = 1.0 + 1.0 / (1.0 + 10.0 * reduced**2)
    else:
        raise ValueError(f"transition_kind must be continuous, first_order, or crossover: {kind}")
    intervention = np.tanh(np.arctanh(np.clip(signed, -0.999, 0.999)) + 0.05)
    histogram = np.histogram(signed, bins=24, range=(-1.0, 1.0))[0].astype(float)
    histogram /= max(histogram.sum(), 1.0)
    centered = signed - signed.mean()
    variance = np.mean(centered**2)
    bimodality = float(np.mean(centered**3) ** 2 / max(variance**3, 1e-12) + 1.0 / max(np.mean(centered**4) / max(variance**2, 1e-12), 1e-12))
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(np.mean((signed - np.sign(reduced or 1.0)) ** 2)),
        ood_generalization_error=float(np.mean((signed - intervention) ** 2)),
        effective_multiplicity=float(np.exp(entropy(histogram + 1e-12))),
        interaction_range=float(relaxation / max(task.size, 1)),
        oracle_gap=float(abs(np.mean(signed) - math.tanh(scale * reduced))),
        intervention_response=float(np.mean(intervention - signed)),
        extras={
            "hysteresis_gap": hysteresis,
            "bimodality_coefficient": bimodality,
            "relaxation_time": float(relaxation),
            "critical_distance": reduced,
            "transition_kind_code": float(("continuous", "first_order", "crossover").index(kind)),
            "response_slope": float(np.mean(intervention - signed) / 0.05),
        },
    )
    arrays.update(order_histogram=histogram.astype(np.float32), intervened_order=intervention.astype(np.float32))
    return metrics, arrays


def _outcome_atlas(task: TaskSpec) -> tuple[dict[str, float], dict[str, Any]]:
    rng = task_rng(task, "outcome_atlas")
    probes = max(128, int(task.parameters.get("n_probe", 1024)))
    control = float(task.control)
    scale = max(task.size, 1) ** 0.3
    noise = rng.normal(size=probes)
    outcome = task.variant
    mode_count = 1.0
    statistical_threshold = 0.5
    computational_threshold = 0.5
    if outcome == "stable":
        signed = np.tanh(scale * (control - 0.5) + noise)
    elif outcome == "renormalized":
        signed = np.tanh(scale * (control - 0.62) + 1.15 * noise)
        statistical_threshold = 0.62
        computational_threshold = 0.62
    elif outcome == "splitting":
        amplitude = max(0.0, control - 0.35)
        signed = np.tanh(scale * (control - 0.5) + noise) + rng.choice((-1.0, 1.0), probes) * amplitude
        signed = np.clip(signed, -1.0, 1.0)
        mode_count = 2.0
    elif outcome == "merging":
        amplitude = max(0.0, 0.75 - control)
        signed = np.clip(rng.choice((-1.0, 1.0), probes) * amplitude + 0.15 * noise, -1.0, 1.0)
        mode_count = 2.0 if amplitude > 0.2 else 1.0
    elif outcome == "rounding":
        signed = np.tanh(3.0 * (control - 0.5) + noise)
    elif outcome == "new_phase":
        phases = np.asarray([-0.85, 0.0, 0.85])
        probabilities = softmax(np.asarray([0.4 - control, control - 0.55, control - 0.4]) * 8.0)
        signed = np.clip(phases[rng.choice(3, probes, p=probabilities)] + 0.1 * noise, -1.0, 1.0)
        mode_count = 3.0
    elif outcome == "computational_statistical_separation":
        statistical_threshold = 0.35
        computational_threshold = 0.68
        signed = np.tanh(scale * (control - computational_threshold) + noise)
    else:
        raise ValueError(f"unknown continuation outcome {outcome!r}")
    oracle = np.tanh(scale * (control - statistical_threshold) + noise)
    continuation_distance = float(np.mean(np.abs(signed - oracle)))
    histogram = np.histogram(signed, bins=24, range=(-1.0, 1.0))[0].astype(float)
    histogram /= max(histogram.sum(), 1.0)
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=continuation_distance**2,
        ood_generalization_error=float(np.mean((signed - np.roll(signed, 1)) ** 2)),
        effective_multiplicity=float(np.exp(entropy(histogram + 1e-12))),
        interaction_range=float(mode_count / 3.0),
        oracle_gap=continuation_distance,
        intervention_response=float(abs(computational_threshold - statistical_threshold)),
        extras={
            "continuation_distance": continuation_distance,
            "mode_count": mode_count,
            "statistical_threshold": statistical_threshold,
            "computational_threshold": computational_threshold,
            "threshold_separation": computational_threshold - statistical_threshold,
        },
    )
    arrays.update(order_histogram=histogram.astype(np.float32), oracle_order=oracle.astype(np.float32))
    return metrics, arrays


def run_continuation_diagnostic(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    del device
    runners = {
        "assumption_pairs": _assumption_pairs,
        "renormalized_bridge": _renormalized_bridge,
        "critical_window": _critical_window,
        "outcome_atlas": _outcome_atlas,
    }
    try:
        return runners[task.family](task)
    except KeyError as error:
        raise ValueError(f"unsupported continuation diagnostic {task.family!r}") from error


__all__ = ["run_continuation_diagnostic"]
