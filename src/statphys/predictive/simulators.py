"""GPU Monte Carlo models for controlled predictive phase continuation.

The four domains share a finite-size stochastic potential but use distinct
controls, observables, symmetry fields, and interventions.  This is a
controlled Tier-B benchmark, not a claim about natural-data endpoints.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import math
from typing import Any

import numpy as np
import torch

from .schema import Task

_DOMAIN_OFFSET = {
    "transformer": -0.08,
    "diffusion": 0.04,
    "reinforcement": 0.12,
    "multiagent": -0.02,
}
_VARIANT_OFFSET = {
    "anchor": 0.0,
    "single": 0.040,
    "augmented": 0.085,
    "holdout": 0.135,
}


def _seed(task: Task, inner: int) -> int:
    # Keep common random numbers across a control sweep, but never couple
    # unrelated domains, variants, or secondary controls accidentally.
    condition = f"{task.domain}|{task.variant}|{task.size}|{task.secondary:.12g}"
    condition_hash = int(sha256(condition.encode()).hexdigest()[:8], 16)
    return int((task.seed * 1_000_003 + inner * 97_409 + condition_hash) % (2**31 - 1))


def _critical_control(task: Task, disorder: float) -> float:
    finite_size = 0.22 / math.sqrt(float(task.size))
    secondary = float(task.secondary)
    interaction = 0.0
    if task.variant in {"augmented", "holdout"}:
        interaction = 0.10 * secondary * secondary
    # The cubic remainder is deliberately absent from the fitted effective
    # model.  It prevents a vacuous machine-precision holdout result.
    holdout_remainder = 0.012 * secondary**3 if task.variant == "holdout" else 0.0
    return (
        1.0
        + _DOMAIN_OFFSET[task.domain]
        + _VARIANT_OFFSET[task.variant]
        + 0.16 * secondary
        + interaction
        + holdout_remainder
        + finite_size
        + disorder
    )


def _langevin(task: Task, generator: torch.Generator, critical: float, device: torch.device) -> torch.Tensor:
    chains = int(task.parameters.get("chains", 512))
    steps = int(task.parameters.get("steps", 160))
    dt = float(task.parameters.get("dt", 0.035))
    m = 0.18 * torch.randn(chains, device=device, generator=generator)
    reduced = float(task.control - critical)
    symmetry_field = 0.0
    if task.domain == "multiagent":
        symmetry_field = float(task.parameters.get("field_sign", 1.0)) * 0.10 * float(task.secondary)
    cubic = 0.16 if task.variant == "holdout" else 0.0
    temperature = 1.0 / math.sqrt(max(float(task.size), 1.0))
    for _ in range(steps):
        grad = -reduced * m + m.pow(3) + cubic * m.pow(5) - symmetry_field
        noise = torch.randn(m.shape, device=device, generator=generator)
        m = (m - dt * grad + math.sqrt(2.0 * dt) * temperature * noise).clamp(-2.5, 2.5)
    return m


def _domain_metrics(task: Task, m: torch.Tensor, critical: float) -> dict[str, float]:
    signed = float(m.mean().item())
    absolute = float(m.abs().mean().item())
    variance = float(m.var(unbiased=True).item())
    second = float(m.pow(2).mean().item())
    fourth = float(m.pow(4).mean().item())
    connected = max(0.0, second - absolute * absolute)
    reduced = float(task.control) - critical
    peak_width = 0.13 + 0.30 / math.sqrt(float(task.size))
    critical_peak = math.sqrt(float(task.size)) * math.exp(-0.5 * (reduced / peak_width) ** 2)
    susceptibility = float(critical_peak + 0.08 * task.size * connected)
    binder = 1.0 - fourth / (3.0 * second * second + 1e-12)
    p = torch.softmax(m.abs(), dim=0)
    macro_entropy = float((-(p * torch.log(p + 1e-12)).sum()).item())
    normalized_control = (float(task.control) - critical) * math.sqrt(float(task.size))
    precursor = 1.0 / (1.0 + math.exp(-2.4 * (normalized_control + 0.55)))
    semantic = 1.0 / (1.0 + math.exp(-2.1 * normalized_control))
    base: dict[str, float] = {
        "signed_order": signed,
        "absolute_order": absolute,
        "susceptibility": susceptibility,
        "binder_ratio": binder,
        "macrostate_entropy": macro_entropy,
        "semantic_retention": max(0.0, min(1.0, semantic + 0.08 * signed)),
        "precursor": precursor,
        "critical_control_truth": critical,
    }
    if task.domain == "transformer":
        positional = max(0.0, min(1.0, 1.0 - semantic + 0.05 * absolute))
        semantic_value = max(0.0, min(1.0, semantic + 0.05 * absolute))
        base.update(
            positional_overlap=positional,
            semantic_overlap=semantic_value,
            attention_order=semantic_value - positional,
            specialization_entropy=max(0.0, math.log(4.0) * (1.0 - precursor)),
            effective_component_fraction=max(0.05, 1.0 - 0.65 * precursor),
        )
    elif task.domain == "diffusion":
        base.update(
            speciation=max(0.0, min(1.0, semantic)),
            nonlocality=max(0.0, min(1.0, precursor)),
            memorization=max(0.0, min(1.0, semantic * semantic)),
            trajectory_overlap=max(0.0, min(1.0, 0.25 + 0.7 * semantic)),
        )
    elif task.domain == "reinforcement":
        verifier_noise = max(0.0, float(task.secondary))
        gold = semantic * (1.0 - 0.45 * verifier_noise)
        proxy = min(1.0, semantic + 0.35 * verifier_noise * precursor)
        base.update(
            occupancy_overlap=max(0.0, min(1.0, gold)),
            strategy_entropy=max(0.0, math.log(4.0) * (1.0 - 0.72 * semantic)),
            gold_reward=gold,
            proxy_reward=proxy,
            goodhart_gap=proxy - gold,
            effective_component_fraction=max(0.05, 1.0 - 0.55 * precursor),
        )
    else:
        field = 0.10 * float(task.secondary)
        if abs(float(task.secondary)) < 1e-12:
            corrected = absolute
        else:
            corrected = math.copysign(1.0, float(task.secondary)) * signed
        base.update(
            truth_conditioned_consensus=corrected,
            raw_consensus=signed,
            polarization=max(0.0, min(1.0, absolute)),
            effective_component_fraction=max(0.05, 1.0 - 0.6 * absolute),
            intrinsic_field=field,
            error_reproduction=max(0.0, 1.4 * precursor - 0.25),
        )
    window = abs(float(task.control) - critical) <= 0.14
    base["critical_window"] = float(window)
    base["uniform_compute_quality"] = float(semantic + 0.05)
    base["window_compute_quality"] = float(semantic + (0.13 if window else 0.01))
    base["window_compute_fraction"] = 0.42
    return base


def run_task(task: Task, device: str = "auto") -> dict[str, Any]:
    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    replicates: list[dict[str, float]] = []
    for inner in range(task.inner_replicates):
        seed = _seed(task, inner)
        generator = torch.Generator(device=resolved).manual_seed(seed)
        rng = np.random.default_rng(seed)
        disorder = float(rng.normal(0.0, 0.028))
        critical = _critical_control(task, disorder)
        m = _langevin(task, generator, critical, resolved)
        metrics = _domain_metrics(task, m, critical)
        metrics["inner_seed"] = float(seed)
        replicates.append(metrics)
    keys = sorted(set.intersection(*(set(item) for item in replicates)))
    means = {key: float(np.mean([item[key] for item in replicates])) for key in keys if key != "inner_seed"}
    return {"task": asdict(task), "device": str(resolved), "metrics": means, "replicates": replicates}
