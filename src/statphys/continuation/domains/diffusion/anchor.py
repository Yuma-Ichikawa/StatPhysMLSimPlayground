"""Exact and approximate score fields for a hierarchical Gaussian mixture."""

from __future__ import annotations

from typing import Any

import torch

from ...metrics import (
    EPS,
    binary_entropy,
    effective_multiplicity,
    phase_statistics,
    seed_everything,
)
from ...schema import TaskSpec


def _hierarchical_means(
    components: int,
    dimension: int,
    separation: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    signs = torch.cat(
        (
            torch.ones(components // 2, device=device),
            -torch.ones(components - components // 2, device=device),
        )
    )
    means = torch.randn((components, dimension), generator=generator, device=device)
    if dimension > 1:
        sub = means[:, 1:]
        sub = sub / sub.norm(dim=1, keepdim=True).clamp_min(EPS)
        means[:, 1:] = 0.65 * separation * sub
    means[:, 0] = separation * signs
    return means, signs


def _sample(
    means: torch.Tensor,
    count: int,
    data_sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(
        means.shape[0],
        (count,),
        generator=generator,
        device=means.device,
    )
    clean = means[labels] + data_sigma * torch.randn(
        (count, means.shape[1]),
        generator=generator,
        device=means.device,
    )
    return clean, labels


def _posterior_score(
    observations: torch.Tensor,
    centers: torch.Tensor,
    variance: float,
    *,
    topk: int | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared_distance = (
        observations.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1)[None, :]
        - 2.0 * observations @ centers.T
    ).clamp_min(0.0)
    logits = -squared_distance / (2.0 * variance * temperature)
    if topk is not None and topk < centers.shape[0]:
        values, indices = logits.topk(topk, dim=1)
        probabilities = torch.zeros_like(logits)
        probabilities.scatter_(1, indices, torch.softmax(values, dim=1))
    else:
        probabilities = torch.softmax(logits, dim=1)
    denoised_mean = probabilities @ centers
    score = (denoised_mean - observations) / variance
    return probabilities, score


def _global_order(probabilities: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return probabilities @ signs


def _spread(
    probabilities: torch.Tensor,
    centers: torch.Tensor,
    separation: float,
) -> torch.Tensor:
    mean = probabilities @ centers
    second = probabilities @ centers.square().sum(dim=1)
    variance = (second - mean.square().sum(dim=1)).clamp_min(0.0)
    return variance.sqrt() / max(separation, EPS)


def _nearest_distance(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    squared = (
        points.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1)[None, :]
        - 2.0 * points @ centers.T
    ).clamp_min(0.0)
    return squared.min(dim=1).values.sqrt()


def run_diffusion(
    task: TaskSpec,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    generator = torch.Generator(device=device).manual_seed(task.seed)
    dimension = task.size
    components = int(task.parameters.get("components", 8))
    if components < 2:
        raise ValueError("diffusion mixture requires at least two components")
    separation = float(task.parameters.get("separation", 3.0))
    data_sigma = float(task.parameters.get("data_sigma", 0.25))
    noise = max(float(task.control), 1e-3)
    means, signs = _hierarchical_means(
        components, dimension, separation, generator, device
    )
    n_probe = int(task.parameters.get("n_probe", 4096))
    clean, _ = _sample(means, n_probe, data_sigma, generator)
    observations = clean + noise * torch.randn(
        clean.shape, generator=generator, device=device
    )
    true_variance = data_sigma**2 + noise**2
    exact_probabilities, exact_score = _posterior_score(
        observations, means, true_variance
    )

    variant = task.variant.lower()
    model_centers = means
    model_signs = signs
    if variant == "oracle":
        probabilities, score = exact_probabilities, exact_score
    elif variant == "topk":
        probabilities, score = _posterior_score(
            observations,
            means,
            true_variance,
            topk=int(task.parameters.get("topk", 2)),
            temperature=float(task.parameters.get("score_temperature", 1.0)),
        )
    elif variant in {"finite_kde", "kde"}:
        n_centers = int(task.parameters.get("train_centers", min(256, 8 * dimension)))
        model_centers, center_labels = _sample(means, n_centers, data_sigma, generator)
        model_signs = signs[center_labels]
        bandwidth = float(task.parameters.get("bandwidth", data_sigma))
        probabilities, score = _posterior_score(
            observations,
            model_centers,
            noise**2 + bandwidth**2,
        )
    else:
        raise ValueError(f"unknown diffusion variant: {task.variant}")

    signed_order = _global_order(probabilities, model_signs)
    score_errors = (score - exact_score).square().mean(dim=1)
    denoised = observations + noise**2 * score
    memory_distance = _nearest_distance(denoised, model_centers)

    shifted_means = means.clone()
    if dimension > 1:
        shifted_means[:, 1] += float(task.parameters.get("ood_shift", 0.75)) * signs
    ood_clean, _ = _sample(shifted_means, n_probe, data_sigma, generator)
    ood_observations = ood_clean + noise * torch.randn(
        ood_clean.shape, generator=generator, device=device
    )
    _, ood_exact_score = _posterior_score(
        ood_observations, shifted_means, true_variance
    )
    if variant == "oracle":
        _, ood_model_score = _posterior_score(
            ood_observations, means, true_variance
        )
    elif variant == "topk":
        _, ood_model_score = _posterior_score(
            ood_observations,
            means,
            true_variance,
            topk=int(task.parameters.get("topk", 2)),
            temperature=float(task.parameters.get("score_temperature", 1.0)),
        )
    else:
        bandwidth = float(task.parameters.get("bandwidth", data_sigma))
        _, ood_model_score = _posterior_score(
            ood_observations,
            model_centers,
            noise**2 + bandwidth**2,
        )
    ood_errors = (ood_model_score - ood_exact_score).square().mean(dim=1)

    global_probability = ((signed_order + 1.0) / 2.0).clamp(0.0, 1.0)
    interaction = _spread(probabilities, model_centers, separation)
    perturbed_noise = noise * 1.1
    perturbed_variance = data_sigma**2 + perturbed_noise**2
    perturbed_probabilities, _ = _posterior_score(
        observations,
        model_centers,
        perturbed_variance,
        topk=int(task.parameters.get("topk", 2)) if variant == "topk" else None,
    )
    perturbed_order = _global_order(perturbed_probabilities, model_signs)

    metrics = phase_statistics(signed_order, size=dimension)
    metrics.update(
        {
            "generalization_error": float(score_errors.mean()),
            "ood_generalization_error": float(ood_errors.mean()),
            "effective_multiplicity": float(effective_multiplicity(probabilities).mean()),
            "interaction_range": float(interaction.mean()),
            "macrostate_entropy": float(binary_entropy(global_probability).mean()),
            "oracle_gap": float(score_errors.mean()),
            "intervention_response": float(
                (perturbed_order.abs() - signed_order.abs()).abs().mean()
            ),
            "memorization_distance": float(memory_distance.mean()),
            "score_norm": float(score.norm(dim=1).mean()),
        }
    )
    arrays = {
        "signed_order_samples": signed_order.detach().cpu().numpy(),
        "score_errors": score_errors.detach().cpu().numpy(),
        "ood_errors": ood_errors.detach().cpu().numpy(),
        "posterior_multiplicity": effective_multiplicity(probabilities).detach().cpu().numpy(),
        "memorization_distance": memory_distance.detach().cpu().numpy(),
    }
    return metrics, arrays


__all__ = ["run_diffusion"]
