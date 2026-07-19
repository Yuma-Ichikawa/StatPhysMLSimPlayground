"""Trainable MLP, U-Net, and DiT denoisers on a matched semantic image ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.metrics import seed_everything
from ...core.schema import TaskSpec
from ..common import common_coordinates, entropy, task_rng


def _prototypes(size: int = 8) -> np.ndarray:
    grid = np.indices((size, size))
    patterns = np.stack(
        (
            np.where(grid[1] < size // 2, 1.0, -1.0),
            np.where(grid[0] < size // 2, 1.0, -1.0),
            np.where(grid[0] >= grid[1], 1.0, -1.0),
            np.where((grid[0] + grid[1]) % 2 == 0, 1.0, -1.0),
        )
    )
    return patterns[:, None].astype(np.float32)


def _images(
    rng: np.random.Generator,
    prototypes: np.ndarray,
    count: int,
    sigma: float,
    *,
    ood: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = rng.integers(0, len(prototypes), size=count)
    clean = prototypes[labels].copy()
    if ood:
        clean = np.roll(clean, shift=1, axis=-1)
    clean += 0.05 * rng.normal(size=clean.shape)
    noisy = clean + sigma * rng.normal(size=clean.shape)
    return noisy.astype(np.float32), clean.astype(np.float32), labels


class _MLPDenoiser(nn.Module):
    def __init__(self, width: int, image_size: int) -> None:
        super().__init__()
        pixels = image_size * image_size
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(pixels, 4 * width), nn.SiLU(),
            nn.Linear(4 * width, 4 * width), nn.SiLU(), nn.Linear(4 * width, pixels),
        )
        self.image_size = image_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1, 1, self.image_size, self.image_size)


class _TinyUNet(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        channels = max(8, min(width, 64))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(values)
        middle = self.middle(F.avg_pool2d(skip, 2))
        middle = F.interpolate(middle, size=skip.shape[-2:], mode="nearest")
        return self.decoder(torch.cat((skip, middle), dim=1))


class _TinyDiT(nn.Module):
    def __init__(self, width: int, image_size: int) -> None:
        super().__init__()
        dimension = max(8, min(width, 96))
        while dimension % 4:
            dimension += 1
        patches = (image_size // 2) ** 2
        self.patch = nn.Conv2d(1, dimension, kernel_size=2, stride=2)
        self.position = nn.Parameter(torch.zeros(1, patches, dimension))
        layer = nn.TransformerEncoderLayer(
            dimension, 4, dim_feedforward=4 * dimension, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.unpatch = nn.ConvTranspose2d(dimension, 1, kernel_size=2, stride=2)
        self.grid = image_size // 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.patch(values).flatten(2).transpose(1, 2)
        hidden = self.transformer(hidden + self.position[:, : hidden.shape[1]])
        hidden = hidden.transpose(1, 2).reshape(values.shape[0], -1, self.grid, self.grid)
        return self.unpatch(hidden)


def _model(variant: str, width: int, image_size: int) -> nn.Module:
    if variant == "mlp":
        return _MLPDenoiser(width, image_size)
    if variant == "unet":
        return _TinyUNet(width)
    if variant == "dit":
        return _TinyDiT(width, image_size)
    raise ValueError(f"learned diffusion variant must be mlp, unet, or dit: {variant}")


def _mode_probabilities(
    prediction: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    distance = (prediction[:, None] - prototypes[None]).square().flatten(2).mean(dim=2)
    return (-distance).softmax(dim=1)


def run_learned_diffusion(
    task: TaskSpec, device: torch.device
) -> tuple[dict[str, float], dict[str, Any]]:
    seed_everything(task.seed)
    rng = task_rng(task, "learned_score")
    image_size = max(4, int(task.parameters.get("image_size", 8)))
    if image_size % 2:
        raise ValueError("image_size must be even")
    width = max(8, min(int(task.size), int(task.parameters.get("width_cap", 96))))
    sigma = max(float(task.control), 1e-3)
    train_count = max(16, int(task.parameters.get("train_examples", 256)))
    n_probe = max(16, int(task.parameters.get("n_probe", 256)))
    prototype_array = _prototypes(image_size)
    train_noisy, train_clean, _ = _images(rng, prototype_array, train_count, sigma)
    noisy = torch.as_tensor(train_noisy, device=device)
    clean = torch.as_tensor(train_clean, device=device)

    model = _model(task.variant, width, image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.parameters.get("learning_rate", 2e-3)))
    steps = max(1, int(task.parameters.get("steps", 64)))
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy)
        loss = F.mse_loss(prediction, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    test_noisy, test_clean, labels = _images(rng, prototype_array, n_probe, sigma)
    ood_noisy, ood_clean, ood_labels = _images(rng, prototype_array, n_probe, sigma, ood=True)
    test_noisy_t = torch.as_tensor(test_noisy, device=device)
    test_clean_t = torch.as_tensor(test_clean, device=device)
    ood_noisy_t = torch.as_tensor(ood_noisy, device=device)
    ood_clean_t = torch.as_tensor(ood_clean, device=device)
    prototypes_t = torch.as_tensor(prototype_array, device=device)
    ablated = test_noisy_t.clone()
    ablated[:, :, :, : image_size // 2] = 0.0

    model.eval()
    with torch.no_grad():
        prediction = model(test_noisy_t)
        ood_prediction = model(ood_noisy_t)
        ablated_prediction = model(ablated)
        mode_probabilities = _mode_probabilities(prediction, prototypes_t)
        ood_probabilities = _mode_probabilities(ood_prediction, torch.roll(prototypes_t, 1, dims=-1))
        oracle_probabilities = _mode_probabilities(test_noisy_t, prototypes_t)
        predicted_labels = mode_probabilities.argmax(dim=1)
        ood_predicted = ood_probabilities.argmax(dim=1)
        oracle_clean = prototypes_t[oracle_probabilities.argmax(dim=1)]
    labels_t = torch.as_tensor(labels, device=device)
    ood_labels_t = torch.as_tensor(ood_labels, device=device)
    correct = predicted_labels.eq(labels_t)
    ood_correct = ood_predicted.eq(ood_labels_t)
    per_sample_error = (prediction - test_clean_t).square().flatten(1).mean(dim=1)
    ood_error = (ood_prediction - ood_clean_t).square().flatten(1).mean(dim=1)
    oracle_error = (oracle_clean - test_clean_t).square().flatten(1).mean(dim=1)
    distant_response = float(
        (prediction[:, :, :, image_size // 2 :] - ablated_prediction[:, :, :, image_size // 2 :])
        .abs().mean()
    )
    signed = 2.0 * correct.detach().cpu().numpy().astype(np.float64) - 1.0
    marginal = mode_probabilities.mean(dim=0).detach().cpu().numpy()
    metrics, arrays = common_coordinates(
        signed,
        size=task.size,
        generalization_error=float(per_sample_error.mean()),
        ood_generalization_error=float(ood_error.mean()),
        effective_multiplicity=float(np.exp(entropy(marginal))),
        interaction_range=distant_response,
        oracle_gap=float(max(0.0, float(per_sample_error.mean() - oracle_error.mean()))),
        intervention_response=distant_response,
        extras={
            "semantic_accuracy": float(correct.float().mean()),
            "ood_semantic_accuracy": float(ood_correct.float().mean()),
            "denoising_error": float(per_sample_error.mean()),
            "oracle_denoising_error": float(oracle_error.mean()),
            "distant_response": distant_response,
            "score_norm": float(((prediction - test_noisy_t) / (sigma**2)).square().mean().sqrt()),
            "training_loss": losses[-1],
            "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
            "model_width": float(width),
        },
    )
    arrays.update(
        loss_curve=np.asarray(losses, dtype=np.float32),
        mode_probabilities=mode_probabilities.detach().cpu().numpy().astype(np.float32),
        denoising_errors=per_sample_error.detach().cpu().numpy().astype(np.float32),
    )
    return metrics, arrays


__all__ = ["run_learned_diffusion"]
