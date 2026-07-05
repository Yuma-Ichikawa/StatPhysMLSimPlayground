"""
Shared building blocks for the frontier (modern-paradigm) experiments.

Kept deliberately small: model factories, a single generic training
loop, function-space overlap measures, and correlated-teacher
construction. Every frontier setting (sft, rlhf, weak_to_strong,
collapse, icl) builds exclusively on these plus `statphys.experiment`
primitives, so new settings can be added without duplicating training
or measurement code.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from statphys.experiment.teacher import Teacher

__all__ = [
    "InputSampler",
    "gaussian_sampler",
    "mlp",
    "train_regression",
    "output_overlap",
    "model_overlap",
    "correlated_teacher",
    "clone_module",
]

# An input sampler maps a sample count n to an (n, d) input batch. It is
# the single injection point for the data distribution: isotropic
# Gaussian by default, or a real-data manifold (images, text windows)
# for trained teachers -- see statphys.frontier.teachers.
InputSampler = Callable[[int], torch.Tensor]


def gaussian_sampler(d: int) -> InputSampler:
    """Standard isotropic input sampler: n -> (n, d) i.i.d. N(0, 1)."""

    def sample(n: int) -> torch.Tensor:
        return torch.randn(n, d)

    return sample


def mlp(d: int, hidden: int, depth: int = 1, out: int = 1) -> nn.Module:
    """Tanh MLP factory: d -> hidden (x depth) -> out."""
    layers: list[nn.Module] = []
    in_f = d
    for _ in range(depth):
        layers += [nn.Linear(in_f, hidden), nn.Tanh()]
        in_f = hidden
    layers.append(nn.Linear(in_f, out))
    return nn.Sequential(*layers)


def clone_module(module: nn.Module) -> nn.Module:
    """Deep copy of a module with detached parameters."""
    import copy

    clone = copy.deepcopy(module)
    for p in clone.parameters():
        p.detach_()
    return clone


def _flat(model: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    out = model(X)
    if out.dim() > 1 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out.flatten()


def train_regression(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 1e-2,
    epochs: int = 1000,
    batch_size: int | None = None,
    weight_decay: float = 0.0,
    tol: float = 1e-7,
    patience: int = 25,
) -> nn.Module:
    """
    Train a model on (X, y) with MSE loss 0.5*(f - y)^2 and Adam.

    The single training loop shared by all frontier settings (SFT
    phases, reward models, weak-to-strong students, collapse
    generations), with plateau-based early stopping.

    Args:
        model: Student module (trained in place).
        X: Inputs of shape (n, d).
        y: Targets of shape (n,).
        lr: Adam learning rate.
        epochs: Max epochs.
        batch_size: Minibatch size (None = full batch).
        weight_decay: L2 regularization.
        tol: Early-stopping improvement tolerance.
        patience: Epochs without improvement before stopping.

    Returns:
        The trained model (same object as `model`).

    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    bs = batch_size or n
    best, stale = float("inf"), 0
    for _ in range(epochs):
        perm = torch.randperm(n) if bs < n else torch.arange(n)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = _flat(model, X[idx])
            loss = 0.5 * ((pred - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        if epoch_loss < best - tol:
            best, stale = epoch_loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return model


@torch.no_grad()
def output_overlap(
    f_a: torch.Tensor,
    f_b: torch.Tensor,
) -> float:
    """
    Normalized function-space overlap between two output vectors.

    m = E[f_a f_b] / sqrt(E[f_a^2] E[f_b^2]) in [-1, 1] -- the same
    normalization as m_hat in `statphys.experiment.observables`.

    Args:
        f_a: Outputs of shape (n,).
        f_b: Outputs of shape (n,).

    Returns:
        Cosine-type overlap.

    """
    num = (f_a * f_b).mean()
    den = (f_a.pow(2).mean() * f_b.pow(2).mean()).sqrt().clamp_min(1e-12)
    return float((num / den).item())


@torch.no_grad()
def model_overlap(
    model: Callable[[torch.Tensor], torch.Tensor],
    reference: Callable[[torch.Tensor], torch.Tensor],
    X_probe: torch.Tensor,
) -> float:
    """
    Normalized overlap between a model and a reference function.

    Evaluates both on a shared probe set and returns m_hat. The
    reference may be a `Teacher` (its clean/noiseless output is used
    when available), another nn.Module, or any callable.

    Args:
        model: Model under measurement.
        reference: Reference function (Teacher, module, or callable).
        X_probe: Probe inputs of shape (n, d).

    Returns:
        m_hat in [-1, 1].

    """
    f_a = _flat(model, X_probe)
    ref_fn = reference.clean if isinstance(reference, Teacher) else reference
    f_b = _flat(ref_fn, X_probe).to(f_a.device)
    return output_overlap(f_a, f_b)


def correlated_teacher(
    teacher: Teacher,
    similarity: float,
    seed: int = 0,
) -> Teacher:
    """
    Construct a second teacher with controlled weight-space similarity.

    Every weight matrix W_B of the new teacher is built as

        W_B = cos(theta) * W_A + sin(theta) * W_rand,   cos(theta) = similarity,

    with W_rand an independent Gaussian matrix of matching per-row norm,
    so `similarity` in [0, 1] interpolates from an independent task (0)
    to an identical task (1). This is the minimal "task relatedness"
    dial used across the frontier settings (SFT/transfer, forgetting).

    Args:
        teacher: Base task-A teacher (must wrap an nn.Module).
        similarity: Weight-space cosine between task A and task B.
        seed: RNG seed for the random component.

    Returns:
        A new frozen Teacher for task B with the same architecture,
        readout, and noise settings.

    """
    if not isinstance(teacher.model, nn.Module):
        raise TypeError("correlated_teacher requires a Teacher wrapping an nn.Module")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError(f"similarity must be in [0, 1], got {similarity}")

    gen = torch.Generator().manual_seed(seed)
    model_b = clone_module(teacher.model)
    cos_t = similarity
    sin_t = (1.0 - similarity**2) ** 0.5
    with torch.no_grad():
        for p in model_b.parameters():
            if p.dim() >= 2:
                rand = torch.randn(p.shape, generator=gen, device=p.device)
                rand = rand * (p.std() / rand.std().clamp_min(1e-12))
                p.copy_(cos_t * p + sin_t * rand)
    return Teacher(
        model_b,
        init=None,
        readout=teacher.readout if teacher.readout != "custom" else teacher._readout,
        noise_std=teacher.noise_std,
        flip_prob=teacher.flip_prob,
        device=str(teacher.device),
    )
