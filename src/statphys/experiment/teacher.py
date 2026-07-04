"""
Teacher wrappers and weight-initialization strategies.

A Teacher wraps *any* nn.Module (or callable) and turns it into a
data-generating oracle for teacher-student experiments. The focus is on
customizable weight structure, since the statistics of the teacher's
weights strongly influence learnability and phase transitions:

- "normal":     i.i.d. Gaussian weights (classic random teacher)
- "sparse":     Gaussian with a fraction of entries zeroed (compressible)
- "low_rank":   low-rank factorized weight matrices (structured)
- "orthogonal": orthonormal rows/columns (isometric propagation)
- "power_law":  heavy-tailed weights w_i ~ sign * |t|, t ~ student-t like
- "binary":     Rademacher +-1 weights (discrete perceptron teachers)
- "spiked":     random + rank-1 spike (BBP-style planted structure)

Example:
    >>> import torch.nn as nn
    >>> from statphys.experiment import Teacher
    >>> net = nn.Sequential(nn.Linear(200, 64), nn.Tanh(), nn.Linear(64, 1))
    >>> teacher = Teacher(net, init="low_rank", init_kwargs={"rank": 4})

"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def _sparse_(tensor: torch.Tensor, sparsity: float = 0.9, std: float | None = None) -> None:
    """Gaussian init with a `sparsity` fraction of entries set to zero."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    fan_in = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
    keep = 1.0 - sparsity
    # Rescale so the layer output variance matches a dense Gaussian init
    std = std if std is not None else (1.0 / (fan_in * keep)) ** 0.5
    with torch.no_grad():
        tensor.normal_(0.0, std)
        mask = torch.rand_like(tensor) < sparsity
        tensor[mask] = 0.0


def _low_rank_(tensor: torch.Tensor, rank: int = 1, std: float | None = None) -> None:
    """Low-rank init: W = U V^T / sqrt(rank) with Gaussian factors."""
    if tensor.dim() < 2:
        tensor.data.normal_(0.0, 1.0 / tensor.numel() ** 0.5)
        return
    out_f, in_f = tensor.shape[0], tensor.shape[1]
    rank = max(1, min(rank, min(out_f, in_f)))
    std = std if std is not None else 1.0 / in_f**0.5
    with torch.no_grad():
        u = torch.randn(out_f, rank, device=tensor.device)
        v = torch.randn(rank, in_f, device=tensor.device)
        tensor.copy_(std * (u @ v) / rank**0.5)


def _power_law_(tensor: torch.Tensor, alpha: float = 3.0, std: float | None = None) -> None:
    """Heavy-tailed init using a symmetric Pareto-like distribution."""
    if alpha <= 2.0:
        raise ValueError(f"alpha must be > 2 for finite variance, got {alpha}")
    fan_in = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
    std = std if std is not None else 1.0 / fan_in**0.5
    with torch.no_grad():
        u = torch.rand_like(tensor).clamp_min(1e-12)
        mag = u.pow(-1.0 / alpha) - 1.0  # Pareto(alpha) - 1, mode at 0
        sign = torch.where(torch.rand_like(tensor) < 0.5, -1.0, 1.0)
        w = sign * mag
        w = w / w.std().clamp_min(1e-12) * std
        tensor.copy_(w)


def _binary_(tensor: torch.Tensor, std: float | None = None) -> None:
    """Rademacher +-scale weights."""
    fan_in = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
    scale = std if std is not None else 1.0 / fan_in**0.5
    with torch.no_grad():
        tensor.copy_(torch.where(torch.rand_like(tensor) < 0.5, -scale, scale))


def _spiked_(tensor: torch.Tensor, snr: float = 2.0, std: float | None = None) -> None:
    """Random Gaussian matrix plus a rank-1 planted spike of strength snr."""
    if tensor.dim() < 2:
        tensor.data.normal_(0.0, 1.0 / tensor.numel() ** 0.5)
        return
    out_f, in_f = tensor.shape[0], tensor.shape[1]
    std = std if std is not None else 1.0 / in_f**0.5
    with torch.no_grad():
        noise = torch.randn_like(tensor) * std
        u = torch.randn(out_f, 1, device=tensor.device)
        v = torch.randn(1, in_f, device=tensor.device)
        u = u / u.norm().clamp_min(1e-12)
        v = v / v.norm().clamp_min(1e-12)
        spike = snr * std * (out_f * in_f) ** 0.25 * (u @ v)
        tensor.copy_(noise + spike)


def _orthogonal_(tensor: torch.Tensor, std: float | None = None) -> None:
    if tensor.dim() < 2:
        tensor.data.normal_(0.0, 1.0 / tensor.numel() ** 0.5)
        return
    gain = std if std is not None else 1.0
    nn.init.orthogonal_(tensor, gain=gain)


def _normal_(tensor: torch.Tensor, std: float | None = None) -> None:
    fan_in = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
    std = std if std is not None else 1.0 / fan_in**0.5
    nn.init.normal_(tensor, 0.0, std)


_INIT_FNS: dict[str, Callable[..., None]] = {
    "normal": _normal_,
    "sparse": _sparse_,
    "low_rank": _low_rank_,
    "orthogonal": _orthogonal_,
    "power_law": _power_law_,
    "binary": _binary_,
    "spiked": _spiked_,
}


def init_weights_(
    module: nn.Module,
    method: str = "normal",
    **kwargs: Any,
) -> nn.Module:
    """
    Initialize all weight matrices of a module in-place with a strategy.

    Biases are set to zero. 1D parameters other than biases are given
    Gaussian init.

    Args:
        module: Any nn.Module.
        method: One of "normal", "sparse", "low_rank", "orthogonal",
            "power_law", "binary", "spiked".
        **kwargs: Strategy-specific options, e.g. sparsity=0.9, rank=2,
            alpha=3.0, snr=2.0, std=...

    Returns:
        The same module (for chaining).

    """
    if method not in _INIT_FNS:
        raise ValueError(f"Unknown init method '{method}'. Choose from {sorted(_INIT_FNS)}")
    fn = _INIT_FNS[method]
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("bias"):
                param.zero_()
            elif param.dim() >= 2:
                fn(param, **kwargs)
            else:
                _normal_(param)
    return module


class Teacher:
    """
    Wraps any nn.Module (or plain callable) as a data-generating teacher.

    The wrapped model is frozen (no gradients) and used only to produce
    labels. Supports regression (identity readout), binary classification
    (sign readout), and custom readouts.

    Args:
        model: nn.Module or callable mapping (n, d) inputs to outputs.
        init: Optional weight-init strategy applied to `model`
            (see init_weights_). None keeps the model's own weights.
        init_kwargs: Options for the init strategy.
        readout: "identity" (regression), "sign" (binary classification),
            or a callable applied to the raw teacher output.
        noise_std: Gaussian label noise std (identity readout).
        flip_prob: Label flip probability (sign readout).
        device: Device for the teacher.

    Example:
        >>> teacher = Teacher(nn.Linear(100, 1), init="sparse",
        ...                   init_kwargs={"sparsity": 0.95}, readout="sign")

    """

    def __init__(
        self,
        model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        init: str | None = None,
        init_kwargs: dict[str, Any] | None = None,
        readout: str | Callable[[torch.Tensor], torch.Tensor] = "identity",
        noise_std: float = 0.0,
        flip_prob: float = 0.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.init = init
        self.init_kwargs = init_kwargs or {}

        if isinstance(model, nn.Module):
            if init is not None:
                init_weights_(model, method=init, **self.init_kwargs)
            model = model.to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        self.model = model

        if callable(readout) and not isinstance(readout, str):
            self._readout = readout
            self.readout = "custom"
        elif readout == "identity":
            self._readout = lambda z: z
            self.readout = readout
        elif readout == "sign":
            self._readout = lambda z: torch.where(z >= 0, 1.0, -1.0)
            self.readout = readout
        else:
            raise ValueError(f"Unknown readout: {readout!r} (use 'identity', 'sign' or callable)")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate labels for a batch of inputs of shape (n, d)."""
        z = self.model(x.to(self.device))
        if z.dim() > 1 and z.shape[-1] == 1:
            z = z.squeeze(-1)
        y = self._readout(z)

        if self.readout == "identity" and self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        if self.readout == "sign" and self.flip_prob > 0:
            flip = torch.rand_like(y) < self.flip_prob
            y = torch.where(flip, -y, y)
        return y

    def named_weights(self) -> dict[str, torch.Tensor]:
        """Return teacher weight tensors (empty dict for plain callables)."""
        if isinstance(self.model, nn.Module):
            return {n: p.detach() for n, p in self.model.named_parameters()}
        return {}

    def get_config(self) -> dict[str, Any]:
        """Return a summary config for logging."""
        return {
            "model": type(self.model).__name__,
            "init": self.init,
            "init_kwargs": self.init_kwargs,
            "readout": self.readout,
            "noise_std": self.noise_std,
            "flip_prob": self.flip_prob,
        }

    def __repr__(self) -> str:
        return (
            f"Teacher(model={type(self.model).__name__}, init={self.init}, "
            f"readout={self.readout})"
        )
