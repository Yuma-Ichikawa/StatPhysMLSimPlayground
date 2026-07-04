"""
Architecture zoo for general teacher-student experiments.

A registry of small-but-representative architectures spanning the model
families studied in the statistical-mechanics-of-learning literature and
their modern deep-learning counterparts:

- "linear":       linear map (perceptron / ridge setting)
- "mlp":          shallow MLP (committee-machine-like)
- "deep_mlp":     3-hidden-layer MLP
- "cnn":          1D convolutional network over folded token sequences
- "lstm":         LSTM over folded token sequences
- "attention":    single-head self-attention block (dot-product attention,
                  cf. Cui, Behrens, Krzakala, Zdeborová 2025)
- "tiny_gpt":     causal transformer encoder (embedding + positional
                  encoding + N blocks + pooling head) — a minimal LLM-style
                  architecture for numerical teacher-student studies

All sequence architectures fold the flat d-dimensional input into
(seq_len, d/seq_len) token sequences, so every architecture consumes the
same (n, d) inputs and is interchangeable inside
TeacherStudentExperiment.

Example:
    >>> from statphys.experiment.zoo import build_architecture, ARCHITECTURES
    >>> net = build_architecture("tiny_gpt", d=256, seq_len=8)
    >>> sorted(ARCHITECTURES)
    ['attention', 'cnn', 'deep_mlp', 'linear', 'lstm', 'mlp', 'tiny_gpt']

"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def _check_seq(d: int, seq_len: int) -> int:
    if d % seq_len != 0:
        raise ValueError(f"d={d} must be divisible by seq_len={seq_len}")
    return d // seq_len


class _FoldTokens(nn.Module):
    """Reshape flat (n, d) inputs into (n, seq_len, d/seq_len) tokens."""

    def __init__(self, d: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.token_dim = _check_seq(d, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.seq_len, self.token_dim)


def build_linear(d: int, bias: bool = False, **_: Any) -> nn.Module:
    """Linear map: the classic perceptron/ridge student."""
    return nn.Linear(d, 1, bias=bias)


def build_mlp(
    d: int,
    hidden: int = 32,
    activation: str = "tanh",
    **_: Any,
) -> nn.Module:
    """Shallow MLP (committee-machine-like)."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    return nn.Sequential(nn.Linear(d, hidden), act(), nn.Linear(hidden, 1))


def build_deep_mlp(
    d: int,
    hidden: int = 64,
    depth: int = 3,
    activation: str = "relu",
    **_: Any,
) -> nn.Module:
    """Deeper MLP with `depth` hidden layers."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    layers: list[nn.Module] = [nn.Linear(d, hidden), act()]
    for _i in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


class _Conv1dNet(nn.Module):
    """1D CNN over folded token sequences."""

    def __init__(self, d: int, seq_len: int, channels: int, kernel_size: int):
        super().__init__()
        self.fold = _FoldTokens(d, seq_len)
        token_dim = self.fold.token_dim
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(token_dim, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.fold(x).transpose(1, 2)  # (n, token_dim, seq_len)
        h = self.conv(tokens).mean(dim=2)
        return self.head(h)


def build_cnn(
    d: int,
    seq_len: int = 8,
    channels: int = 32,
    kernel_size: int = 3,
    **_: Any,
) -> nn.Module:
    """1D convolutional network over folded sequences."""
    return _Conv1dNet(d, seq_len, channels, kernel_size)


class _LSTMNet(nn.Module):
    """LSTM over folded token sequences."""

    def __init__(self, d: int, seq_len: int, hidden: int, num_layers: int):
        super().__init__()
        self.fold = _FoldTokens(d, seq_len)
        self.lstm = nn.LSTM(self.fold.token_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.fold(x))
        return self.head(h[:, -1])


def build_lstm(
    d: int,
    seq_len: int = 8,
    hidden: int = 32,
    num_layers: int = 1,
    **_: Any,
) -> nn.Module:
    """LSTM sequence model."""
    return _LSTMNet(d, seq_len, hidden, num_layers)


class _AttentionNet(nn.Module):
    """Single-head self-attention block over folded sequences."""

    def __init__(self, d: int, seq_len: int, d_model: int):
        super().__init__()
        self.fold = _FoldTokens(d, seq_len)
        self.embed = nn.Linear(self.fold.token_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(self.fold(x))
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.head(h.mean(dim=1))


def build_attention(
    d: int,
    seq_len: int = 8,
    d_model: int = 32,
    **_: Any,
) -> nn.Module:
    """Single attention layer (dot-product attention toy model)."""
    return _AttentionNet(d, seq_len, d_model)


class _TinyGPT(nn.Module):
    """
    Minimal causal transformer for teacher-student studies.

    Embedding + learned positional encoding + `n_blocks` causal
    pre-norm transformer blocks + mean pooling + linear head.
    """

    def __init__(
        self,
        d: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.fold = _FoldTokens(d, seq_len)
        self.embed = nn.Linear(self.fold.token_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Linear(d_model, 1)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(self.fold(x)) + self.pos
        h = self.blocks(h, mask=self.causal_mask)
        return self.head(h.mean(dim=1))


def build_tiny_gpt(
    d: int,
    seq_len: int = 8,
    d_model: int = 32,
    n_heads: int = 2,
    n_blocks: int = 2,
    **_: Any,
) -> nn.Module:
    """Minimal LLM-style causal transformer."""
    return _TinyGPT(d, seq_len, d_model, n_heads, n_blocks)


ARCHITECTURES: dict[str, Callable[..., nn.Module]] = {
    "linear": build_linear,
    "mlp": build_mlp,
    "deep_mlp": build_deep_mlp,
    "cnn": build_cnn,
    "lstm": build_lstm,
    "attention": build_attention,
    "tiny_gpt": build_tiny_gpt,
}


def build_architecture(name: str, d: int, **kwargs: Any) -> nn.Module:
    """
    Instantiate an architecture from the zoo by name.

    Args:
        name: One of ARCHITECTURES keys.
        d: Flat input dimension.
        **kwargs: Architecture-specific options (hidden, seq_len, d_model, ...).

    Returns:
        A fresh nn.Module.

    """
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](d=d, **kwargs)


def architecture_experiment(
    name: str,
    d: int = 256,
    teacher_init: str | None = "normal",
    init_kwargs: dict[str, Any] | None = None,
    noise_std: float = 0.0,
    arch_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    **exp_kwargs: Any,
):
    """
    Build a matched teacher-student experiment for a zoo architecture.

    Teacher and student share the architecture; the teacher's weights are
    re-drawn with the requested initialization strategy.

    Args:
        name: Architecture name from the zoo.
        d: Flat input dimension.
        teacher_init: Weight-init strategy for the teacher
            (see statphys.experiment.init_weights_), or None to keep
            the default PyTorch initialization.
        init_kwargs: Options for the init strategy.
        noise_std: Teacher label noise.
        arch_kwargs: Options forwarded to the architecture builder.
        device: Torch device.
        **exp_kwargs: Extra options for TeacherStudentExperiment.

    Returns:
        TeacherStudentExperiment.

    """
    from statphys.experiment.protocol import TeacherStudentExperiment
    from statphys.experiment.teacher import Teacher

    arch_kwargs = arch_kwargs or {}
    teacher = Teacher(
        build_architecture(name, d=d, **arch_kwargs),
        init=teacher_init,
        init_kwargs=init_kwargs,
        noise_std=noise_std,
        device=device,
    )
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: build_architecture(name, d=d, **arch_kwargs),
        d=d,
        device=device,
        **exp_kwargs,
    )
