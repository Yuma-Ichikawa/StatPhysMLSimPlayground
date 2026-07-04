"""
Ready-made teacher-student experiment presets.

Each preset returns a configured TeacherStudentExperiment showcasing a
setting where interesting phenomena (phase transitions, plateaus,
specialization) are known or expected:

- random_mlp: random-weight MLP teacher, matched student
    (classic committee-style specialization transition)
- sparse_teacher: sparse linear teacher
    (compressed-sensing-like recovery transition vs alpha)
- low_rank_attention: single attention-layer teacher with low-rank
    structure, transformer student (toy LLM-like setting)
- spiked_teacher: planted rank-1 spike inside a noisy linear map
    (BBP-style detectability transition as snr/alpha vary)
- mismatched_width: teacher narrower than student
    (overparameterization effects, benign overfitting regime)

All presets accept overrides so they double as documentation of the API.

Example:
    >>> from statphys.experiment.presets import random_mlp
    >>> exp = random_mlp(d=200, hidden=8)
    >>> res = exp.run_sample_complexity(alphas=[1, 2, 4, 8, 16], n_seeds=3)
    >>> res.plot(logy=True)

"""

from typing import Any

import torch.nn as nn

from statphys.experiment.protocol import TeacherStudentExperiment
from statphys.experiment.teacher import Teacher


def _mlp(d: int, hidden: int, depth: int = 1, activation: type[nn.Module] = nn.Tanh) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = d
    for _ in range(depth):
        layers += [nn.Linear(in_dim, hidden), activation()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def random_mlp(
    d: int = 200,
    hidden: int = 8,
    depth: int = 1,
    noise_std: float = 0.0,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """Random-weight MLP teacher with an identical student architecture."""
    teacher = Teacher(_mlp(d, hidden, depth), init="normal", noise_std=noise_std, device=device)
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _mlp(d, hidden, depth),
        d=d,
        device=device,
        **kwargs,
    )


def sparse_teacher(
    d: int = 400,
    sparsity: float = 0.95,
    noise_std: float = 0.05,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """Sparse linear teacher: recovery transition as alpha grows."""
    teacher = Teacher(
        nn.Linear(d, 1, bias=False),
        init="sparse",
        init_kwargs={"sparsity": sparsity},
        noise_std=noise_std,
        device=device,
    )
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: nn.Linear(d, 1, bias=False),
        d=d,
        device=device,
        **kwargs,
    )


def spiked_teacher(
    d: int = 300,
    snr: float = 2.0,
    noise_std: float = 0.1,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """Rank-1 spiked linear teacher (BBP-style detectability)."""
    teacher = Teacher(
        nn.Linear(d, 1, bias=False),
        init="spiked",
        init_kwargs={"snr": snr},
        noise_std=noise_std,
        device=device,
    )
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: nn.Linear(d, 1, bias=False),
        d=d,
        device=device,
        **kwargs,
    )


def mismatched_width(
    d: int = 200,
    teacher_hidden: int = 4,
    student_hidden: int = 32,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """Overparameterized student learning a narrow teacher."""
    teacher = Teacher(_mlp(d, teacher_hidden), init="normal", device=device)
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _mlp(d, student_hidden),
        d=d,
        device=device,
        **kwargs,
    )


class _TinyAttention(nn.Module):
    """Single-head attention block over a sequence folded from the input."""

    def __init__(self, d: int, seq_len: int, d_model: int):
        super().__init__()
        if d % seq_len != 0:
            raise ValueError(f"d={d} must be divisible by seq_len={seq_len}")
        self.seq_len = seq_len
        self.token_dim = d // seq_len
        self.embed = nn.Linear(self.token_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, x):
        n = x.shape[0]
        tokens = x.reshape(n, self.seq_len, self.token_dim)
        h = self.embed(tokens)
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.readout(h.mean(dim=1))


def low_rank_attention(
    d: int = 256,
    seq_len: int = 8,
    d_model: int = 32,
    rank: int = 2,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    Attention-layer teacher with low-rank weights, attention student.

    A minimal "LLM-like" setting where no analytic theory exists but the
    sample-complexity curve can be measured numerically.
    """
    teacher = Teacher(
        _TinyAttention(d, seq_len, d_model),
        init="low_rank",
        init_kwargs={"rank": rank},
        device=device,
    )
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _TinyAttention(d, seq_len, d_model),
        d=d,
        device=device,
        **kwargs,
    )


def hidden_manifold(
    d: int = 256,
    latent_dim: int = 16,
    hidden: int = 16,
    nonlinearity: str = "tanh",
    noise_std: float = 0.05,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    MLP teacher-student on hidden-manifold inputs (Goldt et al. 2020).

    Inputs live near a latent_dim-dimensional nonlinear manifold instead
    of being isotropic Gaussian — the standard bridge from solvable
    models toward realistic structured data.
    """
    teacher = Teacher(_mlp(d, hidden), init="normal", noise_std=noise_std, device=device)
    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _mlp(d, hidden),
        d=d,
        input_dist="hidden_manifold",
        input_kwargs={"latent_dim": latent_dim, "nonlinearity": nonlinearity},
        device=device,
        **kwargs,
    )


def tiny_gpt(
    d: int = 128,
    seq_len: int = 8,
    d_model: int = 32,
    n_heads: int = 2,
    n_blocks: int = 2,
    noise_std: float = 0.0,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    Minimal LLM-style causal transformer teacher-student pair.

    The most "realistic" preset: embedding + positional encoding +
    causal transformer blocks. No analytic theory exists; order
    parameters are measured purely numerically.
    """
    from statphys.experiment.zoo import build_tiny_gpt

    def make():
        return build_tiny_gpt(
            d, seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_blocks=n_blocks
        )

    teacher = Teacher(make(), init="normal", noise_std=noise_std, device=device)
    return TeacherStudentExperiment(
        teacher=teacher, student_factory=make, d=d, device=device, **kwargs
    )


PRESETS = {
    "random_mlp": random_mlp,
    "sparse_teacher": sparse_teacher,
    "spiked_teacher": spiked_teacher,
    "mismatched_width": mismatched_width,
    "low_rank_attention": low_rank_attention,
    "hidden_manifold": hidden_manifold,
    "tiny_gpt": tiny_gpt,
}


def get_preset(name: str, **kwargs: Any) -> TeacherStudentExperiment:
    """Instantiate a preset experiment by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name](**kwargs)
