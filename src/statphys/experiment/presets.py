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
- multi_index_model: K-direction multi-index teacher, subspace-overlap
    order parameter (works even if K_student != K_teacher)
- mixture_classification: Gaussian-mixture classification (generative,
    clustered data), with an exactly verifiable Bayes error
- lora_finetune: frozen "pretrained" backbone + trainable low-rank
    adapter (LoRA-style fine-tuning teacher-student)

All presets accept overrides so they double as documentation of the API.

Example:
    >>> from statphys.experiment.presets import random_mlp
    >>> exp = random_mlp(d=200, hidden=8)
    >>> res = exp.run_sample_complexity(alphas=[1, 2, 4, 8, 16], n_seeds=3)
    >>> res.plot(logy=True)

"""

from typing import Any

import torch
import torch.nn as nn

from statphys.experiment.observables import subspace_overlap, vector_overlap
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


def _first_linear_weight(net: nn.Module) -> torch.Tensor:
    """Return the weight matrix of the first nn.Linear layer found."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            return m.weight.detach()
    raise ValueError("No nn.Linear layer found in module")


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


def multi_index_model(
    d: int = 200,
    k_teacher: int = 3,
    k_student: int = 3,
    noise_std: float = 0.05,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    Multi-index model: y = readout(g(W^T x)), teacher W in R^{k_teacher x d}.

    Generalizes the single-index (perceptron) teacher-student setting to
    K > 1 relevant directions (Ben Arous, Gerace, Krzakala, Zdeborova and
    collaborators). Because the natural order parameter -- the
    "subspace_overlap" metric registered here -- is permutation- and
    basis-invariant, k_student may differ from k_teacher, allowing
    over-/under-parameterized recovery of the relevant subspace to be
    studied directly.
    """
    teacher_net = _mlp(d, k_teacher)
    teacher = Teacher(teacher_net, init="orthogonal", noise_std=noise_std, device=device)
    w_teacher = _first_linear_weight(teacher.model)

    def metric_subspace(student: nn.Module, _dataset: Any) -> float:
        return subspace_overlap(_first_linear_weight(student), w_teacher)["mean_cosine"]

    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _mlp(d, k_student),
        d=d,
        device=device,
        metrics={"subspace_overlap": metric_subspace},
        **kwargs,
    )


def mixture_classification(
    d: int = 200,
    mu: float = 1.5,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    Gaussian-mixture classification: x = y*mu*v + z (generative model).

    Unlike the rest of the presets, the label determines the input
    rather than the other way around (Mignacco, Krzakala, Mezard,
    Urbani, Zdeborova 2020 and related classification-of-mixtures
    literature). The exact Bayes error Phi(-mu * cos(w, v)) gives a
    ground-truth check on the numerically measured generalization error
    (see tests/test_mixture.py and docs/order_parameters.md).
    """
    from statphys.experiment.mixture import GaussianMixtureDataset

    dataset = GaussianMixtureDataset(d=d, mu=mu, device=device)
    teacher = dataset.oracle_teacher()

    def metric_cluster_overlap(student: nn.Module, _dataset: Any) -> float:
        return vector_overlap(_first_linear_weight(student), dataset.v)

    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: nn.Linear(d, 1, bias=False),
        d=d,
        dataset=dataset,
        device=device,
        metrics={"cluster_overlap": metric_cluster_overlap},
        **kwargs,
    )


class _LoRAModel(nn.Module):
    """Frozen 'pretrained' base + trainable low-rank adapter + head."""

    def __init__(
        self,
        d: int,
        hidden: int,
        rank: int,
        base_weight: torch.Tensor,
        zero_init: bool = True,
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.base = nn.Linear(d, hidden, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(base_weight)
        self.base.weight.requires_grad_(False)
        self.rank = rank
        if rank > 0:
            b0 = torch.zeros(hidden, rank) if zero_init else torch.randn(hidden, rank) / hidden**0.5
            self.B = nn.Parameter(b0)
            self.A = nn.Parameter(torch.randn(rank, d) / d**0.5)
        self.act = activation()
        self.readout = nn.Linear(hidden, 1)

    def delta(self) -> torch.Tensor:
        """Learned/true low-rank weight update B @ A."""
        if self.rank == 0:
            return torch.zeros_like(self.base.weight)
        return self.B @ self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ (self.base.weight + self.delta()).T
        return self.readout(self.act(h))


def lora_finetune(
    d: int = 128,
    hidden: int = 16,
    rank_true: int = 2,
    rank_student: int = 2,
    noise_std: float = 0.02,
    device: str = "cpu",
    **kwargs: Any,
) -> TeacherStudentExperiment:
    """
    LoRA-style fine-tuning: frozen shared backbone, trainable low-rank adapter.

    A frozen random "pretrained" weight matrix W0 (d -> hidden) is
    shared by teacher and student, mimicking a frozen foundation model.
    The teacher has a fixed low-rank task update
    Delta_true = B_true A_true (rank `rank_true`, representing "the
    correctly fine-tuned model"); the student starts at Delta = 0 (as
    in real LoRA initialization, Hu et al. 2021) and learns its own
    adapter of rank `rank_student` from alpha = n_finetune / d samples.
    Order parameter: cosine overlap between the learned and true
    low-rank updates ("adapter_overlap"), the natural analogue of
    m_hat for low-rank matrix (rather than vector) recovery.
    """
    base_weight = torch.randn(hidden, d) / d**0.5

    teacher_net = _LoRAModel(d, hidden, rank_true, base_weight, zero_init=False)
    delta_true = teacher_net.delta().detach()
    teacher = Teacher(teacher_net, init=None, noise_std=noise_std, device=device)

    def metric_adapter_overlap(student: nn.Module, _dataset: Any) -> float:
        return vector_overlap(student.delta(), delta_true)

    return TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: _LoRAModel(d, hidden, rank_student, base_weight, zero_init=True),
        d=d,
        device=device,
        metrics={"adapter_overlap": metric_adapter_overlap},
        **kwargs,
    )


PRESETS = {
    "random_mlp": random_mlp,
    "sparse_teacher": sparse_teacher,
    "spiked_teacher": spiked_teacher,
    "mismatched_width": mismatched_width,
    "low_rank_attention": low_rank_attention,
    "hidden_manifold": hidden_manifold,
    "tiny_gpt": tiny_gpt,
    "multi_index_model": multi_index_model,
    "mixture_classification": mixture_classification,
    "lora_finetune": lora_finetune,
}


def get_preset(name: str, **kwargs: Any) -> TeacherStudentExperiment:
    """Instantiate a preset experiment by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name](**kwargs)
