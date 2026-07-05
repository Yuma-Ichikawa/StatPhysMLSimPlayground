r"""
Teacher taxonomy for the frontier experiments.

Every frontier paradigm (SFT, RLHF, weak-to-strong, collapse, ICL) takes
a teacher as input, and the *statistics of the teacher* -- its weight
structure and the data manifold it lives on -- are a physics axis in
their own right (cf. structured-weight and hidden-manifold results in
the exactly solvable literature). This module makes that axis explicit:
a registry of named teacher constructors spanning

- **random**: i.i.d. Gaussian weights on Gaussian inputs -- the
  classic solvable baseline,
- **structured**: sparse / low-rank / heavy-tailed / spiked / binary
  weights -- planted structure that changes learnability,
- **trained**: a network actually *trained on real data* (scikit-learn
  digits, 8x8 images, d = 64), evaluated on the real image manifold --
  the closest this taxonomy gets to a "real" teacher without external
  downloads.

Each entry returns ``(teacher, input_sampler, d)`` so any frontier
protocol can be swept across the whole taxonomy uniformly (see
`statphys.frontier.taxonomy`).

Example:
    >>> from statphys.frontier.teachers import make_teacher, TEACHER_TAXONOMY
    >>> teacher, sampler, d = make_teacher("trained_digits", seed=0)
    >>> X = sampler(128)          # real digit images, standardized
    >>> y = teacher(X)            # trained-network labels

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from statphys.experiment.teacher import Teacher
from statphys.frontier.common import InputSampler, gaussian_sampler, mlp, train_regression
from statphys.utils.seed import fix_seed

__all__ = ["TEACHER_TAXONOMY", "TeacherSpec", "make_teacher", "taxonomy_table"]

MakeFn = Callable[[int, int, int, float], tuple[Teacher, InputSampler, int]]


@dataclass(frozen=True)
class TeacherSpec:
    """
    One entry of the teacher taxonomy.

    Attributes:
        name: Registry key.
        family: "random", "structured", or "trained".
        weights: Short description of the weight statistics.
        inputs: Short description of the input distribution.
        make: Constructor (d, hidden, seed, noise_std) ->
            (teacher, input_sampler, effective_d).

    """

    name: str
    family: str
    weights: str
    inputs: str
    make: MakeFn


def _init_teacher(
    init: str,
    init_kwargs: dict | None = None,
) -> MakeFn:
    """Build a Gaussian-input MLP teacher constructor for a weight init."""

    def make(d: int, hidden: int, seed: int, noise_std: float):
        fix_seed(seed)
        teacher = Teacher(
            mlp(d, hidden), init=init, init_kwargs=init_kwargs or {}, noise_std=noise_std
        )
        return teacher, gaussian_sampler(d), d

    return make


def _linear_teacher(d: int, hidden: int, seed: int, noise_std: float):
    """Single-index linear teacher (hidden width is ignored)."""
    fix_seed(seed)
    teacher = Teacher(torch.nn.Linear(d, 1), init="normal", noise_std=noise_std)
    return teacher, gaussian_sampler(d), d


_DIGITS_D = 64


def _load_digits_tensor() -> torch.Tensor:
    """Load and globally standardize the sklearn digits images (n, 64)."""
    from sklearn.datasets import load_digits

    X, _ = load_digits(return_X_y=True)
    X = torch.as_tensor(X, dtype=torch.float32)
    return (X - X.mean()) / X.std().clamp_min(1e-12)


def _digits_sampler(X: torch.Tensor, jitter: float = 0.1) -> InputSampler:
    """
    Sample real digit images (rows of X) with small Gaussian jitter.

    The jitter turns the finite dataset into a smooth manifold so that
    repeated draws are never exactly identical (fresh-sample protocols
    stay meaningful).
    """

    def sample(n: int) -> torch.Tensor:
        idx = torch.randint(0, X.shape[0], (n,))
        return X[idx] + jitter * torch.randn(n, X.shape[1])

    return sample


def _trained_digits_teacher(d: int, hidden: int, seed: int, noise_std: float):
    """
    An MLP genuinely trained on real images: digit parity regression.

    The teacher is trained to regress y = +1 (digit >= 5) / -1
    (digit < 5) on the standardized 8x8 sklearn digits, then frozen.
    Inputs are drawn from the real image manifold (with jitter), so both
    the weight statistics and the data statistics are "real".
    """
    fix_seed(seed)
    X = _load_digits_tensor()
    from sklearn.datasets import load_digits

    _, y = load_digits(return_X_y=True)
    target = torch.where(torch.as_tensor(y) >= 5, 1.0, -1.0)

    net = mlp(_DIGITS_D, max(hidden, 32))
    train_regression(net, X, target, lr=5e-3, epochs=2000, batch_size=256)
    teacher = Teacher(net, init=None, noise_std=noise_std)
    return teacher, _digits_sampler(X), _DIGITS_D


def _random_digits_teacher(d: int, hidden: int, seed: int, noise_std: float):
    """Random-weight MLP teacher evaluated on the real digits manifold.

    The control for `trained_digits`: identical (real) input statistics
    but unstructured weights, isolating the effect of *trained* weight
    structure from the effect of data structure.
    """
    fix_seed(seed)
    X = _load_digits_tensor()
    teacher = Teacher(mlp(_DIGITS_D, hidden), init="normal", noise_std=noise_std)
    return teacher, _digits_sampler(X), _DIGITS_D


TEACHER_TAXONOMY: dict[str, TeacherSpec] = {
    "random_mlp": TeacherSpec(
        "random_mlp",
        "random",
        "i.i.d. Gaussian",
        "Gaussian",
        _init_teacher("normal"),
    ),
    "linear": TeacherSpec(
        "linear",
        "random",
        "single index (Gaussian vector)",
        "Gaussian",
        _linear_teacher,
    ),
    "sparse_mlp": TeacherSpec(
        "sparse_mlp",
        "structured",
        "90% zeros (compressible)",
        "Gaussian",
        _init_teacher("sparse", {"sparsity": 0.9}),
    ),
    "low_rank_mlp": TeacherSpec(
        "low_rank_mlp",
        "structured",
        "rank-2 factorized",
        "Gaussian",
        _init_teacher("low_rank", {"rank": 2}),
    ),
    "power_law_mlp": TeacherSpec(
        "power_law_mlp",
        "structured",
        "heavy-tailed (Pareto-like)",
        "Gaussian",
        _init_teacher("power_law", {"alpha": 3.0}),
    ),
    "spiked_mlp": TeacherSpec(
        "spiked_mlp",
        "structured",
        "Gaussian + rank-1 spike",
        "Gaussian",
        _init_teacher("spiked", {"snr": 2.0}),
    ),
    "binary_mlp": TeacherSpec(
        "binary_mlp",
        "structured",
        "Rademacher +-1",
        "Gaussian",
        _init_teacher("binary"),
    ),
    "random_digits": TeacherSpec(
        "random_digits",
        "structured",
        "i.i.d. Gaussian",
        "real images (digits)",
        _random_digits_teacher,
    ),
    "trained_digits": TeacherSpec(
        "trained_digits",
        "trained",
        "trained on digit parity",
        "real images (digits)",
        _trained_digits_teacher,
    ),
}


def make_teacher(
    name: str,
    d: int = 64,
    hidden: int = 16,
    seed: int = 0,
    noise_std: float = 0.1,
) -> tuple[Teacher, InputSampler, int]:
    """
    Instantiate a teacher from the taxonomy.

    Args:
        name: Key in TEACHER_TAXONOMY.
        d: Requested input dimension (real-data teachers override it).
        hidden: Teacher hidden width (where applicable).
        seed: Random seed.
        noise_std: Label noise.

    Returns:
        (teacher, input_sampler, effective_d) -- effective_d may differ
        from `d` for real-data teachers (digits: 64).

    """
    if name not in TEACHER_TAXONOMY:
        raise ValueError(f"Unknown teacher '{name}'. Choose from {sorted(TEACHER_TAXONOMY)}")
    return TEACHER_TAXONOMY[name].make(d, hidden, seed, noise_std)


def taxonomy_table() -> str:
    """Render the teacher taxonomy as a markdown table."""
    lines = [
        "| name | family | weights | inputs |",
        "|---|---|---|---|",
    ]
    for spec in TEACHER_TAXONOMY.values():
        lines.append(f"| `{spec.name}` | {spec.family} | {spec.weights} | {spec.inputs} |")
    return "\n".join(lines)
