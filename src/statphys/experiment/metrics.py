"""
Model-agnostic observables for teacher-student experiments.

These metrics require no analytic theory and work for arbitrary
architectures:

- test_error: generalization error on fresh samples (MSE or 0-1)
- weight_overlap: normalized overlap of matching weight tensors
- linear_cka: centered kernel alignment between representations
- representation_similarity: CKA between teacher and student hidden
  activations on shared inputs (works even when architectures differ)

"""

from collections.abc import Callable

import torch
import torch.nn as nn


@torch.no_grad()
def test_error(
    student: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    dataset,
    n_test: int = 2048,
    task: str = "auto",
) -> float:
    """
    Generalization error of a student on fresh teacher-labelled data.

    Args:
        student: Student model (callable on (n, d) inputs).
        dataset: TeacherStudentDataset (or any object with .sample(n)).
        n_test: Number of fresh test samples.
        task: "regression" (MSE), "classification" (0-1 error), or "auto"
            (classification iff labels take only values in {-1, +1}).

    Returns:
        Scalar test error.

    """
    X, y = dataset.sample(n_test)
    pred = student(X)
    if pred.dim() > 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    if task == "auto":
        is_binary = bool(torch.isin(y.unique(), torch.tensor([-1.0, 1.0], device=y.device)).all())
        task = "classification" if is_binary else "regression"

    if task == "classification":
        pred_labels = torch.where(pred >= 0, 1.0, -1.0)
        return (pred_labels != y).float().mean().item()
    return ((pred - y) ** 2).mean().item() * 0.5


@torch.no_grad()
def weight_overlap(
    student: nn.Module,
    teacher_weights: dict[str, torch.Tensor],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Overlap between student and teacher weight tensors with matching
    names and shapes.

    For each matching parameter pair (w, w0):
        overlap = <w, w0> / (||w|| ||w0||)   if normalize
        overlap = <w, w0> / numel            otherwise

    Note: for permutation-symmetric hidden layers this is a naive
    (alignment-sensitive) measure; use representation_similarity for a
    permutation-invariant alternative.

    Args:
        student: Student model.
        teacher_weights: Dict of teacher parameter tensors
            (e.g. Teacher.named_weights()).
        normalize: Use cosine similarity instead of raw scaled overlap.

    Returns:
        Dict mapping parameter names to overlap values (plus "avg").

    """
    overlaps: dict[str, float] = {}
    for name, param in student.named_parameters():
        w0 = teacher_weights.get(name)
        if w0 is None or w0.shape != param.shape:
            continue
        w = param.detach().flatten()
        w0f = w0.to(w.device).flatten()
        if normalize:
            denom = (w.norm() * w0f.norm()).clamp_min(1e-12)
            overlaps[name] = (w @ w0f / denom).item()
        else:
            overlaps[name] = (w @ w0f / w.numel()).item()

    if overlaps:
        overlaps["avg"] = sum(v for k, v in overlaps.items()) / len(overlaps)
    return overlaps


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear centered kernel alignment (CKA) between two representations.

    CKA is invariant to orthogonal transformations and isotropic scaling,
    making it suitable for comparing hidden representations of different
    architectures (Kornblith et al., 2019).

    Args:
        X: (n, p) activation matrix.
        Y: (n, q) activation matrix (same n).

    Returns:
        CKA similarity in [0, 1].

    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    xty = (X.T @ Y).norm() ** 2
    xtx = (X.T @ X).norm()
    yty = (Y.T @ Y).norm()
    return (xty / (xtx * yty).clamp_min(1e-12)).item()


@torch.no_grad()
def representation_similarity(
    student: nn.Module,
    teacher_model: nn.Module,
    X: torch.Tensor,
    layer_types: tuple[type, ...] = (nn.Linear,),
) -> dict[str, float]:
    """
    CKA between teacher and student intermediate representations.

    Hooks all modules of the given types, runs both networks on the same
    inputs, and computes CKA between each pair of same-index layers.

    Args:
        student: Student network.
        teacher_model: Teacher network (the raw nn.Module).
        X: Shared input batch of shape (n, d).
        layer_types: Module types whose outputs are compared.

    Returns:
        Dict mapping "layer{i}" to CKA values (plus "avg").

    """

    def collect(net: nn.Module) -> list[torch.Tensor]:
        acts: list[torch.Tensor] = []
        hooks = []
        for mod in net.modules():
            if isinstance(mod, layer_types):
                hooks.append(mod.register_forward_hook(lambda _m, _i, out: acts.append(out)))
        try:
            net(X)
        finally:
            for h in hooks:
                h.remove()
        return acts

    acts_s = collect(student)
    acts_t = collect(teacher_model)

    sims: dict[str, float] = {}
    for i, (a_s, a_t) in enumerate(zip(acts_s, acts_t, strict=False)):
        a_s = a_s.flatten(start_dim=1) if a_s.dim() > 2 else a_s
        a_t = a_t.flatten(start_dim=1) if a_t.dim() > 2 else a_t
        sims[f"layer{i}"] = linear_cka(a_s, a_t)

    if sims:
        sims["avg"] = sum(sims.values()) / len(sims)
    return sims
