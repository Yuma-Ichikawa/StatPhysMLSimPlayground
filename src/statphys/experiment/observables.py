"""
Physics-style order parameters for theory-free teacher-student experiments.

All observables are defined in *function space* on a shared probe set, so
they apply to arbitrary architectures (MLPs, CNNs, attention, ...), not
only to models with a single weight vector:

- function_order_params: magnetization m_f = E[f_s f_t] (teacher-student
  overlap), self-overlap q_f = E[f_s^2], teacher norm rho_f, and the
  normalized overlap m_hat = m_f / sqrt(q_f rho_f) in [-1, 1]
- replica_overlaps: overlap q_ab between independently trained students
  ("replicas") at the same sample ratio -- the numerical analogue of the
  replica-symmetric order parameter
- susceptibility: chi = scale * Var[m] over replicas; peaks at transitions
- binder_cumulant: U_4 = 1 - <m^4> / (3 <m^2>^2); curves for different d
  cross at the critical point (finite-size scaling)
- participation_ratio: effective dimension of hidden representations
- specialization_index: permutation-resolved diagonal dominance of the
  student-teacher hidden-unit overlap matrix (committee-machine
  specialization for any matched pair of architectures)

Together with test_error these give a statistical-physics dashboard for
locating and characterizing phase transitions purely numerically.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from statphys.experiment.teacher import Teacher

__all__ = [
    "binder_cumulant",
    "function_order_params",
    "participation_ratio",
    "replica_overlaps",
    "specialization_index",
    "susceptibility",
]


@torch.no_grad()
def _flat_output(model, X: torch.Tensor) -> torch.Tensor:
    out = model(X)
    if out.dim() > 1 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out.flatten()


@torch.no_grad()
def function_order_params(
    student: nn.Module,
    teacher: Teacher,
    X: torch.Tensor,
) -> dict[str, float]:
    """
    Function-space order parameters on a probe set.

    Uses the *noiseless* teacher output, so m_hat -> 1 means perfect
    recovery of the teacher function regardless of label noise.

    Args:
        student: Trained student model.
        teacher: Teacher (labels are taken via teacher.clean).
        X: Probe inputs of shape (n, d).

    Returns:
        Dict with m_f, q_f, rho_f, and normalized overlap m_hat.

    """
    f_s = _flat_output(student, X)
    f_t = teacher.clean(X).flatten().to(f_s.device)

    m_f = (f_s * f_t).mean().item()
    q_f = (f_s**2).mean().item()
    rho_f = (f_t**2).mean().item()
    denom = max(np.sqrt(max(q_f, 0.0) * max(rho_f, 0.0)), 1e-12)
    return {"m_f": m_f, "q_f": q_f, "rho_f": rho_f, "m_hat": m_f / denom}


@torch.no_grad()
def replica_overlaps(
    students: Sequence[nn.Module],
    X: torch.Tensor,
    normalize: bool = True,
) -> dict[str, float | list[float]]:
    """
    Pairwise function-space overlaps between independently trained
    students (replicas) at the same experimental conditions.

    q_ab close to 1 means all replicas found essentially the same
    function (a condensed / ordered phase); small q_ab signals many
    distinct minima (glassy / underdetermined regime).

    Args:
        students: Trained student models (>= 2).
        X: Shared probe inputs of shape (n, d).
        normalize: Use cosine overlaps (divide by function norms).

    Returns:
        Dict with "q_ab_mean", "q_ab_std", and the list of pairwise
        values "q_ab_pairs".

    """
    outs = [_flat_output(s, X) for s in students]
    pairs: list[float] = []
    for a in range(len(outs)):
        for b in range(a + 1, len(outs)):
            num = (outs[a] * outs[b]).mean()
            if normalize:
                den = (outs[a].pow(2).mean() * outs[b].pow(2).mean()).sqrt().clamp_min(1e-12)
                pairs.append((num / den).item())
            else:
                pairs.append(num.item())
    if not pairs:
        return {"q_ab_mean": float("nan"), "q_ab_std": float("nan"), "q_ab_pairs": []}
    arr = np.asarray(pairs)
    return {"q_ab_mean": float(arr.mean()), "q_ab_std": float(arr.std()), "q_ab_pairs": pairs}


def susceptibility(samples: Sequence[float], scale: float = 1.0) -> float:
    """
    Finite-size susceptibility chi = scale * Var[samples].

    With samples = per-replica order parameters (e.g. m_hat) and
    scale = d, chi develops a peak at the transition that sharpens
    with increasing d.

    Args:
        samples: Per-replica values of an order parameter.
        scale: Multiplicative factor (typically the dimension d).

    Returns:
        Scalar susceptibility.

    """
    arr = np.asarray(list(samples), dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(scale * arr.var(ddof=1))


def binder_cumulant(samples: Sequence[float]) -> float:
    """
    Binder cumulant U_4 = 1 - <m^4> / (3 <m^2>^2) over replicas.

    U_4 is scale-invariant at criticality: curves U_4(alpha) for
    different d intersect near the critical sample ratio, giving a
    finite-size-scaling estimate of the transition without fitting.

    Args:
        samples: Per-replica order-parameter values.

    Returns:
        Scalar Binder cumulant (nan if fewer than 2 samples).

    """
    arr = np.asarray(list(samples), dtype=float)
    if arr.size < 2:
        return float("nan")
    m2 = float((arr**2).mean())
    m4 = float((arr**4).mean())
    if m2 <= 1e-24:
        return float("nan")
    return 1.0 - m4 / (3.0 * m2**2)


@torch.no_grad()
def participation_ratio(acts: torch.Tensor) -> float:
    """
    Participation ratio (effective dimension) of a representation.

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2 for the eigenvalues
    lambda_i of the activation covariance. PR ~ 1 means a collapsed
    (rank-one) representation; PR ~ width means fully isotropic.

    Args:
        acts: Activation matrix of shape (n, p).

    Returns:
        Scalar effective dimension in [1, p].

    """
    acts = acts - acts.mean(dim=0, keepdim=True)
    cov = acts.T @ acts / max(acts.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    s1 = eig.sum()
    s2 = (eig**2).sum().clamp_min(1e-24)
    return float((s1**2 / s2).item())


@torch.no_grad()
def specialization_index(
    student: nn.Module,
    teacher: Teacher,
    layer: int = 0,
) -> float:
    """
    Permutation-resolved specialization of hidden units.

    Computes the normalized overlap matrix M_km between the rows of the
    student's and teacher's `layer`-th Linear weight, greedily matches
    student units to teacher units, and returns

        (mean matched overlap) - (mean unmatched overlap)

    Near 0: unspecialized (symmetric) phase. Near 1: each student unit
    aligned with a distinct teacher unit (specialized phase). The greedy
    matching makes the index invariant to hidden-unit permutations.

    Args:
        student: Student model with at least `layer`+1 Linear modules.
        teacher: Teacher whose model has matching Linear shapes.
        layer: Which Linear layer to compare (0 = first).

    Returns:
        Scalar specialization index (nan if shapes do not match).

    """

    def nth_linear_weight(net: nn.Module, idx: int) -> torch.Tensor | None:
        i = 0
        for mod in net.modules():
            if isinstance(mod, nn.Linear):
                if i == idx:
                    return mod.weight.detach()
                i += 1
        return None

    w_s = nth_linear_weight(student, layer)
    if not isinstance(teacher.model, nn.Module):
        return float("nan")
    w_t = nth_linear_weight(teacher.model, layer)
    if w_s is None or w_t is None or w_s.shape[1] != w_t.shape[1]:
        return float("nan")

    w_s = w_s / w_s.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w_t = w_t.to(w_s.device) / w_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    M = (w_s @ w_t.T).abs()  # (K_s, K_t) cosine overlaps

    k = min(M.shape)
    work = M.clone()
    matched = 0.0
    for _ in range(k):
        idx = torch.argmax(work)
        r, c = idx // work.shape[1], idx % work.shape[1]
        matched += work[r, c].item()
        work[r, :] = -1.0
        work[:, c] = -1.0
    matched /= k

    total = M.sum().item()
    n_unmatched = M.numel() - k
    unmatched = (total - matched * k) / n_unmatched if n_unmatched > 0 else 0.0
    return matched - unmatched
