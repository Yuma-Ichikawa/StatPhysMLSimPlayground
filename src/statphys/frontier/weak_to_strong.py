r"""
Weak-to-strong generalization in a teacher-student chain.

Setting (Burns et al. 2023, reduced to its statistical core). A true
teacher :math:`f^*` generates the world. A *weak supervisor* -- a
small-capacity student -- is trained on :math:`\alpha_w d` true labels
and reaches imperfect overlap :math:`m_{\rm weak} < 1`. A *strong
student* (larger capacity) is then trained **only on the weak
supervisor's labels** on fresh inputs (:math:`\alpha_s d` samples) and
never sees a true label.

Order parameters:

- :math:`m_{\rm weak}` -- weak supervisor's overlap with the truth
- :math:`m_{\rm strong}` -- strong student's overlap with the *truth*
  (not with the weak supervisor!)
- :math:`m_{\rm imit}` -- strong student's overlap with the weak
  supervisor (imitation fidelity)
- performance gap recovered
  :math:`\mathrm{PGR} = (m_{\rm strong} - m_{\rm weak}) /
  (m_{\rm ceil} - m_{\rm weak})`, where :math:`m_{\rm ceil}` is the
  strong architecture trained directly on true labels with the same
  budget. PGR > 0 means the student *surpasses its supervisor*.

Phenomenology probed. Whether (and when) weak-to-strong gain appears:
the strong student can exceed the weak teacher when the weak teacher's
errors are "unstructured" enough for the strong prior to average them
out. The interesting object is PGR as a function of
:math:`(\alpha_w, \alpha_s)` -- an imitation-to-generalization
crossover surface.
"""

from __future__ import annotations

import numpy as np
import torch

from statphys.frontier.common import (
    InputSampler,
    gaussian_sampler,
    mlp,
    model_overlap,
    train_regression,
)
from statphys.experiment.teacher import Teacher
from statphys.utils.seed import fix_seed

__all__ = ["run_weak_to_strong", "sweep_weak_to_strong"]


def run_weak_to_strong(
    d: int = 64,
    hidden_true: int = 16,
    hidden_weak: int = 2,
    hidden_strong: int = 32,
    alpha_weak: float = 4.0,
    alpha_strong: float = 16.0,
    lr: float = 5e-3,
    epochs: int = 3000,
    noise_std: float = 0.1,
    n_probe: int = 4096,
    seed: int = 0,
    teacher: Teacher | None = None,
    input_sampler: InputSampler | None = None,
) -> dict:
    """
    Run one weak-to-strong chain: truth -> weak supervisor -> strong student.

    Args:
        d: Input dimension.
        hidden_true: Hidden width of the ground-truth teacher.
        hidden_weak: Hidden width of the weak supervisor (small capacity).
        hidden_strong: Hidden width of the strong student.
        alpha_weak: True-label samples for the weak supervisor, / d.
        alpha_strong: Weak-label samples for the strong student, / d.
        lr: Adam learning rate.
        epochs: Max epochs per training stage.
        noise_std: Label noise on the true labels.
        n_probe: Probe-set size.
        seed: Random seed.
        teacher: Optional ground-truth teacher (taxonomy injection);
            defaults to a random-weight tanh MLP of width hidden_true.
        input_sampler: Optional input distribution n -> (n, d).

    Returns:
        Dict with "m_weak", "m_strong", "m_imit", "m_ceiling", "pgr",
        and the config.

    """
    fix_seed(seed)
    truth = (
        teacher
        if teacher is not None
        else Teacher(mlp(d, hidden_true), init="normal", noise_std=noise_std)
    )
    sample = input_sampler if input_sampler is not None else gaussian_sampler(d)
    X_probe = sample(n_probe)

    # --- weak supervisor: small net on few true labels ---
    n_w = max(int(alpha_weak * d), 1)
    X_w = sample(n_w)
    weak = mlp(d, hidden_weak)
    train_regression(weak, X_w, truth(X_w), lr=lr, epochs=epochs)
    m_weak = model_overlap(weak, truth, X_probe)

    # --- strong student: large net on weak labels only ---
    n_s = max(int(alpha_strong * d), 1)
    X_s = sample(n_s)
    with torch.no_grad():
        y_weak = weak(X_s).squeeze(-1)
    strong = mlp(d, hidden_strong)
    train_regression(strong, X_s, y_weak, lr=lr, epochs=epochs)
    m_strong = model_overlap(strong, truth, X_probe)
    m_imit = model_overlap(strong, weak, X_probe)

    # --- ceiling: same strong architecture on true labels ---
    fix_seed(seed + 1)
    ceiling = mlp(d, hidden_strong)
    train_regression(ceiling, X_s, truth(X_s), lr=lr, epochs=epochs)
    m_ceiling = model_overlap(ceiling, truth, X_probe)

    gap = m_ceiling - m_weak
    pgr = (m_strong - m_weak) / gap if abs(gap) > 1e-9 else float("nan")
    return {
        "m_weak": m_weak,
        "m_strong": m_strong,
        "m_imit": m_imit,
        "m_ceiling": m_ceiling,
        "pgr": pgr,
        "config": {
            "d": d,
            "hidden_true": hidden_true,
            "hidden_weak": hidden_weak,
            "hidden_strong": hidden_strong,
            "alpha_weak": alpha_weak,
            "alpha_strong": alpha_strong,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


def sweep_weak_to_strong(
    alphas_weak: np.ndarray | list[float],
    alphas_strong: np.ndarray | list[float],
    n_seeds: int = 3,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """
    Sweep the (alpha_weak, alpha_strong) plane.

    Args:
        alphas_weak: Weak-supervisor data budgets.
        alphas_strong: Strong-student data budgets.
        n_seeds: Repetitions per grid point.
        verbose: Print progress.
        **run_kwargs: Forwarded to `run_weak_to_strong`.

    Returns:
        Dict with axes and (len(a_w), len(a_s)) grids "m_weak",
        "m_strong", "m_ceiling", "pgr", "gain" (= m_strong - m_weak).

    """
    a_w = np.asarray(list(alphas_weak), dtype=float)
    a_s = np.asarray(list(alphas_strong), dtype=float)
    shape = (len(a_w), len(a_s))
    grids = {k: np.zeros(shape) for k in ("m_weak", "m_strong", "m_ceiling", "pgr", "gain")}

    for i, aw in enumerate(a_w):
        for j, as_ in enumerate(a_s):
            vals: dict[str, list[float]] = {k: [] for k in grids}
            for s in range(n_seeds):
                res = run_weak_to_strong(
                    alpha_weak=float(aw), alpha_strong=float(as_), seed=s, **run_kwargs
                )
                for k in ("m_weak", "m_strong", "m_ceiling", "pgr"):
                    vals[k].append(res[k])
                vals["gain"].append(res["m_strong"] - res["m_weak"])
            for k in grids:
                grids[k][i, j] = float(np.nanmean(vals[k]))
            if verbose:
                print(
                    f"a_w={aw:.1f} a_s={as_:.1f}: m_weak={grids['m_weak'][i, j]:.3f} "
                    f"m_strong={grids['m_strong'][i, j]:.3f} PGR={grids['pgr'][i, j]:+.2f}"
                )
    return {"alphas_weak": a_w, "alphas_strong": a_s, **grids}
