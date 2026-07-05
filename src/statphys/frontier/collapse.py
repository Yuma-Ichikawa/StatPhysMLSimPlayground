r"""
Model collapse under recursive synthetic-data training.

Setting (Shumailov et al. 2024, teacher-student core). Generation 0 is
trained on :math:`\alpha d` true teacher labels. Every later generation
:math:`g+1` is trained on labels produced by generation :math:`g` on
fresh inputs, with a fraction :math:`p_{\rm real}` of the batch drawn
from the *true* teacher (anchoring / accumulated real data):

.. math::
    y_i = \begin{cases}
        f^*(x_i) + \xi & \text{w.p. } p_{\rm real} \\
        f_{g}(x_i) & \text{otherwise.}
    \end{cases}

Order parameters, per generation :math:`g`:

- :math:`m(g)` -- function overlap of generation :math:`g` with the
  true teacher (magnetization of the "signal")
- variance ratio :math:`q(g)/q(0)` with :math:`q = \mathbb E[f_g^2]`
  -- collapse is accompanied by shrinking output variance (loss of
  distribution tails), the standard collapse signature

Phenomenology probed. For :math:`p_{\rm real} = 0` the overlap should
decay over generations (error accumulation ~ a random walk of the
student function away from the teacher); a modest real-data fraction
should stabilize it. The interesting object is the "phase boundary" in
the :math:`(p_{\rm real}, g)` plane and the terminal overlap
:math:`m(g_{\max})` as a function of :math:`p_{\rm real}` and
:math:`\alpha`.
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

__all__ = ["run_collapse", "sweep_collapse"]


def run_collapse(
    d: int = 64,
    hidden: int = 16,
    alpha: float = 8.0,
    n_generations: int = 10,
    p_real: float = 0.0,
    lr: float = 5e-3,
    epochs: int = 2000,
    noise_std: float = 0.1,
    n_probe: int = 4096,
    seed: int = 0,
    teacher: Teacher | None = None,
    input_sampler: InputSampler | None = None,
) -> dict:
    """
    Iterate student generations trained on the previous generation's labels.

    Args:
        d: Input dimension.
        hidden: Hidden width (teacher and all student generations).
        alpha: Per-generation sample ratio n / d.
        n_generations: Number of synthetic-data generations after gen 0.
        p_real: Fraction of each generation's labels drawn from the
            true teacher (0 = fully synthetic loop).
        lr: Adam learning rate.
        epochs: Max epochs per generation.
        noise_std: Label noise on *real* labels.
        n_probe: Probe-set size.
        seed: Random seed.
        teacher: Optional true teacher (taxonomy injection); defaults
            to a random-weight tanh MLP.
        input_sampler: Optional input distribution n -> (n, d).

    Returns:
        Dict with per-generation arrays "m" (overlap with truth,
        length n_generations+1), "q_ratio" (output variance / gen-0
        variance), and the config.

    """
    fix_seed(seed)
    truth = (
        teacher
        if teacher is not None
        else Teacher(mlp(d, hidden), init="normal", noise_std=noise_std)
    )
    sample = input_sampler if input_sampler is not None else gaussian_sampler(d)
    X_probe = sample(n_probe)
    n = max(int(alpha * d), 1)

    # generation 0: trained on real data
    X0 = sample(n)
    current = mlp(d, hidden)
    train_regression(current, X0, truth(X0), lr=lr, epochs=epochs)

    with torch.no_grad():
        q0 = float(current(X_probe).squeeze(-1).var().clamp_min(1e-12))
    m_traj = [model_overlap(current, truth, X_probe)]
    q_traj = [1.0]

    for g in range(n_generations):
        fix_seed(seed + 100 + g)
        X = sample(n)
        with torch.no_grad():
            y_syn = current(X).squeeze(-1)
        y_real = truth(X)
        real_mask = torch.rand(n) < p_real
        y = torch.where(real_mask, y_real, y_syn)

        nxt = mlp(d, hidden)
        train_regression(nxt, X, y, lr=lr, epochs=epochs)
        current = nxt

        with torch.no_grad():
            q_g = float(current(X_probe).squeeze(-1).var())
        m_traj.append(model_overlap(current, truth, X_probe))
        q_traj.append(q_g / q0)

    return {
        "generations": np.arange(n_generations + 1),
        "m": np.asarray(m_traj),
        "q_ratio": np.asarray(q_traj),
        "config": {
            "d": d,
            "hidden": hidden,
            "alpha": alpha,
            "n_generations": n_generations,
            "p_real": p_real,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


def sweep_collapse(
    p_reals: np.ndarray | list[float],
    n_seeds: int = 3,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """
    Sweep the real-data fraction p_real.

    Args:
        p_reals: Real-data fractions to sweep.
        n_seeds: Repetitions per value.
        verbose: Print progress.
        **run_kwargs: Forwarded to `run_collapse`.

    Returns:
        Dict with "p_reals", "generations", seed-averaged trajectories
        "m_curves" and "q_curves" of shape (len(p), n_gen+1), and the
        terminal overlaps "m_final" (+ std).

    """
    ps = np.asarray(list(p_reals), dtype=float)
    m_curves, q_curves, m_final, m_final_std = [], [], [], []
    gens = None
    for p in ps:
        ms, qs = [], []
        for s in range(n_seeds):
            res = run_collapse(p_real=float(p), seed=s, **run_kwargs)
            ms.append(res["m"])
            qs.append(res["q_ratio"])
            gens = res["generations"]
        m_curves.append(np.mean(ms, axis=0))
        q_curves.append(np.mean(qs, axis=0))
        finals = [m[-1] for m in ms]
        m_final.append(float(np.mean(finals)))
        m_final_std.append(float(np.std(finals)))
        if verbose:
            print(f"p_real={p:.2f}: m_final={m_final[-1]:.3f} q_final={q_curves[-1][-1]:.3f}")
    return {
        "p_reals": ps,
        "generations": gens,
        "m_curves": np.asarray(m_curves),
        "q_curves": np.asarray(q_curves),
        "m_final": np.asarray(m_final),
        "m_final_std": np.asarray(m_final_std),
    }
