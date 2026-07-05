r"""
Supervised fine-tuning (SFT) as a two-teacher-student problem.

Setting. A student is first pretrained on task A (teacher :math:`f_A`)
and then fine-tuned on task B (teacher :math:`f_B`), where the two
teachers share the same nonlinear architecture and their weights have
controlled cosine similarity :math:`\rho` (see
`statphys.frontier.common.correlated_teacher`):

.. math::
    W_B = \rho\, W_A + \sqrt{1-\rho^2}\, W_\perp .

Order parameters (function space, probe-set averages):

- :math:`m_A(t)` -- overlap of the student with teacher A during
  fine-tuning; its decay quantifies *catastrophic forgetting*
- :math:`m_B(t)` -- overlap with teacher B (fine-tuning progress)
- forgetting :math:`F = m_A(0^+) - m_A(\infty)` (drop from the
  pretrained value over fine-tuning)
- transfer gain :math:`\Delta_B = m_B^{\text{pretrained}} -
  m_B^{\text{scratch}}` at equal fine-tuning budget; its sign boundary
  in the :math:`(\rho, \alpha_{\rm ft})` plane is the positive /
  negative transfer "phase boundary"

Phenomenology probed. Catastrophic forgetting should switch on sharply
as task similarity decreases and the fine-tuning sample budget
:math:`\alpha_{\rm ft} = n_{\rm ft}/d` grows; transfer should change
sign from positive (related tasks) to negative (interference).
"""

from __future__ import annotations

import numpy as np

from statphys.experiment.teacher import Teacher
from statphys.frontier.common import (
    InputSampler,
    correlated_teacher,
    gaussian_sampler,
    mlp,
    model_overlap,
    train_regression,
)
from statphys.utils.seed import fix_seed

__all__ = ["run_finetune", "sweep_sft_phase_diagram"]


def _make_teacher(d: int, hidden: int, noise_std: float, seed: int) -> Teacher:
    fix_seed(seed)
    return Teacher(mlp(d, hidden), init="normal", noise_std=noise_std)


def run_finetune(
    d: int = 64,
    hidden: int = 16,
    similarity: float = 0.5,
    alpha_pre: float = 8.0,
    alpha_ft: float = 4.0,
    n_checkpoints: int = 20,
    epochs_per_checkpoint: int = 100,
    pretrain_epochs: int = 2000,
    lr: float = 5e-3,
    noise_std: float = 0.1,
    n_probe: int = 4096,
    seed: int = 0,
    compare_scratch: bool = True,
    teacher: Teacher | None = None,
    input_sampler: InputSampler | None = None,
) -> dict:
    """
    Pretrain on task A, fine-tune on task B, and track both overlaps.

    Args:
        d: Input dimension.
        hidden: Hidden width of teachers and student.
        similarity: Weight-space cosine between teacher A and B.
        alpha_pre: Pretraining sample ratio n_pre / d.
        alpha_ft: Fine-tuning sample ratio n_ft / d.
        n_checkpoints: Number of measurement points during fine-tuning.
        epochs_per_checkpoint: Adam epochs between measurements.
        pretrain_epochs: Epochs for the pretraining phase.
        lr: Adam learning rate (both phases).
        noise_std: Label noise on both tasks.
        n_probe: Probe-set size for function-space overlaps.
        seed: Random seed.
        compare_scratch: Also train a from-scratch student on task B
            with the same budget (for the transfer gain).
        teacher: Optional task-A teacher (e.g. from the taxonomy in
            `statphys.frontier.teachers`). Defaults to a random-weight
            tanh MLP. Task B is always the weight-space interpolation
            of this teacher.
        input_sampler: Optional input distribution n -> (n, d).
            Defaults to isotropic Gaussian.

    Returns:
        Dict with checkpoint trajectories "m_A", "m_B", "epochs", the
        scalars "m_A_pre", "m_B_pre", "forgetting", "transfer_gain",
        "m_B_scratch", and the config.

    """
    teacher_a = teacher if teacher is not None else _make_teacher(d, hidden, noise_std, seed)
    sample = input_sampler if input_sampler is not None else gaussian_sampler(d)
    teacher_b = correlated_teacher(teacher_a, similarity, seed=seed + 1)
    fix_seed(seed + 2)
    X_probe = sample(n_probe)

    # --- phase 1: pretrain on task A ---
    n_pre = max(int(alpha_pre * d), 1)
    X_pre = sample(n_pre)
    y_pre = teacher_a(X_pre)
    student = mlp(d, hidden)
    train_regression(student, X_pre, y_pre, lr=lr, epochs=pretrain_epochs)
    m_a_pre = model_overlap(student, teacher_a, X_probe)
    m_b_pre = model_overlap(student, teacher_b, X_probe)

    # --- phase 2: fine-tune on task B, tracking both overlaps ---
    n_ft = max(int(alpha_ft * d), 1)
    X_ft = sample(n_ft)
    y_ft = teacher_b(X_ft)
    m_a_traj, m_b_traj, epochs = [m_a_pre], [m_b_pre], [0]
    for c in range(n_checkpoints):
        train_regression(student, X_ft, y_ft, lr=lr, epochs=epochs_per_checkpoint, patience=10**9)
        m_a_traj.append(model_overlap(student, teacher_a, X_probe))
        m_b_traj.append(model_overlap(student, teacher_b, X_probe))
        epochs.append((c + 1) * epochs_per_checkpoint)

    m_b_scratch = float("nan")
    if compare_scratch:
        fix_seed(seed + 3)
        scratch = mlp(d, hidden)
        total_ft_epochs = n_checkpoints * epochs_per_checkpoint
        train_regression(scratch, X_ft, y_ft, lr=lr, epochs=total_ft_epochs, patience=10**9)
        m_b_scratch = model_overlap(scratch, teacher_b, X_probe)

    return {
        "epochs": np.asarray(epochs),
        "m_A": np.asarray(m_a_traj),
        "m_B": np.asarray(m_b_traj),
        "m_A_pre": m_a_pre,
        "m_B_pre": m_b_pre,
        "forgetting": m_a_pre - m_a_traj[-1],
        "m_B_scratch": m_b_scratch,
        "transfer_gain": m_b_traj[-1] - m_b_scratch,
        "config": {
            "d": d,
            "hidden": hidden,
            "similarity": similarity,
            "alpha_pre": alpha_pre,
            "alpha_ft": alpha_ft,
            "lr": lr,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


def sweep_sft_phase_diagram(
    similarities: np.ndarray | list[float],
    alphas_ft: np.ndarray | list[float],
    n_seeds: int = 3,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """
    Sweep the (task similarity, fine-tuning data) plane.

    For each grid point, `run_finetune` is repeated over seeds and the
    seed-averaged forgetting F and transfer gain Delta_B are collected.

    Args:
        similarities: Values of the teacher-teacher cosine rho.
        alphas_ft: Fine-tuning sample ratios n_ft / d.
        n_seeds: Independent repetitions per grid point.
        verbose: Print progress.
        **run_kwargs: Forwarded to `run_finetune`.

    Returns:
        Dict with "similarities", "alphas_ft", and (len(sim), len(alpha))
        grids "forgetting", "transfer_gain", "m_A_final", "m_B_final".

    """
    sims = np.asarray(list(similarities), dtype=float)
    alphas = np.asarray(list(alphas_ft), dtype=float)
    shape = (len(sims), len(alphas))
    grids = {k: np.zeros(shape) for k in ("forgetting", "transfer_gain", "m_A_final", "m_B_final")}

    for i, rho in enumerate(sims):
        for j, a_ft in enumerate(alphas):
            vals: dict[str, list[float]] = {k: [] for k in grids}
            for s in range(n_seeds):
                res = run_finetune(
                    similarity=float(rho), alpha_ft=float(a_ft), seed=s, **run_kwargs
                )
                vals["forgetting"].append(res["forgetting"])
                vals["transfer_gain"].append(res["transfer_gain"])
                vals["m_A_final"].append(res["m_A"][-1])
                vals["m_B_final"].append(res["m_B"][-1])
            for k in grids:
                grids[k][i, j] = float(np.mean(vals[k]))
            if verbose:
                print(
                    f"rho={rho:.2f} alpha_ft={a_ft:.1f}: "
                    f"F={grids['forgetting'][i, j]:+.3f} "
                    f"dB={grids['transfer_gain'][i, j]:+.3f}"
                )
    return {"similarities": sims, "alphas_ft": alphas, **grids}
