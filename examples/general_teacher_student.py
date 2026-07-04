"""
General teacher-student experiments for arbitrary models.

This example shows the theory-free experimental framework
(`statphys.experiment`): any PyTorch module can act as a teacher, with
customizable weight structure, and phase-transition-like phenomena are
measured purely numerically.

Run:
    python examples/general_teacher_student.py

"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from statphys.experiment import Teacher, TeacherStudentExperiment, get_preset
from statphys.utils.seed import fix_seed


def example_sparse_recovery():
    """Sparse linear teacher: recovery transition as alpha = n/d grows."""
    print("=" * 60)
    print("1. Sparse teacher recovery transition")
    print("=" * 60)

    exp = get_preset("sparse_teacher", d=200, sparsity=0.9, noise_std=0.05)
    result = exp.run_sample_complexity(
        alphas=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        n_seeds=3,
        max_epochs=500,
        verbose=False,
    )
    fig, _ = result.plot(logx=True, logy=True)
    fig.savefig("sparse_recovery.png", dpi=120)
    print("test error vs alpha:", np.round(result.mean("test_error"), 4))
    print("saved: sparse_recovery.png\n")


def example_custom_attention_teacher():
    """Custom setup: low-rank attention teacher (LLM-like toy)."""
    print("=" * 60)
    print("2. Low-rank attention teacher (no analytic theory exists)")
    print("=" * 60)

    exp = get_preset("low_rank_attention", d=128, seq_len=8, d_model=16, rank=2)
    result = exp.run_online(t_max=30.0, t_steps=20, n_seeds=2, lr=0.05, verbose=False)
    fig, _ = result.plot()
    fig.savefig("attention_online.png", dpi=120)
    print("final test error:", round(result.mean("test_error")[-1], 4))
    print("saved: attention_online.png\n")


def example_fully_custom():
    """Fully custom teacher/student/inputs/metrics."""
    print("=" * 60)
    print("3. Fully custom experiment")
    print("=" * 60)

    d = 100
    teacher = Teacher(
        nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1)),
        init="power_law",           # heavy-tailed teacher weights
        init_kwargs={"alpha": 3.0},
        readout="sign",             # binary classification
        flip_prob=0.05,             # 5% label noise
    )

    def weight_norm(student, dataset):
        return sum(p.norm().item() ** 2 for p in student.parameters())

    exp = TeacherStudentExperiment(
        teacher=teacher,
        student_factory=lambda: nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1)
        ),
        input_dist="correlated",
        input_kwargs={"ar_coeff": 0.5},
        metrics={"weight_norm": weight_norm},
    )

    result = exp.run_sample_complexity(
        alphas=[1.0, 2.0, 4.0, 8.0], n_seeds=2, max_epochs=300, verbose=False
    )
    print("0-1 test error vs alpha:", np.round(result.mean("test_error"), 4))
    fig, _ = result.plot(metrics=["test_error"])
    fig.savefig("custom_experiment.png", dpi=120)
    print("saved: custom_experiment.png\n")


if __name__ == "__main__":
    fix_seed(42)
    example_sparse_recovery()
    example_custom_attention_teacher()
    example_fully_custom()
    plt.close("all")
    print("All examples completed.")
