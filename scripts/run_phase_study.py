"""
Statistical-physics phase-transition studies for teacher-student setups.

Runs replica-resolved numerical experiments and records function-space
order parameters (magnetization m_hat, replica overlap q_ab,
susceptibility chi, Binder cumulant, specialization index):

- committee : specialization transition of a narrow tanh committee
- fss       : finite-size scaling of the sparse-recovery transition
              (m_hat curves sharpen with d; Binder curves cross at alpha_c)
- diagram   : 2D recovery phase diagram (teacher sparsity x alpha)
- attention : order parameters across the transition for an attention
              teacher-student pair (no analytic theory available)
- manifold  : hidden-manifold inputs (Goldt et al. 2020) — how realistic
              low-dimensional data structure shifts the transition and
              the generalization error compared to isotropic inputs
- gpt       : tiny causal transformer (LLM-style) teacher-student pair;
              order parameters measured purely numerically

Usage:
    python scripts/run_phase_study.py --study all --output-dir phase_results
    python scripts/run_phase_study.py --study fss --quick

Outputs one JSON (raw records) and one PNG (order-parameter dashboard)
per study.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from statphys.experiment import (
    Teacher,
    TeacherStudentExperiment,
    run_phase_diagram,
)
from statphys.vis import plot_order_parameter_dashboard


def _mlp(d: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def _save(result_dict: dict, fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result_dict, indent=2))
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    plt.close(fig)
    print(f"saved -> {out_dir}/{name}.json, {name}.png")


def _dashboard(res, title: str, extra_metrics: tuple[str, ...] = ()):
    fig, _ = plot_order_parameter_dashboard(res, title=title, extra_metrics=extra_metrics)
    return fig


def study_committee(out_dir: Path, quick: bool) -> None:
    """Specialization transition in a narrow tanh committee machine."""
    d, k = (64, 3) if quick else (128, 4)
    alphas = [0.5, 1, 2, 4, 8] if quick else [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    teacher = Teacher(_mlp(d, k), init="normal", noise_std=0.01)
    exp = TeacherStudentExperiment(teacher=teacher, student_factory=lambda: _mlp(d, k), d=d)
    res = exp.run_order_parameters(
        alphas=alphas,
        n_replicas=2 if quick else 4,
        lr=5e-3,
        max_epochs=300 if quick else 3000,
        weight_decay=1e-4,
    )
    fig = _dashboard(
        res,
        f"tanh committee (d={d}, K={k}): specialization transition",
        extra_metrics=("specialization",),
    )
    _save(res.to_dict(), fig, out_dir, "committee")


def study_fss(out_dir: Path, quick: bool) -> None:
    """Finite-size scaling of the L1 sparse-recovery transition."""
    dims = [64, 128] if quick else [64, 128, 256, 512]
    alphas = (
        [0.1, 0.2, 0.3, 0.5, 1.0]
        if quick
        else [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]
    )
    sparsity, results = 0.9, {}
    for d in dims:
        teacher = Teacher(
            nn.Linear(d, 1, bias=False),
            init="sparse",
            init_kwargs={"sparsity": sparsity},
            noise_std=0.05,
        )
        exp = TeacherStudentExperiment(
            teacher=teacher, student_factory=lambda d=d: nn.Linear(d, 1, bias=False), d=d
        )
        res = exp.run_order_parameters(
            alphas=alphas,
            n_replicas=3 if quick else 6,
            share_data=False,
            lr=2e-2,
            max_epochs=300 if quick else 2000,
            l1_penalty=2e-3,
        )
        results[d] = res

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for d, res in results.items():
        x = np.array(res.x_values)
        axes[0].errorbar(
            x,
            res.mean("m_hat"),
            yerr=res.std("m_hat"),
            fmt="o-",
            markersize=4,
            capsize=2,
            label=f"d={d}",
        )
        axes[1].plot(x, res.mean("chi_m"), "^-", markersize=4, label=f"d={d}")
        axes[2].plot(x, res.mean("binder_m"), "d-", markersize=4, label=f"d={d}")
    for ax, ylab in zip(axes, (r"$\hat m$", r"$\chi_m$", r"Binder $U_4$"), strict=True):
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    fig.suptitle(f"sparse linear teacher (sparsity={sparsity}): finite-size scaling")
    fig.tight_layout()
    _save({str(d): r.to_dict() for d, r in results.items()}, fig, out_dir, "fss")


def study_diagram(out_dir: Path, quick: bool) -> None:
    """2D recovery diagram: teacher sparsity x sample ratio."""
    d = 96 if quick else 192
    sparsities = [0.5, 0.8, 0.95] if quick else [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98]
    alphas = [0.25, 0.5, 1, 2] if quick else [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]

    def factory(sparsity: float) -> TeacherStudentExperiment:
        teacher = Teacher(
            nn.Linear(d, 1, bias=False),
            init="sparse",
            init_kwargs={"sparsity": sparsity},
            noise_std=0.05,
        )
        return TeacherStudentExperiment(
            teacher=teacher, student_factory=lambda: nn.Linear(d, 1, bias=False), d=d
        )

    res = run_phase_diagram(
        factory,
        param_name="sparsity",
        param_values=sparsities,
        alphas=alphas,
        n_replicas=2 if quick else 3,
        lr=2e-2,
        max_epochs=300 if quick else 1500,
        l1_penalty=2e-3,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    res.plot("m_hat", ax=axes[0], logx=True, contour_level=0.5)
    axes[0].set_title(r"$\hat m$ (teacher recovery)")
    res.plot("chi_m", ax=axes[1], logx=True, cmap="magma")
    axes[1].set_title(r"$\chi_m$ (susceptibility ridge $\approx$ boundary)")
    fig.suptitle(f"sparse recovery phase diagram (d={d})")
    fig.tight_layout()
    _save(res.to_dict(), fig, out_dir, "diagram")


def study_attention(out_dir: Path, quick: bool) -> None:
    """Order parameters across the transition for an attention pair."""
    from statphys.experiment.presets import low_rank_attention

    d = 64 if quick else 128
    exp = low_rank_attention(d=d, seq_len=8, d_model=8 if quick else 16, rank=1)
    alphas = [1, 4, 16] if quick else [0.5, 1, 2, 4, 8, 16, 32]
    res = exp.run_order_parameters(
        alphas=alphas,
        n_replicas=2 if quick else 4,
        lr=3e-3,
        max_epochs=300 if quick else 2500,
        weight_decay=1e-4,
    )
    fig = _dashboard(res, f"low-rank attention teacher (d={d}): numerical transition")
    _save(res.to_dict(), fig, out_dir, "attention")


def study_manifold(out_dir: Path, quick: bool) -> None:
    """Hidden-manifold inputs: realistic data structure vs isotropic."""
    from statphys.experiment.presets import hidden_manifold

    d = 96 if quick else 192
    hidden = 8
    latent_dims = [8, 48] if quick else [8, 24, 96, 192]
    alphas = [0.5, 1, 2, 4] if quick else [0.25, 0.5, 1, 2, 4, 8, 16]

    results = {}
    for dl in latent_dims:
        exp = hidden_manifold(d=d, latent_dim=dl, hidden=hidden, noise_std=0.01)
        res = exp.run_order_parameters(
            alphas=alphas,
            n_replicas=2 if quick else 4,
            lr=5e-3,
            max_epochs=300 if quick else 3000,
            weight_decay=1e-4,
        )
        results[dl] = res

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for dl, res in results.items():
        x = np.array(res.x_values)
        label = f"$D_{{lat}}={dl}$"
        axes[0].errorbar(
            x,
            res.mean("m_hat"),
            yerr=res.std("m_hat"),
            fmt="o-",
            markersize=4,
            capsize=2,
            label=label,
        )
        axes[1].plot(x, res.mean("test_error"), "s-", markersize=4, label=label)
        axes[2].plot(x, res.mean("q_ab_mean"), "d-", markersize=4, label=label)
    for i, (ax, ylab) in enumerate(
        zip(axes, (r"$\hat m$", r"$\epsilon_g$", r"$q_{ab}$"), strict=True)
    ):
        ax.set_xscale("log")
        if i == 1:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha = n/d$")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"hidden manifold inputs (d={d}, tanh, MLP K={hidden}): "
        "data structure shifts the transition"
    )
    fig.tight_layout()
    _save({str(k): r.to_dict() for k, r in results.items()}, fig, out_dir, "manifold")


def study_gpt(out_dir: Path, quick: bool) -> None:
    """Tiny causal transformer (LLM-style) teacher-student order parameters."""
    from statphys.experiment.presets import tiny_gpt

    d = 64 if quick else 128
    exp = tiny_gpt(
        d=d,
        seq_len=8,
        d_model=8 if quick else 16,
        n_heads=2,
        n_blocks=1,
        noise_std=0.01,
    )
    alphas = [1, 4, 16] if quick else [0.5, 1, 2, 4, 8, 16, 32]
    res = exp.run_order_parameters(
        alphas=alphas,
        n_replicas=2 if quick else 3,
        lr=2e-3,
        max_epochs=300 if quick else 800,
        batch_size=512,
        weight_decay=1e-4,
    )
    fig = _dashboard(res, f"tiny GPT teacher-student (d={d}): numerical order parameters")
    _save(res.to_dict(), fig, out_dir, "gpt")


STUDIES = {
    "committee": study_committee,
    "fss": study_fss,
    "diagram": study_diagram,
    "attention": study_attention,
    "manifold": study_manifold,
    "gpt": study_gpt,
}


def main() -> None:
    """Run the selected phase-transition studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="all", help=f"{sorted(STUDIES)} or 'all'")
    parser.add_argument("--output-dir", default="phase_results")
    parser.add_argument("--quick", action="store_true", help="small smoke-test sizes")
    args = parser.parse_args()

    names = sorted(STUDIES) if args.study == "all" else [args.study]
    out_dir = Path(args.output_dir)
    for name in names:
        if name not in STUDIES:
            raise SystemExit(f"Unknown study '{name}'. Choose from {sorted(STUDIES)}")
        print(f"=== study: {name} ===")
        t0 = time.time()
        STUDIES[name](out_dir, args.quick)
        print(f"({name}: {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
