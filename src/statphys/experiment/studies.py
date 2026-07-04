"""
Ready-made statistical-physics studies for teacher-student setups.

Each study runs a replica-resolved numerical experiment, saves the raw
records as JSON, and renders a publication-style figure:

- committee      : specialization transition of a narrow tanh committee
- fss            : finite-size scaling of the L1 sparse-recovery transition
- diagram        : 2D recovery phase diagram (teacher sparsity x alpha)
- attention      : order parameters for an attention teacher-student pair
- manifold       : hidden-manifold inputs (realistic data structure)
- gpt            : tiny causal transformer (LLM-style) pair
- grokking       : delayed generalization in epoch time (large init +
                   weight decay; Power et al. 2022, Liu et al. 2022)
- universality   : Gaussian universality of the learning curve across
                   input distributions (and where it breaks)
- double_descent : model-wise double descent vs student width
                   (Belkin et al. 2019; Nakkiran et al. 2021)
- scaling        : generalization-error scaling exponents eps_g ~ alpha^-b
                   across architectures (neural-scaling-law style)

Use `run_study(name, out_dir, quick=...)` or the `statphys study` CLI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch.nn as nn

from statphys.experiment.protocol import TeacherStudentExperiment
from statphys.experiment.teacher import Teacher

__all__ = ["STUDIES", "run_study"]


def _mlp(d: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def _save(result_dict: dict, fig, out_dir: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result_dict, indent=2))
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    plt.close(fig)
    print(f"saved -> {out_dir}/{name}.json, {name}.png")


def _dashboard(res, title: str, extra_metrics: tuple[str, ...] = ()):
    from statphys.vis import plot_order_parameter_dashboard

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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

    from statphys.experiment.phase import run_phase_diagram

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
    import matplotlib.pyplot as plt

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


def study_grokking(out_dir: Path, quick: bool) -> None:
    """
    Delayed generalization (grokking) in epoch time.

    Large initialization + weight decay + a small training set: the
    student memorizes first (train error drops, test error stays high),
    then generalizes long afterwards. The teacher overlap m_hat tracks
    the delayed alignment; q_ab shows all replicas grokking to the same
    function.
    """
    import matplotlib.pyplot as plt

    d, k = (48, 4) if quick else (64, 6)
    alpha = 1.5
    epochs = 1500 if quick else 40000
    teacher = Teacher(_mlp(d, k), init="normal", noise_std=0.0)
    exp = TeacherStudentExperiment(teacher=teacher, student_factory=lambda: _mlp(d, k), d=d)
    res = exp.run_training_dynamics(
        alpha=alpha,
        n_replicas=2 if quick else 3,
        epochs=epochs,
        n_evals=25 if quick else 60,
        lr=1e-3,
        weight_decay=1e-2,
        init_scale=8.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(res.x_values)
    ax = axes[0]
    for name, color in (("train_error", "steelblue"), ("test_error", "crimson")):
        mean, std = res.mean(name), res.std(name)
        ax.plot(x, mean, "o-", markersize=3, color=color, label=name.replace("_", " "))
        ax.fill_between(x, np.maximum(mean - std, 1e-12), mean + std, alpha=0.2, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.plot(x, res.mean("m_hat"), "o-", markersize=3, label=r"$\hat m$")
    ax.plot(x, res.mean("q_ab_mean"), "d-", markersize=3, label=r"$q_{ab}$")
    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("order parameters")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"grokking: tanh MLP (d={d}, K={k}, alpha={alpha}, " "init x8, weight decay 1e-2)")
    fig.tight_layout()
    _save(res.to_dict(), fig, out_dir, "grokking")


def study_universality(out_dir: Path, quick: bool) -> None:
    """
    Gaussian universality of the learning curve — and where it breaks.

    Ridge-like linear regression on a linear teacher with i.i.d.-like
    input distributions (gaussian / rademacher / sphere) should give the
    *same* eps_g(alpha) curve in high dimension; correlated and
    hidden-manifold inputs break the collapse.
    """
    import matplotlib.pyplot as plt

    d = 64 if quick else 192
    alphas = [0.5, 1, 2, 4] if quick else [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8]
    dists: list[tuple[str, str, dict]] = [
        ("gaussian", "gaussian", {}),
        ("rademacher", "rademacher", {}),
        ("sphere", "sphere", {}),
        ("correlated (AR 0.8)", "correlated", {"ar_coeff": 0.8}),
        ("hidden manifold", "hidden_manifold", {"latent_dim": max(d // 8, 2)}),
    ]
    if quick:
        dists = dists[:3]

    results = {}
    for label, dist, kwargs in dists:
        teacher = Teacher(nn.Linear(d, 1, bias=False), init="normal", noise_std=0.1)
        exp = TeacherStudentExperiment(
            teacher=teacher,
            student_factory=lambda: nn.Linear(d, 1, bias=False),
            d=d,
            input_dist=dist,
            input_kwargs=kwargs,
        )
        res = exp.run_order_parameters(
            alphas=alphas,
            n_replicas=2 if quick else 4,
            share_data=False,
            lr=2e-2,
            max_epochs=200 if quick else 1500,
            weight_decay=1e-3,
        )
        results[label] = res

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, res in results.items():
        x = np.array(res.x_values)
        iid = label in ("gaussian", "rademacher", "sphere")
        style = {"linestyle": "-", "alpha": 0.9} if iid else {"linestyle": "--", "alpha": 0.9}
        axes[0].errorbar(
            x,
            res.mean("test_error"),
            yerr=res.std("test_error"),
            marker="o",
            markersize=4,
            capsize=2,
            label=label,
            **style,
        )
        axes[1].plot(x, res.mean("m_hat"), marker="o", markersize=4, label=label, **style)
    for ax, ylab in zip(axes, (r"$\epsilon_g$", r"$\hat m$"), strict=True):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha = n/d$")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.suptitle(
        f"Gaussian universality (linear teacher, d={d}): "
        "i.i.d.-like inputs collapse; structured inputs deviate"
    )
    fig.tight_layout()
    _save({k: r.to_dict() for k, r in results.items()}, fig, out_dir, "universality")


def study_double_descent(out_dir: Path, quick: bool) -> None:
    """
    Model-wise double descent vs student width.

    A narrow noisy teacher, students of increasing width trained without
    explicit regularization: eps_g peaks near the interpolation
    threshold and descends again in the overparameterized regime.
    """
    import matplotlib.pyplot as plt

    d, k_teacher, alpha = (32, 2, 3.0) if quick else (48, 4, 3.0)
    widths = [1, 2, 4, 8, 16] if quick else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    teacher = Teacher(_mlp(d, k_teacher), init="normal", noise_std=0.2)

    curves: dict[str, list[float]] = {m: [] for m in ("test_error", "train_gap", "m_hat", "q_ab")}
    stds: dict[str, list[float]] = {"test_error": []}
    raw = {}
    for k in widths:
        exp = TeacherStudentExperiment(teacher=teacher, student_factory=lambda k=k: _mlp(d, k), d=d)
        res = exp.run_order_parameters(
            alphas=[alpha],
            n_replicas=2 if quick else 5,
            share_data=True,
            lr=1e-2,
            max_epochs=400 if quick else 4000,
            weight_decay=0.0,
            verbose=False,
        )
        curves["test_error"].append(float(res.mean("test_error")[0]))
        stds["test_error"].append(float(res.std("test_error")[0]))
        curves["m_hat"].append(float(res.mean("m_hat")[0]))
        curves["q_ab"].append(float(res.mean("q_ab_mean")[0]))
        raw[k] = res.to_dict()
        print(f"width={k}: eps_g={curves['test_error'][-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    w = np.array(widths, dtype=float)
    ax = axes[0]
    ax.errorbar(
        w,
        curves["test_error"],
        yerr=stds["test_error"],
        fmt="o-",
        markersize=4,
        capsize=2,
        color="crimson",
    )
    ax.axvline(k_teacher, color="gray", linestyle=":", label=f"teacher K={k_teacher}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("student width $K_s$")
    ax.set_ylabel(r"$\epsilon_g$")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.plot(w, curves["m_hat"], "o-", markersize=4, label=r"$\hat m$")
    ax.plot(w, curves["q_ab"], "d-", markersize=4, label=r"$q_{ab}$")
    ax.set_xscale("log")
    ax.set_xlabel("student width $K_s$")
    ax.set_ylabel("order parameters")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"model-wise double descent (d={d}, teacher K={k_teacher}, "
        f"alpha={alpha}, no regularization, label noise 0.2)"
    )
    fig.tight_layout()
    _save({"widths": widths, "curves": curves, "raw": raw}, fig, out_dir, "double_descent")


def study_scaling(out_dir: Path, quick: bool) -> None:
    """
    Generalization-error scaling exponents across architectures.

    Fits eps_g ~ alpha^-b on the tail of the learning curve for several
    architectures — a teacher-student analogue of neural scaling laws
    (data-limited regime).
    """
    import matplotlib.pyplot as plt

    from statphys.experiment.zoo import architecture_experiment

    d = 64 if quick else 128
    alphas = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16, 32]
    archs: list[tuple[str, dict]] = [
        ("linear", {}),
        ("mlp", {"hidden": 8}),
        ("attention", {"seq_len": 8, "d_model": 8 if quick else 16}),
    ]

    results = {}
    for name, akw in archs:
        exp = architecture_experiment(
            name, d=d, teacher_init="normal", noise_std=0.05, arch_kwargs=akw
        )
        res = exp.run_order_parameters(
            alphas=alphas,
            n_replicas=2 if quick else 4,
            share_data=False,
            lr=1e-2,
            max_epochs=300 if quick else 3000,
            batch_size=512,
            weight_decay=1e-4,
            verbose=False,
        )
        results[name] = res
        print(f"{name}: eps_g(alpha_max)={res.mean('test_error')[-1]:.4g}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    exponents = {}
    for name, res in results.items():
        x = np.array(res.x_values)
        y = res.mean("test_error")
        (line,) = ax.plot(x, y, "o", markersize=5, label=None)
        # fit the tail (alpha >= 2) in log-log
        mask = x >= 2
        if mask.sum() >= 2 and np.all(y[mask] > 0):
            b, a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            exponents[name] = float(-b)
            xs = np.linspace(x[mask].min(), x.max(), 50)
            ax.plot(
                xs,
                np.exp(a) * xs**b,
                "--",
                color=line.get_color(),
                label=rf"{name}: $\epsilon_g \sim \alpha^{{-{-b:.2f}}}$",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"$\epsilon_g$")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(f"data-scaling exponents across architectures (d={d})")
    fig.tight_layout()
    _save(
        {"exponents": exponents, "results": {k: r.to_dict() for k, r in results.items()}},
        fig,
        out_dir,
        "scaling",
    )


STUDIES = {
    "committee": study_committee,
    "fss": study_fss,
    "diagram": study_diagram,
    "attention": study_attention,
    "manifold": study_manifold,
    "gpt": study_gpt,
    "grokking": study_grokking,
    "universality": study_universality,
    "double_descent": study_double_descent,
    "scaling": study_scaling,
}


def run_study(name: str, out_dir: str | Path = "phase_results", quick: bool = False) -> None:
    """
    Run a named study (or "all") and save JSON + PNG into out_dir.

    Args:
        name: Study name from STUDIES, or "all".
        out_dir: Output directory.
        quick: Use small smoke-test sizes.

    """
    names = sorted(STUDIES) if name == "all" else [name]
    out = Path(out_dir)
    for n in names:
        if n not in STUDIES:
            raise ValueError(f"Unknown study '{n}'. Choose from {sorted(STUDIES)}")
        print(f"=== study: {n} ===")
        t0 = time.time()
        STUDIES[n](out, quick)
        print(f"({n}: {time.time() - t0:.1f}s)")
