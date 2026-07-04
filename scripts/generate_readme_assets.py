"""
Generate the animated GIFs embedded in the top-level README.

Produces (into assets/ by default):

- anim_learning_curve.gif : online SGD order parameters vs exact ODE theory
- anim_phase_plane.gif    : (m, q) trajectory moving over the ODE flow field
- anim_specialization.gif : student-teacher overlap matrix of a soft committee
                            machine specializing during online SGD

Usage:
    python scripts/generate_readme_assets.py [--out-dir assets] [--fps 20]

The script only depends on the installed `statphys` package and writes
small (dpi ~80) GIFs suitable for embedding in a repository README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.special import erf

import statphys
from statphys.theory.online import GaussianLinearMseEquations, ODESolver
from statphys.vis.animation import (
    animate_learning_curve,
    animate_overlap_matrix,
    animate_phase_plane,
    save_animation,
)

RHO = 1.0
LR = 0.5
T_MAX = 12.0
N_FRAMES = 80


def _online_linear_sgd(
    d: int, lr: float, t_max: float, n_record: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-run online SGD for linear regression; returns (t, m, q)."""
    w0 = rng.standard_normal(d)
    w0 *= np.sqrt(RHO * d) / np.linalg.norm(w0)
    w = 0.1 * rng.standard_normal(d)

    n_steps = int(t_max * d)
    record_every = max(1, n_steps // n_record)
    ts, ms, qs = [], [], []
    for step in range(n_steps + 1):
        if step % record_every == 0:
            ts.append(step / d)
            ms.append(w @ w0 / d)
            qs.append(w @ w / d)
        x = rng.standard_normal(d)
        y = w0 @ x / np.sqrt(d)
        y_hat = w @ x / np.sqrt(d)
        # MSE loss 0.5*(y - y_hat)^2, gradient step with 1/sqrt(d) input scaling
        w += lr * (y - y_hat) * x / np.sqrt(d)
    return np.asarray(ts), np.asarray(ms), np.asarray(qs)


def make_learning_curve_gif(out: Path, fps: int) -> None:
    """Order parameters being drawn over time, with theory as dashed lines."""
    rng = np.random.default_rng(0)
    t_exp, m_exp, q_exp = _online_linear_sgd(400, LR, T_MAX, N_FRAMES, rng)

    solver = ODESolver(
        equations=GaussianLinearMseEquations(rho=RHO, lr=LR),
        order_params=["m", "q"],
    )
    theory = solver.solve(
        (0.0, T_MAX),
        init_values=(float(m_exp[0]), float(q_exp[0])),
        t_eval=t_exp,
    )

    anim = animate_learning_curve(
        t_exp,
        {"$m$ (experiment)": m_exp, "$q$ (experiment)": q_exp},
        theory={
            "$m$ (experiment)": theory["m"],
            "$q$ (experiment)": theory["q"],
        },
        figsize=(6, 4),
    )
    save_animation(anim, str(out), fps=fps, dpi=80)


def make_phase_plane_gif(out: Path, fps: int) -> None:
    """A trajectory moving through the (m, q) plane over the flow field."""
    eqs = GaussianLinearMseEquations(rho=RHO, lr=LR)
    solver = ODESolver(equations=eqs, order_params=["m", "q"])
    t_eval = np.linspace(0, T_MAX, N_FRAMES)
    theory = solver.solve((0.0, T_MAX), init_values=(0.0, 0.01), t_eval=t_eval)

    anim = animate_phase_plane(
        theory["m"],
        theory["q"],
        equations=eqs,
        m_range=(-0.05, 1.15),
        q_range=(0.01, 1.3),
        figsize=(5.5, 4.5),
    )
    save_animation(anim, str(out), fps=fps, dpi=80)


def make_specialization_gif(out: Path, fps: int) -> None:
    """Soft committee machine (erf) specializing: M matrix over training."""
    rng = np.random.default_rng(1)
    d, k = 300, 3
    lr = 1.0

    w_teacher = rng.standard_normal((k, d))
    w_teacher /= np.linalg.norm(w_teacher, axis=1, keepdims=True) / np.sqrt(d)
    w = 0.05 * rng.standard_normal((k, d))

    t_max, n_snap = 120.0, 60
    n_steps = int(t_max * d)
    snap_every = n_steps // n_snap

    def g(h: np.ndarray) -> np.ndarray:
        return erf(h / np.sqrt(2.0))

    def g_prime(h: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / np.pi) * np.exp(-0.5 * h**2)

    snaps, t_vals = [], []
    for step in range(n_steps):
        if step % snap_every == 0:
            snaps.append(w @ w_teacher.T / d)
            t_vals.append(step / d)
        x = rng.standard_normal(d)
        h_s = w @ x / np.sqrt(d)
        h_t = w_teacher @ x / np.sqrt(d)
        delta = g(h_t).sum() - g(h_s).sum()
        w += lr * delta * np.outer(g_prime(h_s), x) / np.sqrt(d)

    anim = animate_overlap_matrix(
        snaps,
        t_values=np.asarray(t_vals),
        figsize=(4.8, 4.2),
        title=r"$M_{km} = \mathbf{w}_k \cdot \mathbf{w}^0_m / d$",
    )
    save_animation(anim, str(out), fps=max(fps // 2, 5), dpi=80)


def main() -> None:
    """Generate all README GIF assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="assets", help="output directory")
    parser.add_argument("--fps", type=int, default=20, help="frames per second")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    statphys.fix_seed(0)

    print("[1/3] learning curve animation ...")
    make_learning_curve_gif(out_dir / "anim_learning_curve.gif", args.fps)
    print("[2/3] phase plane animation ...")
    make_phase_plane_gif(out_dir / "anim_phase_plane.gif", args.fps)
    print("[3/3] specialization animation ...")
    make_specialization_gif(out_dir / "anim_specialization.gif", args.fps)
    print(f"done -> {out_dir}/")


if __name__ == "__main__":
    main()
