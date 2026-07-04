"""
Animation utilities for learning dynamics.

Creates GIF/MP4 animations from simulation results:

- animate_learning_curve: order parameters (and/or generalization error)
  growing over time, with optional log axes
- animate_phase_plane: a point moving through the (m, q) plane,
  optionally on top of the ODE flow field
- animate_overlap_matrix: the M overlap matrix evolving during training
  (specialization becoming visible as an emerging diagonal)
- animate_decision_boundary: a 2D classifier's decision line rotating
  into place as it is trained, next to the (fixed) labelled data —
  the most intuitive, "textbook-figure" animation for teaching what
  generalization/recovery means
- animate_curve_and_matrix: a learning curve and an overlap matrix
  evolving side by side — the standard two-panel view of plateau
  escape / specialization in committee machines

Saving uses matplotlib writers: "pillow" (GIF, always available with
Pillow) or "ffmpeg" (MP4, requires ffmpeg on PATH).

Example:
    >>> from statphys.vis.animation import animate_phase_plane
    >>> anim = animate_phase_plane(m_traj, q_traj, equations=eqs)
    >>> anim.save("dynamics.gif", writer="pillow", fps=20)

"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from statphys.vis.plotter import PlotStyle


def _style() -> PlotStyle:
    style = PlotStyle()
    style.apply()
    return style


def animate_learning_curve(
    t_values: np.ndarray,
    trajectories: dict[str, np.ndarray],
    theory: dict[str, np.ndarray] | None = None,
    interval: int = 50,
    figsize: tuple[float, float] = (7, 5),
    xlabel: str = r"$t = \tau/d$",
    ylabel: str = "order parameters",
    logx: bool = False,
    logy: bool = False,
) -> FuncAnimation:
    """
    Animate order-parameter (or error) curves being drawn over time.

    Args:
        t_values: Time grid of shape (T,).
        trajectories: Mapping from name to array of shape (T,). Any
            quantity works, e.g. {"m": ..., "q": ..., "eps_g": ...} or
            {"train_error": ..., "test_error": ...} for grokking-style
            plots (use logx=True, logy=True for the latter).
        theory: Optional theory curves drawn as static dashed lines.
        interval: Milliseconds between frames.
        figsize: Figure size.
        xlabel, ylabel: Axis labels.
        logx, logy: Use log-scaled axes (e.g. for epoch-resolved error
            curves spanning several decades, as in grokking plots).

    Returns:
        FuncAnimation (call .save(path, writer="pillow") to export).

    """
    style = _style()
    fig, ax = plt.subplots(figsize=figsize)

    t = np.asarray(t_values)
    names = list(trajectories.keys())
    data = {k: np.asarray(v) for k, v in trajectories.items()}

    lines = {}
    dots = {}
    for i, name in enumerate(names):
        color = style.colors[i % len(style.colors)]
        (lines[name],) = ax.plot([], [], color=color, linewidth=2, label=name)
        (dots[name],) = ax.plot([], [], "o", color=color, markersize=7)
        if theory and name in theory:
            ax.plot(t, theory[name], color=color, linestyle="--", linewidth=1, alpha=0.7)

    all_vals = np.concatenate(list(data.values()))
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(max(t.min(), 1e-3), t.max())
    else:
        ax.set_xlim(t.min(), t.max())
    if logy:
        ax.set_yscale("log")
        positive = all_vals[all_vals > 0]
        lo = positive.min() * 0.5 if positive.size else 1e-3
        ax.set_ylim(lo, all_vals.max() * 2)
    else:
        pad = 0.1 * (all_vals.max() - all_vals.min() + 1e-12)
        ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    def update(frame: int):
        artists = []
        for name in names:
            lines[name].set_data(t[: frame + 1], data[name][: frame + 1])
            dots[name].set_data([t[frame]], [data[name][frame]])
            artists += [lines[name], dots[name]]
        return artists

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=True)


def animate_phase_plane(
    m_traj: np.ndarray,
    q_traj: np.ndarray,
    equations: Any | None = None,
    params: dict[str, Any] | None = None,
    m_range: tuple[float, float] | None = None,
    q_range: tuple[float, float] | None = None,
    n_grid: int = 18,
    interval: int = 50,
    trail: bool = True,
    figsize: tuple[float, float] = (6.5, 5.5),
) -> FuncAnimation:
    """
    Animate a trajectory moving through the (m, q) plane.

    Args:
        m_traj: m values over time, shape (T,).
        q_traj: q values over time, shape (T,).
        equations: Optional online ODE equations f(t, [m, q], params);
            if given, the flow field is drawn as a static background.
        params: Parameter overrides for the equations.
        m_range, q_range: Plot ranges; inferred from trajectory if None.
        n_grid: Flow-field grid resolution.
        interval: Milliseconds between frames.
        trail: Draw the path travelled so far.
        figsize: Figure size.

    Returns:
        FuncAnimation.

    """
    style = _style()
    fig, ax = plt.subplots(figsize=figsize)

    m_traj = np.asarray(m_traj)
    q_traj = np.asarray(q_traj)

    if m_range is None:
        pad = 0.15 * (m_traj.max() - m_traj.min() + 1e-6)
        m_range = (m_traj.min() - pad, m_traj.max() + pad)
    if q_range is None:
        pad = 0.15 * (q_traj.max() - q_traj.min() + 1e-6)
        q_range = (max(q_traj.min() - pad, 1e-3), q_traj.max() + pad)

    if equations is not None:
        params = params or {}
        m_grid = np.linspace(m_range[0], m_range[1], n_grid)
        q_grid = np.linspace(q_range[0], q_range[1], n_grid)
        M, Q = np.meshgrid(m_grid, q_grid)
        dM = np.full_like(M, np.nan)
        dQ = np.full_like(Q, np.nan)
        for idx in np.ndindex(M.shape):
            try:
                d = equations(0.0, np.array([M[idx], Q[idx]]), params)
                dM[idx], dQ[idx] = d[0], d[1]
            except Exception:
                pass
        ax.streamplot(M, Q, dM, dQ, color="lightgray", density=1.1, linewidth=0.8)

    color = style.colors[0]
    (trail_line,) = ax.plot([], [], color=color, linewidth=2, alpha=0.8)
    (dot,) = ax.plot([], [], "o", color=color, markersize=9)

    ax.set_xlim(*m_range)
    ax.set_ylim(*q_range)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$q$")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    def update(frame: int):
        if trail:
            trail_line.set_data(m_traj[: frame + 1], q_traj[: frame + 1])
        dot.set_data([m_traj[frame]], [q_traj[frame]])
        return [trail_line, dot]

    return FuncAnimation(fig, update, frames=len(m_traj), interval=interval, blit=True)


def animate_overlap_matrix(
    matrices: list[np.ndarray] | np.ndarray,
    t_values: np.ndarray | None = None,
    cmap: str = "RdBu_r",
    interval: int = 100,
    figsize: tuple[float, float] = (5.5, 5),
    title: str = r"$M$ (student-teacher overlaps)",
) -> FuncAnimation:
    """
    Animate an overlap matrix evolving over training.

    Specialization in committee machines / MLPs shows up as an emerging
    (permuted) diagonal.

    Args:
        matrices: Sequence of 2D arrays over time.
        t_values: Times for frame titles (optional).
        cmap: Diverging colormap.
        interval: Milliseconds between frames.
        figsize: Figure size.
        title: Base title.

    Returns:
        FuncAnimation.

    """
    _style()
    mats = [np.atleast_2d(np.asarray(m, dtype=float)) for m in matrices]
    if not mats:
        raise ValueError("matrices is empty")

    vabs = max(max(np.abs(m).max() for m in mats), 1e-12)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mats[0], cmap=cmap, vmin=-vabs, vmax=vabs)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("teacher unit")
    ax.set_ylabel("student unit")
    ax.set_xticks(range(mats[0].shape[1]))
    ax.set_yticks(range(mats[0].shape[0]))
    fig.tight_layout()

    def update(frame: int):
        im.set_data(mats[frame])
        suffix = f"  (t = {t_values[frame]:.2f})" if t_values is not None else f"  [{frame}]"
        ax.set_title(title + suffix)
        return [im]

    return FuncAnimation(fig, update, frames=len(mats), interval=interval, blit=False)


def animate_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    weights: list[np.ndarray] | np.ndarray,
    t_values: np.ndarray | None = None,
    metric_values: np.ndarray | None = None,
    metric_name: str = r"$\epsilon_g$",
    interval: int = 60,
    figsize: tuple[float, float] = (5.5, 5),
    title: str = "",
) -> FuncAnimation:
    r"""
    Animate a 2D linear classifier's decision boundary during training.

    The most immediately intuitive animation in the package: labelled
    data points are fixed, and the decision line $\{x : w \cdot x = 0\}$
    rotates into place as the classifier is trained -- a direct visual
    for what "recovering the teacher direction" / generalization means,
    useful for teaching the Gaussian-mixture classification setting
    (see docs/order_parameters.md, section 5) to newcomers.

    Args:
        X: Inputs of shape (n, 2) (exactly 2D; project down first for
            higher-dimensional data).
        y: Labels of shape (n,), any two distinct values (colored).
        weights: Sequence of (2,) classifier-direction snapshots over
            training (only the direction matters, not the norm).
        t_values: Optional times/epochs for the frame title.
        metric_values: Optional scalar (e.g. test error or overlap) to
            annotate each frame.
        metric_name: Label for `metric_values` in the frame title.
        interval: Milliseconds between frames.
        figsize: Figure size.
        title: Base title.

    Returns:
        FuncAnimation.

    """
    style = _style()
    X = np.asarray(X)
    y = np.asarray(y)
    W = np.stack([np.asarray(w, dtype=float).reshape(2) for w in weights])
    if X.shape[1] != 2:
        raise ValueError(f"animate_decision_boundary requires 2D inputs, got shape {X.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    classes = np.unique(y)
    colors = [style.colors[i % len(style.colors)] for i in range(len(classes))]
    for c, color in zip(classes, colors, strict=True):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], s=14, alpha=0.6, color=color, label=f"$y={c:g}$")

    r = 1.3 * np.abs(X).max()
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    (boundary,) = ax.plot([], [], "-", color="black", linewidth=2.5)
    (normal,) = ax.plot([], [], "-", color="crimson", linewidth=1.5, alpha=0.8)
    ttl = ax.set_title(title + "\n ", fontsize=9.5)
    fig.tight_layout()

    def update(frame: int):
        w = W[frame]
        wn = w / (np.linalg.norm(w) + 1e-12)
        perp = np.array([-wn[1], wn[0]])
        boundary.set_data([-r * perp[0], r * perp[0]], [-r * perp[1], r * perp[1]])
        normal.set_data([0, 0.5 * r * wn[0]], [0, 0.5 * r * wn[1]])
        suffix = f"\n(t = {t_values[frame]:.2f})" if t_values is not None else "\n"
        if metric_values is not None:
            suffix += f"measured {metric_name} = {metric_values[frame]:.3f}"
        ttl.set_text(title + suffix)
        return [boundary, normal, ttl]

    return FuncAnimation(fig, update, frames=len(W), interval=interval, blit=False)


def animate_curve_and_matrix(
    t_values: np.ndarray,
    curves: dict[str, np.ndarray],
    matrices: list[np.ndarray] | np.ndarray,
    curve_ylabel: str = r"$\epsilon_g$",
    matrix_title: str = r"$R_{km} = \mathbf{w}_k \cdot \mathbf{w}^*_m / d$",
    logy: bool = True,
    cmap: str = "RdBu_r",
    interval: int = 60,
    figsize: tuple[float, float] = (10.5, 4.4),
    suptitle: str = "",
) -> FuncAnimation:
    """
    Animate a learning curve and an overlap matrix side by side.

    The canonical two-panel view of committee-machine specialization:
    the left panel draws e.g. the exact generalization error over time
    (plateau, then symmetry-breaking drop), while the right panel shows
    the student-teacher overlap matrix R developing its (permuted)
    diagonal at the moment of escape.

    Args:
        t_values: Time grid of shape (S,).
        curves: Mapping name -> trajectory of shape (S,) for the left panel.
        matrices: Sequence of S 2D overlap matrices for the right panel.
        curve_ylabel: Left-panel y label.
        matrix_title: Right-panel base title.
        logy: Log-scale the left panel (typical for eps_g).
        cmap: Diverging colormap for the matrix.
        interval: Milliseconds between frames.
        figsize: Figure size.
        suptitle: Overall figure title.

    Returns:
        FuncAnimation.

    """
    style = _style()
    t = np.asarray(t_values, dtype=float)
    mats = [np.atleast_2d(np.asarray(m, dtype=float)) for m in matrices]
    if len(mats) != len(t):
        raise ValueError(f"len(matrices)={len(mats)} must equal len(t_values)={len(t)}")

    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=figsize)

    lines = {}
    dots = {}
    data = {k: np.asarray(v, dtype=float) for k, v in curves.items()}
    for i, (name, v) in enumerate(data.items()):
        color = style.colors[i % len(style.colors)]
        (lines[name],) = ax_c.plot([], [], color=color, linewidth=2, label=name)
        (dots[name],) = ax_c.plot([], [], "o", color=color, markersize=7)
    all_vals = np.concatenate(list(data.values()))
    ax_c.set_xlim(t.min(), t.max())
    if logy:
        ax_c.set_yscale("log")
        positive = all_vals[all_vals > 0]
        lo = positive.min() * 0.5 if positive.size else 1e-3
        ax_c.set_ylim(lo, all_vals.max() * 2)
    else:
        pad = 0.1 * (all_vals.max() - all_vals.min() + 1e-12)
        ax_c.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
    ax_c.set_xlabel(r"$t = \#\mathrm{samples}/d$")
    ax_c.set_ylabel(curve_ylabel)
    ax_c.grid(True, linestyle="--", alpha=0.3)
    ax_c.legend(loc="best", fontsize=9)

    vabs = max(max(np.abs(m).max() for m in mats), 1e-12)
    im = ax_m.imshow(mats[0], cmap=cmap, vmin=-vabs, vmax=vabs)
    fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
    ax_m.set_xlabel("teacher unit")
    ax_m.set_ylabel("student unit")
    ax_m.set_xticks(range(mats[0].shape[1]))
    ax_m.set_yticks(range(mats[0].shape[0]))
    ttl_m = ax_m.set_title(matrix_title)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()

    def update(frame: int):
        for name in data:
            lines[name].set_data(t[: frame + 1], data[name][: frame + 1])
            dots[name].set_data([t[frame]], [data[name][frame]])
        im.set_data(mats[frame])
        ttl_m.set_text(matrix_title + f"  (t = {t[frame]:.1f})")
        return [*lines.values(), *dots.values(), im, ttl_m]

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)


def save_animation(
    anim: FuncAnimation,
    path: str,
    fps: int = 20,
    dpi: int = 100,
) -> None:
    """
    Save an animation, choosing the writer from the file extension.

    Args:
        anim: FuncAnimation to save.
        path: Output path (.gif uses pillow; .mp4 uses ffmpeg).
        fps: Frames per second.
        dpi: Resolution.

    """
    writer = "pillow" if path.lower().endswith(".gif") else "ffmpeg"
    anim.save(path, writer=writer, fps=fps, dpi=dpi)
