"""
Default plotting functions for publication-quality figures.

This module provides ready-to-use plotting functions for:
- Generalization error vs alpha (Replica)
- All order parameters vs alpha (Replica)
- Generalization error vs t (Online)
- All order parameters vs t (Online)
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Publication-quality style settings
PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.size": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.linewidth": 1.0,
    "axes.xmargin": 0.01,
    "axes.ymargin": 0.01,
    "legend.fancybox": False,
    "legend.framealpha": 1,
    "legend.edgecolor": "black",
    "mathtext.fontset": "stix",
}

# Default figure size (single plot)
DEFAULT_FIGSIZE = (6.4, 4.8)

# Style lists for multiple curves
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "1", "2", "3"]
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def apply_paper_style() -> None:
    """Apply publication-quality style to matplotlib."""
    plt.rcParams.update(PAPER_STYLE)


def _setup_ax(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
) -> None:
    """Setup axis with labels, legend, and grid."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, linestyle="--", alpha=0.3)


def plot_generalization_error_alpha(
    alpha_values: np.ndarray,
    eg_mean: np.ndarray,
    eg_std: np.ndarray | None = None,
    eg_theory: np.ndarray | None = None,
    title: str | None = None,
    xlabel: str = r"$\alpha = n/d$",
    ylabel: str = r"$E_g$ (Generalization Error)",
    exp_label: str = "Experiment",
    theory_label: str = "Theory",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot generalization error vs alpha.

    Args:
        alpha_values: Array of alpha values.
        eg_mean: Mean generalization error.
        eg_std: Standard deviation (optional, for error bars).
        eg_theory: Theoretical prediction (optional).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        exp_label: Label for experimental data.
        theory_label: Label for theoretical curve.
        figsize: Figure size.
        save_path: Path to save figure (optional).
        show: Whether to display the figure.

    Returns:
        Tuple of (Figure, Axes).

    """
    apply_paper_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot experimental data
    if eg_std is not None:
        ax.errorbar(
            alpha_values,
            eg_mean,
            yerr=eg_std,
            marker=MARKERS[0],
            color=COLORS[0],
            linestyle="none",
            capsize=3,
            markersize=8,
            label=exp_label,
        )
    else:
        ax.plot(
            alpha_values,
            eg_mean,
            marker=MARKERS[0],
            color=COLORS[0],
            linestyle="none",
            markersize=8,
            label=exp_label,
        )

    # Plot theory if provided
    if eg_theory is not None:
        ax.plot(
            alpha_values,
            eg_theory,
            linestyle=LINE_STYLES[0],
            color=COLORS[1],
            linewidth=2,
            label=theory_label,
        )

    _setup_ax(ax, xlabel, ylabel, title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_order_params_alpha(
    alpha_values: np.ndarray,
    m_mean: np.ndarray,
    q_mean: np.ndarray,
    eg_mean: np.ndarray,
    m_std: np.ndarray | None = None,
    q_std: np.ndarray | None = None,
    eg_std: np.ndarray | None = None,
    m_theory: np.ndarray | None = None,
    q_theory: np.ndarray | None = None,
    eg_theory: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot all order parameters (m, q, Eg) vs alpha in a single figure.

    Args:
        alpha_values: Array of alpha values.
        m_mean: Mean teacher overlap.
        q_mean: Mean self-overlap.
        eg_mean: Mean generalization error.
        m_std: Std of m (optional).
        q_std: Std of q (optional).
        eg_std: Std of Eg (optional).
        m_theory: Theoretical m (optional).
        q_theory: Theoretical q (optional).
        eg_theory: Theoretical Eg (optional).
        title: Overall title.
        figsize: Figure size.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Tuple of (Figure, array of Axes).

    """
    apply_paper_style()

    if figsize is None:
        figsize = (DEFAULT_FIGSIZE[0] * 3, DEFAULT_FIGSIZE[1])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    data = [
        (m_mean, m_std, m_theory, r"$m$ (Teacher Overlap)"),
        (q_mean, q_std, q_theory, r"$q$ (Self Overlap)"),
        (eg_mean, eg_std, eg_theory, r"$E_g$ (Gen. Error)"),
    ]

    for i, (ax, (mean, std, theory, ylabel)) in enumerate(zip(axes, data, strict=False)):
        # Plot experimental data
        if std is not None:
            ax.errorbar(
                alpha_values,
                mean,
                yerr=std,
                marker=MARKERS[0],
                color=COLORS[i],
                linestyle="none",
                capsize=3,
                markersize=8,
                label="Experiment",
            )
        else:
            ax.plot(
                alpha_values,
                mean,
                marker=MARKERS[0],
                color=COLORS[i],
                linestyle="none",
                markersize=8,
                label="Experiment",
            )

        # Plot theory if provided
        if theory is not None:
            ax.plot(
                alpha_values,
                theory,
                linestyle=LINE_STYLES[0],
                color="black",
                linewidth=2,
                label="Theory",
            )

        _setup_ax(ax, r"$\alpha = n/d$", ylabel)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_generalization_error_time(
    t_values: np.ndarray,
    eg_mean: np.ndarray,
    eg_std: np.ndarray | None = None,
    eg_theory: np.ndarray | None = None,
    title: str | None = None,
    xlabel: str = r"$t = \tau / d$",
    ylabel: str = r"$E_g$ (Generalization Error)",
    exp_label: str = "Experiment",
    theory_label: str = "Theory",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot generalization error vs time for online learning.

    Args:
        t_values: Array of time values (t = tau/d).
        eg_mean: Mean generalization error.
        eg_std: Standard deviation (optional).
        eg_theory: Theoretical prediction (optional).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        exp_label: Label for experimental data.
        theory_label: Label for theoretical curve.
        figsize: Figure size.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Tuple of (Figure, Axes).

    """
    apply_paper_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot experimental data with shaded region
    if eg_std is not None:
        ax.plot(
            t_values,
            eg_mean,
            color=COLORS[0],
            linewidth=2,
            label=exp_label,
        )
        ax.fill_between(
            t_values,
            eg_mean - eg_std,
            eg_mean + eg_std,
            color=COLORS[0],
            alpha=0.3,
        )
    else:
        ax.plot(
            t_values,
            eg_mean,
            color=COLORS[0],
            linewidth=2,
            label=exp_label,
        )

    # Plot theory if provided
    if eg_theory is not None:
        ax.plot(
            t_values,
            eg_theory,
            linestyle=LINE_STYLES[1],
            color=COLORS[1],
            linewidth=2,
            label=theory_label,
        )

    _setup_ax(ax, xlabel, ylabel, title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_order_params_time(
    t_values: np.ndarray,
    m_mean: np.ndarray,
    q_mean: np.ndarray,
    eg_mean: np.ndarray,
    m_std: np.ndarray | None = None,
    q_std: np.ndarray | None = None,
    eg_std: np.ndarray | None = None,
    m_theory: np.ndarray | None = None,
    q_theory: np.ndarray | None = None,
    eg_theory: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot all order parameters (m, q, Eg) vs time for online learning.

    Args:
        t_values: Array of time values (t = tau/d).
        m_mean: Mean teacher overlap.
        q_mean: Mean self-overlap.
        eg_mean: Mean generalization error.
        m_std: Std of m (optional).
        q_std: Std of q (optional).
        eg_std: Std of Eg (optional).
        m_theory: Theoretical m (optional).
        q_theory: Theoretical q (optional).
        eg_theory: Theoretical Eg (optional).
        title: Overall title.
        figsize: Figure size.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Tuple of (Figure, array of Axes).

    """
    apply_paper_style()

    if figsize is None:
        figsize = (DEFAULT_FIGSIZE[0] * 3, DEFAULT_FIGSIZE[1])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    data = [
        (m_mean, m_std, m_theory, r"$m$ (Teacher Overlap)"),
        (q_mean, q_std, q_theory, r"$q$ (Self Overlap)"),
        (eg_mean, eg_std, eg_theory, r"$E_g$ (Gen. Error)"),
    ]

    for i, (ax, (mean, std, theory, ylabel)) in enumerate(zip(axes, data, strict=False)):
        # Plot experimental data with shaded region
        if std is not None:
            ax.plot(
                t_values,
                mean,
                color=COLORS[i],
                linewidth=2,
                label="Experiment",
            )
            ax.fill_between(
                t_values,
                mean - std,
                mean + std,
                color=COLORS[i],
                alpha=0.3,
            )
        else:
            ax.plot(
                t_values,
                mean,
                color=COLORS[i],
                linewidth=2,
                label="Experiment",
            )

        # Plot theory if provided
        if theory is not None:
            ax.plot(
                t_values,
                theory,
                linestyle=LINE_STYLES[1],
                color="black",
                linewidth=2,
                label="Theory",
            )

        _setup_ax(ax, r"$t = \tau / d$", ylabel)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_from_replica_results(
    results: Any,
    plot_type: str = "all",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes] | tuple[Figure, np.ndarray]:
    """
    Plot from ReplicaSimulation results.

    Args:
        results: SimulationResult object from ReplicaSimulation.
        plot_type: "eg" for generalization error only, "all" for all order params.
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Tuple of (Figure, Axes or array of Axes).

    """
    exp_results = results.experiment_results
    alpha_values = np.array(exp_results["alpha_values"])
    op_mean = np.array(exp_results["order_params_mean"])
    op_std = np.array(exp_results["order_params_std"])

    m_mean, q_mean, eg_mean = op_mean[:, 0], op_mean[:, 1], op_mean[:, 2]
    m_std, q_std, eg_std = op_std[:, 0], op_std[:, 1], op_std[:, 2]

    # Get theory results if available
    m_theory = q_theory = eg_theory = None
    if results.theory_results:
        theory = results.theory_results
        if "m" in theory:
            m_theory = np.array(theory["m"])
        if "q" in theory:
            q_theory = np.array(theory["q"])
        if "eg" in theory:
            eg_theory = np.array(theory["eg"])

    if plot_type == "eg":
        return plot_generalization_error_alpha(
            alpha_values=alpha_values,
            eg_mean=eg_mean,
            eg_std=eg_std,
            eg_theory=eg_theory,
            title=title,
            save_path=save_path,
            show=show,
        )
    else:
        return plot_order_params_alpha(
            alpha_values=alpha_values,
            m_mean=m_mean,
            q_mean=q_mean,
            eg_mean=eg_mean,
            m_std=m_std,
            q_std=q_std,
            eg_std=eg_std,
            m_theory=m_theory,
            q_theory=q_theory,
            eg_theory=eg_theory,
            title=title,
            save_path=save_path,
            show=show,
        )


def plot_from_online_results(
    results: Any,
    plot_type: str = "all",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes] | tuple[Figure, np.ndarray]:
    """
    Plot from OnlineSimulation results.

    Args:
        results: SimulationResult object from OnlineSimulation.
        plot_type: "eg" for generalization error only, "all" for all order params.
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Tuple of (Figure, Axes or array of Axes).

    """
    exp_results = results.experiment_results
    t_values = np.array(exp_results["t_values"])
    traj_mean = np.array(exp_results["trajectories_mean"])
    traj_std = np.array(exp_results["trajectories_std"])

    m_mean, q_mean, eg_mean = traj_mean[:, 0], traj_mean[:, 1], traj_mean[:, 2]
    m_std, q_std, eg_std = traj_std[:, 0], traj_std[:, 1], traj_std[:, 2]

    # Get theory results if available
    m_theory = q_theory = eg_theory = None
    if results.theory_results:
        theory = results.theory_results
        if "m" in theory:
            m_theory = np.array(theory["m"])
        if "q" in theory:
            q_theory = np.array(theory["q"])
        if "eg" in theory:
            eg_theory = np.array(theory["eg"])

    if plot_type == "eg":
        return plot_generalization_error_time(
            t_values=t_values,
            eg_mean=eg_mean,
            eg_std=eg_std,
            eg_theory=eg_theory,
            title=title,
            save_path=save_path,
            show=show,
        )
    else:
        return plot_order_params_time(
            t_values=t_values,
            m_mean=m_mean,
            q_mean=q_mean,
            eg_mean=eg_mean,
            m_std=m_std,
            q_std=q_std,
            eg_std=eg_std,
            m_theory=m_theory,
            q_theory=q_theory,
            eg_theory=eg_theory,
            title=title,
            save_path=save_path,
            show=show,
        )
