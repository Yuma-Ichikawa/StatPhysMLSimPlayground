"""
Dashboard plots for physics order parameters.

plot_order_parameter_dashboard renders the standard 4-panel summary of a
replica-resolved sweep (ExperimentResult from run_order_parameters):

    top-left:     order parameters m_hat / q_ab (+ extras) vs alpha
    top-right:    generalization error eps_g vs alpha (log-log)
    bottom-left:  susceptibility chi_m vs alpha
    bottom-right: Binder cumulant U_4 vs alpha
"""

import numpy as np

__all__ = ["plot_order_parameter_dashboard"]


def plot_order_parameter_dashboard(
    result,
    title: str = "",
    extra_metrics: tuple[str, ...] = (),
    logx: bool = True,
    show: bool = False,
):
    """
    4-panel order-parameter dashboard for a run_order_parameters result.

    Args:
        result: ExperimentResult produced by
            TeacherStudentExperiment.run_order_parameters.
        title: Figure title.
        extra_metrics: Additional per-replica metrics to overlay on the
            order-parameter panel (e.g. ("specialization",)).
        logx: Log-scale the alpha axis.
        show: Call plt.show().

    Returns:
        Tuple of (Figure, array of Axes).

    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.array(result.x_values)

    ax = axes[0, 0]
    for m in ("m_hat", "q_ab_mean") + tuple(extra_metrics):
        if m in result.records:
            mean, std = result.mean(m), result.std(m)
            (line,) = ax.plot(x, mean, "o-", markersize=4, label=m)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel("order parameters")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    mean, std = result.mean("test_error"), result.std("test_error")
    ax.plot(x, mean, "s-", color="crimson", markersize=4)
    ax.fill_between(x, np.maximum(mean - std, 1e-12), mean + std, alpha=0.2, color="crimson")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"generalization error $\epsilon_g$")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, result.mean("chi_m"), "^-", color="darkorange", markersize=5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"$\chi_m = d\,\mathrm{Var}[\hat m]$")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, result.mean("binder_m"), "d-", color="seagreen", markersize=5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"Binder $U_4$")
    ax.grid(True, linestyle="--", alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
