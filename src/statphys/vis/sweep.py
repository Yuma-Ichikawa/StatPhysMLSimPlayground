"""
Parameter-sweep and diagnostic visualization.

Works directly with the output of ``SimulationRunner.run_parameter_sweep``
(a dict mapping parameter values to SimulationResult) and provides
theory-vs-experiment diagnostic plots.

Example:
    >>> runner = SimulationRunner()
    >>> sweep = runner.run_parameter_sweep(
    ...     base_config, dataset, LinearRegression, RidgeLoss(0.01),
    ...     param_name="lr", param_values=[0.1, 0.3, 0.5],
    ... )
    >>> from statphys.vis.sweep import SweepPlotter
    >>> SweepPlotter().plot_sweep(sweep, param_label=r"$\\eta$")

"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statphys.simulation.base import SimulationResult
from statphys.vis.plotter import PlotStyle, Plotter


class SweepPlotter(Plotter):
    """Plotter for parameter sweeps and theory-experiment diagnostics."""

    def __init__(self, style: PlotStyle | None = None):
        super().__init__(style)

    @staticmethod
    def _extract_xy(
        result: SimulationResult,
        param_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, mean, std) for one order parameter from a result."""
        exp = result.experiment_results
        if result.theory_type.value == "replica":
            x = np.array(exp["alpha_values"])
            mean = np.array(exp["order_params_mean"])[:, param_index]
            std = np.array(exp["order_params_std"])[:, param_index]
        else:
            x = np.array(exp["t_values"])
            mean = np.array(exp["trajectories_mean"])[:, param_index]
            std = np.array(exp["trajectories_std"])[:, param_index]
        return x, mean, std

    def plot_sweep(
        self,
        sweep_results: dict[Any, SimulationResult],
        param: str = "eg",
        param_index: int = 2,
        param_label: str | None = None,
        ax: Axes | None = None,
        show_theory: bool = True,
        logy: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Overlay one order parameter across a parameter sweep.

        Args:
            sweep_results: Dict mapping swept values to SimulationResult
                (output of SimulationRunner.run_parameter_sweep).
            param: Name of the order parameter (for theory lookup / label).
            param_index: Column index in the experiment arrays.
            param_label: Label of the swept parameter for the legend.
            ax: Axes. Creates new if None.
            show_theory: Draw theory curves when available.
            logy: Log-scale y axis (useful for E_g decay).
            **kwargs: Extra errorbar arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        xlabel = r"$\alpha$"
        for i, (value, result) in enumerate(sweep_results.items()):
            color = self.style.colors[i % len(self.style.colors)]
            x, mean, std = self._extract_xy(result, param_index)
            xlabel = r"$\alpha$" if result.theory_type.value == "replica" else r"$t$"

            label = f"{param_label} = {value}" if param_label else f"{value}"
            ax.errorbar(
                x,
                mean,
                yerr=std,
                color=color,
                marker="o",
                linestyle="none",
                markersize=self.style.marker_size * 0.7,
                capsize=3,
                label=label,
                **kwargs,
            )

            if (
                show_theory
                and result.theory_results is not None
                and param in result.theory_results.order_params
            ):
                ax.plot(
                    np.array(result.theory_results.param_values),
                    np.array(result.theory_results.order_params[param]),
                    color=color,
                    linewidth=self.style.line_width,
                )

        if logy:
            ax.set_yscale("log")

        self.set_labels(ax, xlabel=xlabel, ylabel=param)
        self.add_legend(ax)
        self.add_grid(ax)
        return fig, ax

    def plot_theory_experiment_scatter(
        self,
        result: SimulationResult,
        params_to_plot: list[str] | None = None,
        param_indices: dict[str, int] | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Diagnostic scatter: theory prediction vs experimental mean.

        Points on the diagonal indicate perfect theory-experiment agreement.

        Args:
            result: SimulationResult with theory_results.
            params_to_plot: Order parameter names. Defaults to those
                present in the theory results (excluding axes variables).
            param_indices: Mapping from name to experiment column index.
            ax: Axes.
            **kwargs: Extra scatter arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if result.theory_results is None:
            raise ValueError("result has no theory_results to compare against")

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        theory_params = result.theory_results.order_params
        if params_to_plot is None:
            params_to_plot = [k for k in theory_params if k not in ("alpha", "t")]
        if param_indices is None:
            param_indices = {name: i for i, name in enumerate(params_to_plot)}

        all_vals: list[float] = []
        for i, name in enumerate(params_to_plot):
            idx = param_indices.get(name, i)
            _, exp_mean, exp_std = self._extract_xy(result, idx)
            theory_vals = np.array(theory_params[name])

            n = min(len(exp_mean), len(theory_vals))
            color = self.style.colors[i % len(self.style.colors)]
            ax.errorbar(
                theory_vals[:n],
                exp_mean[:n],
                yerr=exp_std[:n],
                color=color,
                marker="o",
                linestyle="none",
                capsize=2,
                markersize=self.style.marker_size * 0.6,
                label=name,
                **kwargs,
            )
            all_vals.extend(theory_vals[:n].tolist())
            all_vals.extend(exp_mean[:n].tolist())

        # y = x reference line
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad = 0.05 * (hi - lo + 1e-12)
            line = np.array([lo - pad, hi + pad])
            ax.plot(line, line, color="gray", linestyle="--", linewidth=1, zorder=0)

        self.set_labels(ax, xlabel="theory", ylabel="experiment")
        self.add_legend(ax)
        self.add_grid(ax)
        ax.set_aspect("equal", adjustable="datalim")
        return fig, ax

    def plot_convergence_check(
        self,
        result: SimulationResult,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Visualize per-alpha convergence flags and training loss (replica).

        Args:
            result: SimulationResult from ReplicaSimulation.
            ax: Axes.
            **kwargs: Extra plot arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        exp = result.experiment_results
        if "train_loss_mean" not in exp:
            raise ValueError("result does not contain replica training diagnostics")

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        alpha = np.array(exp["alpha_values"])
        loss_mean = np.array(exp["train_loss_mean"])
        loss_std = np.array(exp["train_loss_std"])

        ax.errorbar(
            alpha,
            loss_mean,
            yerr=loss_std,
            color=self.style.colors[0],
            marker="o",
            capsize=3,
            label="train loss (per sample)",
            **kwargs,
        )

        # Mark non-converged points
        converged = np.array(exp.get("converged", []))
        if converged.size:
            frac = converged.mean(axis=0)  # fraction of seeds converged per alpha
            bad = frac < 1.0
            if bad.any():
                ax.plot(
                    alpha[bad],
                    loss_mean[bad],
                    "x",
                    color="tab:red",
                    markersize=12,
                    markeredgewidth=2,
                    label="not fully converged",
                )

        self.set_labels(ax, xlabel=r"$\alpha$", ylabel="training loss")
        self.add_legend(ax)
        self.add_grid(ax)
        return fig, ax
