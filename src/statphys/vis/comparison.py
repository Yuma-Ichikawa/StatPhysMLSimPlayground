"""Theory vs experiment comparison plots."""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statphys.simulation.base import SimulationResult
from statphys.vis.plotter import PlotStyle, Plotter


class ComparisonPlotter(Plotter):
    """
    Plotter for comparing theory predictions to experiments.

    Creates publication-quality comparison plots showing
    theoretical curves alongside experimental data with error bars.
    """

    def __init__(
        self,
        style: PlotStyle | None = None,
        theory_linestyle: str = "-",
        experiment_marker: str = "o",
        param_labels: dict[str, str] | None = None,
    ):
        """
        Initialize ComparisonPlotter.

        Args:
            style: Plot style.
            theory_linestyle: Line style for theory curves.
            experiment_marker: Marker for experiment points.
            param_labels: Custom parameter labels.

        """
        super().__init__(style)
        self.theory_linestyle = theory_linestyle
        self.experiment_marker = experiment_marker

        self.param_labels = param_labels or {
            "m": r"$m$",
            "q": r"$q$",
            "eg": r"$E_g$",
            "et": r"$E_t$",
        }

    def plot_theory_vs_experiment(
        self,
        result: SimulationResult,
        params_to_plot: list[str] | None = None,
        param_indices: dict[str, int] | None = None,
        separate_plots: bool = False,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Any]:
        """
        Plot theory vs experiment comparison.

        Args:
            result: SimulationResult with both theory and experiment data.
            params_to_plot: Which parameters to plot.
            param_indices: Mapping from param name to index in experiment array.
            separate_plots: Create separate subplot for each parameter.
            figsize: Figure size.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        # Determine parameters to plot
        if params_to_plot is None:
            if result.theory_results is not None:
                params_to_plot = list(result.theory_results.order_params.keys())
            else:
                params_to_plot = ["m", "q", "eg"]

        # Remove non-plottable params
        params_to_plot = [p for p in params_to_plot if p not in ["alpha", "t"]]

        # Default param indices
        if param_indices is None:
            param_indices = {name: i for i, name in enumerate(params_to_plot)}

        n_params = len(params_to_plot)

        # Get x values
        if result.theory_type.value == "replica":
            x_values = np.array(result.experiment_results["alpha_values"])
            xlabel = r"$\alpha$"
            exp_mean_key = "order_params_mean"
            exp_std_key = "order_params_std"
        else:
            x_values = np.array(result.experiment_results["t_values"])
            xlabel = r"$t$"
            exp_mean_key = "trajectories_mean"
            exp_std_key = "trajectories_std"

        # Create figure
        if separate_plots:
            figsize = figsize or (4 * n_params, 4)
            fig, axes = self.create_figure(1, n_params, figsize=figsize)
            if n_params == 1:
                axes = [axes]
        else:
            figsize = figsize or self.style.figsize
            fig, ax = self.create_figure(figsize=figsize)
            axes = [ax] * n_params

        # Get experiment data
        exp_mean = np.array(result.experiment_results[exp_mean_key])
        exp_std = np.array(result.experiment_results[exp_std_key])

        # Plot each parameter
        for i, param in enumerate(params_to_plot):
            ax = axes[i] if separate_plots else axes[0]
            color = self.style.colors[i % len(self.style.colors)]
            label = self.param_labels.get(param, param)

            # Get param index for experiment data
            idx = param_indices.get(param, i)

            # Plot theory if available
            if result.theory_results is not None and param in result.theory_results.order_params:
                theory_values = np.array(result.theory_results.order_params[param])
                theory_x = np.array(result.theory_results.param_values)

                ax.plot(
                    theory_x,
                    theory_values,
                    color=color,
                    linestyle=self.theory_linestyle,
                    linewidth=self.style.line_width,
                    label=f"{label} (theory)" if separate_plots else label,
                )

            # Plot experiment
            if idx < exp_mean.shape[1]:
                ax.errorbar(
                    x_values,
                    exp_mean[:, idx],
                    yerr=exp_std[:, idx],
                    color=color,
                    marker=self.experiment_marker,
                    linestyle="none",
                    markersize=self.style.marker_size,
                    capsize=3,
                    label=f"{label} (exp)" if result.theory_results else label,
                )

            if separate_plots:
                self.set_labels(ax, xlabel=xlabel, ylabel=label)
                self.add_grid(ax)

        if not separate_plots:
            self.set_labels(ax, xlabel=xlabel, ylabel="Order Parameters")
            self.add_legend(ax)
            self.add_grid(ax)
        else:
            # Add common legend for separate plots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))

        fig.tight_layout()
        return fig, axes[0] if not separate_plots else axes

    def plot_generalization_error(
        self,
        result: SimulationResult,
        eg_index: int = 2,
        ax: Axes | None = None,
        show_training_error: bool = False,
        et_index: int = 3,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot generalization error comparison.

        Args:
            result: Simulation result.
            eg_index: Index of E_g in order params array.
            ax: Axes.
            show_training_error: Also plot training error.
            et_index: Index of E_t.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        # Get x values
        if result.theory_type.value == "replica":
            x_values = np.array(result.experiment_results["alpha_values"])
            xlabel = r"$\alpha$"
            exp_mean = np.array(result.experiment_results["order_params_mean"])
            exp_std = np.array(result.experiment_results["order_params_std"])
        else:
            x_values = np.array(result.experiment_results["t_values"])
            xlabel = r"$t$"
            exp_mean = np.array(result.experiment_results["trajectories_mean"])
            exp_std = np.array(result.experiment_results["trajectories_std"])

        # Plot generalization error
        color_eg = self.style.colors[0]

        # Theory
        if result.theory_results is not None and "eg" in result.theory_results.order_params:
            theory_eg = np.array(result.theory_results.order_params["eg"])
            theory_x = np.array(result.theory_results.param_values)
            ax.plot(
                theory_x,
                theory_eg,
                color=color_eg,
                linestyle=self.theory_linestyle,
                linewidth=self.style.line_width,
                label=r"$E_g$ (theory)",
            )

        # Experiment
        if eg_index < exp_mean.shape[1]:
            ax.errorbar(
                x_values,
                exp_mean[:, eg_index],
                yerr=exp_std[:, eg_index],
                color=color_eg,
                marker=self.experiment_marker,
                linestyle="none",
                capsize=3,
                label=r"$E_g$ (experiment)",
            )

        # Training error if requested
        if show_training_error:
            color_et = self.style.colors[1]

            if "et" in result.experiment_results:
                et_mean = np.array(result.experiment_results["train_loss_mean"])
                et_std = np.array(result.experiment_results["train_loss_std"])

                ax.errorbar(
                    x_values,
                    et_mean,
                    yerr=et_std,
                    color=color_et,
                    marker="s",
                    linestyle="none",
                    capsize=3,
                    label=r"$E_t$ (experiment)",
                )

        self.set_labels(ax, xlabel=xlabel, ylabel="Error")
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax

    def plot_multiple_results(
        self,
        results: dict[str, SimulationResult],
        param: str = "eg",
        param_index: int = 2,
        ax: Axes | None = None,
        show_theory: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot comparison of multiple simulation results.

        Useful for comparing different models or configurations.

        Args:
            results: Dictionary mapping names to results.
            param: Parameter to plot.
            param_index: Index in experiment array.
            ax: Axes.
            show_theory: Show theory curves.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        for i, (name, result) in enumerate(results.items()):
            color = self.style.colors[i % len(self.style.colors)]

            # Get x values
            if result.theory_type.value == "replica":
                x_values = np.array(result.experiment_results["alpha_values"])
                exp_mean = np.array(result.experiment_results["order_params_mean"])
                exp_std = np.array(result.experiment_results["order_params_std"])
            else:
                x_values = np.array(result.experiment_results["t_values"])
                exp_mean = np.array(result.experiment_results["trajectories_mean"])
                exp_std = np.array(result.experiment_results["trajectories_std"])

            # Theory
            if (
                show_theory
                and result.theory_results
                and param in result.theory_results.order_params
            ):
                theory_values = np.array(result.theory_results.order_params[param])
                theory_x = np.array(result.theory_results.param_values)
                ax.plot(
                    theory_x,
                    theory_values,
                    color=color,
                    linestyle=self.theory_linestyle,
                    linewidth=self.style.line_width,
                )

            # Experiment
            if param_index < exp_mean.shape[1]:
                ax.errorbar(
                    x_values,
                    exp_mean[:, param_index],
                    yerr=exp_std[:, param_index],
                    color=color,
                    marker=self.experiment_marker,
                    linestyle="none",
                    capsize=3,
                    label=name,
                )

        xlabel = r"$\alpha$" if result.theory_type.value == "replica" else r"$t$"
        ylabel = self.param_labels.get(param, param)

        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax
