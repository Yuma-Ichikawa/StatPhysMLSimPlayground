"""
Order parameter visualization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from statphys.vis.plotter import Plotter, PlotStyle
from statphys.simulation.base import SimulationResult


class OrderParamPlotter(Plotter):
    """
    Plotter specialized for order parameter visualization.

    Provides methods for plotting order parameter trajectories,
    phase diagrams, and convergence diagnostics.
    """

    def __init__(
        self,
        style: Optional[PlotStyle] = None,
        param_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize OrderParamPlotter.

        Args:
            style: Plot style.
            param_labels: Custom labels for order parameters.
        """
        super().__init__(style)

        # Default parameter labels (with LaTeX)
        self.param_labels = param_labels or {
            "m": r"$m$ (teacher overlap)",
            "q": r"$q$ (self overlap)",
            "eg": r"$E_g$ (generalization error)",
            "et": r"$E_t$ (training error)",
            "alpha": r"$\alpha$",
            "t": r"$t$",
        }

    def plot_vs_alpha(
        self,
        alpha_values: np.ndarray,
        order_params: Dict[str, np.ndarray],
        ax: Optional[Axes] = None,
        params_to_plot: Optional[List[str]] = None,
        show_all: bool = True,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot order parameters vs alpha.

        Args:
            alpha_values: Alpha values.
            order_params: Dictionary of order parameter arrays.
            ax: Axes to plot on. Creates new if None.
            params_to_plot: Which parameters to plot.
            show_all: Plot all params if params_to_plot is None.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        if params_to_plot is None:
            params_to_plot = list(order_params.keys()) if show_all else ["m", "q", "eg"]

        for i, param in enumerate(params_to_plot):
            if param in order_params:
                color = self.style.colors[i % len(self.style.colors)]
                label = self.param_labels.get(param, param)
                self.plot_line(
                    ax,
                    alpha_values,
                    order_params[param],
                    label=label,
                    color=color,
                    marker="o",
                    **kwargs,
                )

        self.set_labels(ax, xlabel=r"$\alpha$", ylabel="Order Parameters")
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax

    def plot_vs_time(
        self,
        t_values: np.ndarray,
        order_params: Dict[str, np.ndarray],
        ax: Optional[Axes] = None,
        params_to_plot: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot order parameters vs time.

        Args:
            t_values: Time values.
            order_params: Order parameter trajectories.
            ax: Axes.
            params_to_plot: Parameters to plot.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        if params_to_plot is None:
            params_to_plot = list(order_params.keys())

        for i, param in enumerate(params_to_plot):
            if param in order_params:
                color = self.style.colors[i % len(self.style.colors)]
                label = self.param_labels.get(param, param)
                self.plot_line(
                    ax,
                    t_values,
                    order_params[param],
                    label=label,
                    color=color,
                    **kwargs,
                )

        self.set_labels(ax, xlabel=r"$t$", ylabel="Order Parameters")
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax

    def plot_with_std(
        self,
        x_values: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        use_fill: bool = True,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot mean with standard deviation.

        Args:
            x_values: X values.
            mean: Mean values.
            std: Standard deviation.
            ax: Axes.
            label: Label.
            color: Color.
            use_fill: Use fill_between instead of errorbars.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        color = color or self.style.colors[0]

        if use_fill:
            self.plot_fill_between(ax, x_values, mean, std, label=label, color=color, **kwargs)
        else:
            self.plot_errorbar(ax, x_values, mean, std, label=label, color=color, **kwargs)

        return fig, ax

    def plot_from_result(
        self,
        result: SimulationResult,
        param_indices: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot order parameters from SimulationResult.

        Args:
            result: Simulation result.
            param_indices: Mapping from param name to index in array.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        # Get x values based on theory type
        if result.theory_type.value == "replica":
            x_values = np.array(result.experiment_results["alpha_values"])
            xlabel = r"$\alpha$"
        else:
            x_values = np.array(result.experiment_results["t_values"])
            xlabel = r"$t$"

        # Get order parameters
        mean_data = np.array(result.experiment_results.get(
            "order_params_mean",
            result.experiment_results.get("trajectories_mean")
        ))
        std_data = np.array(result.experiment_results.get(
            "order_params_std",
            result.experiment_results.get("trajectories_std")
        ))

        # Default param indices
        if param_indices is None:
            param_indices = {"m": 0, "q": 1, "eg": 2}

        fig, ax = self.create_figure()

        for i, (name, idx) in enumerate(param_indices.items()):
            if idx < mean_data.shape[1]:
                color = self.style.colors[i % len(self.style.colors)]
                label = self.param_labels.get(name, name)

                self.plot_fill_between(
                    ax,
                    x_values,
                    mean_data[:, idx],
                    std_data[:, idx],
                    label=label,
                    color=color,
                    **kwargs,
                )

        self.set_labels(ax, xlabel=xlabel, ylabel="Order Parameters")
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax
