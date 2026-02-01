"""
Phase diagram visualization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib import cm

from statphys.vis.plotter import Plotter, PlotStyle


class PhaseDiagramPlotter(Plotter):
    """
    Plotter for phase diagrams.

    Creates visualizations of phase transitions in
    parameter space (alpha vs lambda, etc.).
    """

    def __init__(
        self,
        style: Optional[PlotStyle] = None,
        cmap: str = "viridis",
    ):
        """
        Initialize PhaseDiagramPlotter.

        Args:
            style: Plot style.
            cmap: Colormap for heatmaps.
        """
        super().__init__(style)
        self.cmap = cmap

    def plot_heatmap(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        ax: Optional[Axes] = None,
        xlabel: str = r"$\alpha$",
        ylabel: str = r"$\lambda$",
        zlabel: str = r"$E_g$",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D heatmap.

        Args:
            x_values: X grid values.
            y_values: Y grid values.
            z_values: Z values (2D array).
            ax: Axes.
            xlabel: X label.
            ylabel: Y label.
            zlabel: Colorbar label.
            vmin, vmax: Color range.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        # Create meshgrid if needed
        X, Y = np.meshgrid(x_values, y_values)

        # Plot heatmap
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(X, Y, z_values, cmap=self.cmap, norm=norm, **kwargs)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(zlabel)

        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)

        return fig, ax

    def plot_contour(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        ax: Optional[Axes] = None,
        levels: int = 10,
        xlabel: str = r"$\alpha$",
        ylabel: str = r"$\lambda$",
        zlabel: str = r"$E_g$",
        filled: bool = True,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot contour plot.

        Args:
            x_values: X grid values.
            y_values: Y grid values.
            z_values: Z values.
            ax: Axes.
            levels: Number of contour levels.
            xlabel, ylabel, zlabel: Labels.
            filled: Use filled contours.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        X, Y = np.meshgrid(x_values, y_values)

        if filled:
            cs = ax.contourf(X, Y, z_values, levels=levels, cmap=self.cmap, **kwargs)
        else:
            cs = ax.contour(X, Y, z_values, levels=levels, cmap=self.cmap, **kwargs)
            ax.clabel(cs, inline=True, fontsize=8)

        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label(zlabel)

        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)

        return fig, ax

    def plot_phase_boundary(
        self,
        x_values: np.ndarray,
        boundary_values: np.ndarray,
        ax: Optional[Axes] = None,
        label: str = "Phase boundary",
        fill_below: bool = True,
        fill_alpha: float = 0.3,
        fill_color: Optional[str] = None,
        xlabel: str = r"$\alpha$",
        ylabel: str = r"$\lambda_c$",
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot phase boundary line.

        Args:
            x_values: X values.
            boundary_values: Y values of boundary.
            ax: Axes.
            label: Boundary label.
            fill_below: Fill region below boundary.
            fill_alpha: Fill transparency.
            fill_color: Fill color.
            xlabel, ylabel: Labels.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        color = self.style.colors[0]
        fill_color = fill_color or color

        ax.plot(
            x_values,
            boundary_values,
            color=color,
            linewidth=self.style.line_width,
            label=label,
            **kwargs,
        )

        if fill_below:
            ax.fill_between(
                x_values,
                0,
                boundary_values,
                color=fill_color,
                alpha=fill_alpha,
            )

        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax

    def plot_order_param_diagram(
        self,
        alpha_values: np.ndarray,
        lambda_values: np.ndarray,
        order_params: Dict[str, np.ndarray],
        param: str = "m",
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot order parameter as function of two parameters.

        Args:
            alpha_values: Alpha grid.
            lambda_values: Lambda grid.
            order_params: Order params at each grid point.
            param: Which parameter to plot.
            ax: Axes.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        z_values = order_params[param]

        return self.plot_heatmap(
            alpha_values,
            lambda_values,
            z_values,
            ax=ax,
            xlabel=r"$\alpha$",
            ylabel=r"$\lambda$",
            zlabel=f"${param}$",
            **kwargs,
        )

    def plot_critical_line(
        self,
        critical_alpha: np.ndarray,
        critical_lambda: np.ndarray,
        ax: Optional[Axes] = None,
        label: str = "Critical line",
        regions: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """
        Plot critical line separating phases.

        Args:
            critical_alpha: Alpha values on critical line.
            critical_lambda: Lambda values on critical line.
            ax: Axes.
            label: Line label.
            regions: Dictionary mapping region names to positions for annotation.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (Figure, Axes).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        ax.plot(
            critical_alpha,
            critical_lambda,
            color=self.style.colors[0],
            linewidth=self.style.line_width * 1.5,
            label=label,
            **kwargs,
        )

        # Add region annotations if provided
        if regions:
            for region_name, (x_pos, y_pos) in regions.items():
                ax.annotate(
                    region_name,
                    xy=(x_pos, y_pos),
                    fontsize=self.style.font_size,
                    ha="center",
                )

        self.set_labels(ax, xlabel=r"$\alpha$", ylabel=r"$\lambda$")
        self.add_legend(ax)
        self.add_grid(ax)

        return fig, ax
