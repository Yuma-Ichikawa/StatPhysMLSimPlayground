"""Base plotter and utilities."""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class PlotStyle:
    """
    Plot style configuration.

    Attributes:
        figsize: Figure size (width, height).
        dpi: Resolution.
        font_size: Base font size.
        line_width: Default line width.
        marker_size: Default marker size.
        colors: Color palette.
        grid: Whether to show grid.
        grid_style: Grid line style.
        legend_loc: Legend location.

    """

    figsize: tuple[float, float] = (8, 6)
    dpi: int = 100
    font_size: int = 12
    line_width: float = 2.0
    marker_size: float = 8.0
    colors: list[str] | None = None
    grid: bool = True
    grid_style: str = "--"
    grid_alpha: float = 0.3
    legend_loc: str = "best"
    latex: bool = False

    def __post_init__(self):
        """Set default colors if not provided."""
        if self.colors is None:
            self.colors = [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Pink
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
            ]

    def apply(self) -> None:
        """Apply style to matplotlib."""
        plt.rcParams.update(
            {
                "figure.figsize": self.figsize,
                "figure.dpi": self.dpi,
                "font.size": self.font_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
                "axes.grid": self.grid,
                "grid.linestyle": self.grid_style,
                "grid.alpha": self.grid_alpha,
                "legend.loc": self.legend_loc,
            }
        )

        if self.latex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )


class Plotter:
    """
    Base class for plotting utilities.

    Provides common functionality for creating consistent
    visualizations of simulation results.
    """

    def __init__(self, style: PlotStyle | None = None):
        """
        Initialize Plotter.

        Args:
            style: Plot style configuration. Uses default if None.

        """
        self.style = style or PlotStyle()
        self.style.apply()

    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray]:
        """
        Create figure and axes.

        Args:
            nrows: Number of rows.
            ncols: Number of columns.
            figsize: Figure size (overrides style).
            **kwargs: Additional arguments for subplots.

        Returns:
            Tuple of (Figure, Axes or array of Axes).

        """
        figsize = figsize or self.style.figsize
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes

    def plot_line(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        label: str | None = None,
        color: str | None = None,
        linestyle: str = "-",
        marker: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot a line.

        Args:
            ax: Axes to plot on.
            x: X values.
            y: Y values.
            label: Line label.
            color: Line color.
            linestyle: Line style.
            marker: Marker style.
            **kwargs: Additional plot arguments.

        """
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=self.style.line_width,
            markersize=self.style.marker_size,
            **kwargs,
        )

    def plot_errorbar(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        label: str | None = None,
        color: str | None = None,
        marker: str = "o",
        capsize: float = 5,
        **kwargs: Any,
    ) -> None:
        """
        Plot with error bars.

        Args:
            ax: Axes to plot on.
            x: X values.
            y: Y values (mean).
            yerr: Y error (std).
            label: Label.
            color: Color.
            marker: Marker style.
            capsize: Error bar cap size.
            **kwargs: Additional arguments.

        """
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=label,
            color=color,
            marker=marker,
            capsize=capsize,
            linewidth=self.style.line_width,
            markersize=self.style.marker_size,
            **kwargs,
        )

    def plot_fill_between(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        label: str | None = None,
        color: str | None = None,
        alpha: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """
        Plot line with shaded error region.

        Args:
            ax: Axes.
            x: X values.
            y: Y values (mean).
            yerr: Y error (std).
            label: Label.
            color: Color.
            alpha: Fill transparency.
            **kwargs: Additional arguments.

        """
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=self.style.line_width,
            **kwargs,
        )
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=color,
            alpha=alpha,
        )

    def set_labels(
        self,
        ax: Axes,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Set axis labels and title."""
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

    def add_legend(
        self,
        ax: Axes,
        loc: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add legend to axes."""
        loc = loc or self.style.legend_loc
        ax.legend(loc=loc, **kwargs)

    def add_grid(self, ax: Axes) -> None:
        """Add grid to axes."""
        ax.grid(True, linestyle=self.style.grid_style, alpha=self.style.grid_alpha)

    def save_figure(
        self,
        fig: Figure,
        filepath: str,
        dpi: int | None = None,
        bbox_inches: str = "tight",
        **kwargs: Any,
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: Figure to save.
            filepath: Output path.
            dpi: Resolution.
            bbox_inches: Bounding box setting.
            **kwargs: Additional arguments.

        """
        dpi = dpi or self.style.dpi
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    def show(self) -> None:
        """Display all figures."""
        plt.show()

    def close_all(self) -> None:
        """Close all figures."""
        plt.close("all")
