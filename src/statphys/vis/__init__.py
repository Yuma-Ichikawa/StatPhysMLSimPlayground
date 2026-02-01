"""
Visualization module for simulation results.

This module provides:
- Plotter: General plotting utilities
- OrderParamPlotter: Plot order parameter trajectories
- ComparisonPlotter: Compare theory vs experiment
- PhaseDiagramPlotter: Phase diagram visualization
- Default plotting functions for publication-quality figures

Example:
    >>> from statphys.vis import plot_from_replica_results
    >>> plot_from_replica_results(results, plot_type="all")

    >>> from statphys.vis import plot_generalization_error_alpha
    >>> plot_generalization_error_alpha(alpha, eg_mean, eg_std)
"""

from statphys.vis.plotter import Plotter, PlotStyle
from statphys.vis.order_params import OrderParamPlotter
from statphys.vis.comparison import ComparisonPlotter
from statphys.vis.phase_diagram import PhaseDiagramPlotter
from statphys.vis.default_plots import (
    apply_paper_style,
    plot_generalization_error_alpha,
    plot_order_params_alpha,
    plot_generalization_error_time,
    plot_order_params_time,
    plot_from_replica_results,
    plot_from_online_results,
    PAPER_STYLE,
    DEFAULT_FIGSIZE,
    LINE_STYLES,
    MARKERS,
    COLORS,
)

__all__ = [
    # Base classes
    "Plotter",
    "PlotStyle",
    "OrderParamPlotter",
    "ComparisonPlotter",
    "PhaseDiagramPlotter",
    # Default plotting functions
    "apply_paper_style",
    "plot_generalization_error_alpha",
    "plot_order_params_alpha",
    "plot_generalization_error_time",
    "plot_order_params_time",
    "plot_from_replica_results",
    "plot_from_online_results",
    # Style constants
    "PAPER_STYLE",
    "DEFAULT_FIGSIZE",
    "LINE_STYLES",
    "MARKERS",
    "COLORS",
]
