"""
Visualization module for simulation results.

This module provides:
- Plotter: General plotting utilities
- OrderParamPlotter: Plot order parameter trajectories
- ComparisonPlotter: Compare theory vs experiment
- PhaseDiagramPlotter: Phase diagram visualization (+ compute_phase_grid)
- DynamicsPlotter: (m, q) phase portraits, ODE flow fields, nullclines
- OverlapMatrixPlotter: M/Q/R overlap-matrix heatmaps, specialization
- SweepPlotter: Parameter sweeps, theory-vs-experiment diagnostics
- Animations: learning curves, phase-plane motion, overlap matrices (GIF/MP4)
- Default plotting functions for publication-quality figures

Requirements:
    This module requires optional visualization dependencies.
    Install them with: pip install statphys-ml[vis]

Example:
    >>> from statphys.vis import plot_from_replica_results
    >>> plot_from_replica_results(results, plot_type="all")

    >>> from statphys.vis import plot_generalization_error_alpha
    >>> plot_generalization_error_alpha(alpha, eg_mean, eg_std)

"""

try:
    import matplotlib  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Visualization module requires matplotlib. " "Install with: pip install statphys-ml[vis]"
    ) from e

from statphys.vis.animation import (
    animate_learning_curve,
    animate_overlap_matrix,
    animate_phase_plane,
    save_animation,
)
from statphys.vis.comparison import ComparisonPlotter
from statphys.vis.dashboard import plot_order_parameter_dashboard
from statphys.vis.dynamics import DynamicsPlotter
from statphys.vis.overlap_matrix import OverlapMatrixPlotter
from statphys.vis.sweep import SweepPlotter
from statphys.vis.default_plots import (
    COLORS,
    DEFAULT_FIGSIZE,
    LINE_STYLES,
    MARKERS,
    PAPER_STYLE,
    apply_paper_style,
    plot_from_online_results,
    plot_from_replica_results,
    plot_generalization_error_alpha,
    plot_generalization_error_time,
    plot_order_params_alpha,
    plot_order_params_time,
)
from statphys.vis.order_params import OrderParamPlotter
from statphys.vis.phase_diagram import PhaseDiagramPlotter, compute_phase_grid
from statphys.vis.plotter import PlotStyle, Plotter

__all__ = [
    # Base classes
    "Plotter",
    "PlotStyle",
    "OrderParamPlotter",
    "ComparisonPlotter",
    "PhaseDiagramPlotter",
    "DynamicsPlotter",
    "OverlapMatrixPlotter",
    "SweepPlotter",
    "compute_phase_grid",
    # Animations
    "animate_learning_curve",
    "animate_phase_plane",
    "animate_overlap_matrix",
    "save_animation",
    # Dashboards
    "plot_order_parameter_dashboard",
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
