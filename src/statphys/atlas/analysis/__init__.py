"""Statistical analysis utilities for the Transformer thermodynamics atlas."""

from .discovery import adjacent_histogram_js, adjacent_js_divergence, change_point_candidates
from .phase import (
    binder_crossing,
    classify_phase,
    classify_transition_evidence,
    estimate_transition_boundary,
    finite_difference_response,
)
from .scaling import (
    bootstrap_finite_size_scaling,
    data_collapse_score,
    fit_finite_size_scaling_grid,
)
from .statistics import (
    binder_cumulant,
    binder_summary,
    hierarchical_bootstrap,
    nested_estimator,
    order_parameter_histogram,
    rate_function,
    susceptibility,
)

__all__ = [
    "adjacent_histogram_js",
    "adjacent_js_divergence",
    "binder_crossing",
    "binder_cumulant",
    "binder_summary",
    "bootstrap_finite_size_scaling",
    "change_point_candidates",
    "classify_phase",
    "classify_transition_evidence",
    "data_collapse_score",
    "estimate_transition_boundary",
    "finite_difference_response",
    "fit_finite_size_scaling_grid",
    "hierarchical_bootstrap",
    "nested_estimator",
    "order_parameter_histogram",
    "rate_function",
    "susceptibility",
]

