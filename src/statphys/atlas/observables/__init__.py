"""Mechanistic and thermodynamic observables for the Transformer atlas."""

from .attention import attention_geometry
from .functional import centered_functional_overlap, two_template_decomposition
from .heads import (
    head_specialization_metrics,
    latent_overlap_matrix,
    match_heads,
    permutation_invariant_head_spectrum,
)
from .interventions import intervention_loss_deltas
from .replica import functional_replica_distribution, functional_replica_overlaps
from .representation import correlation_length, position_correlation, representation_statistics
from .spectra import (
    effective_ov_matrix,
    effective_qk_matrix,
    matrix_spectrum,
    qk_ov_spectra,
    subspace_principal_angles,
)

__all__ = [
    "attention_geometry",
    "centered_functional_overlap",
    "correlation_length",
    "effective_ov_matrix",
    "effective_qk_matrix",
    "functional_replica_distribution",
    "functional_replica_overlaps",
    "head_specialization_metrics",
    "intervention_loss_deltas",
    "latent_overlap_matrix",
    "match_heads",
    "matrix_spectrum",
    "permutation_invariant_head_spectrum",
    "position_correlation",
    "qk_ov_spectra",
    "representation_statistics",
    "subspace_principal_angles",
    "two_template_decomposition",
]

