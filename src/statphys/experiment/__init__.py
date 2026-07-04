"""
General teacher-student numerical experiments.

This subpackage provides a *theory-free* experimental framework: any
PyTorch module (MLPs, transformers, LLM-style blocks, ...) can be used
as a teacher and/or student, and phase-transition-like phenomena can be
observed purely numerically — no replica/ODE solution required.

Components:
    - Teacher: wraps any nn.Module or callable as a data-generating teacher,
      with rich weight-initialization strategies (random, sparse, low-rank,
      orthogonal, power-law, binary, ...).
    - TeacherStudentDataset: generates (x, y) pairs from a teacher with
      configurable input distributions and label noise.
    - Metrics: model-agnostic observables (test error, weight overlap
      when shapes match, representation similarity / CKA).
    - Observables: statistical-physics order parameters in function space
      (magnetization m_hat, replica overlap q_ab, susceptibility chi,
      Binder cumulant, participation ratio, specialization index,
      subspace overlap for multi-index models, weight movement for the
      lazy/rich feature-learning diagnostic).
    - GaussianMixtureDataset / bayes_error: a generative (not
      discriminative) classification setting with an exact closed-form
      generalization error, used to validate the eps_g bookkeeping.
    - TeacherStudentExperiment: sample-complexity sweeps (alpha = n/d),
      replica-resolved order-parameter sweeps, epoch-resolved training
      dynamics, and online-SGD dynamics for arbitrary teacher-student
      pairs (optionally with a custom generative `dataset`).
    - run_phase_diagram: 2D numerical phase diagrams
      (control parameter x alpha).
    - presets: ready-made interesting setups, including realistic/modern
      settings (hidden-manifold inputs, a tiny causal transformer,
      multi-index models, Gaussian-mixture classification, LoRA-style
      low-rank fine-tuning).

Example:
    >>> import torch.nn as nn
    >>> from statphys.experiment import Teacher, TeacherStudentExperiment
    >>>
    >>> teacher = Teacher(nn.Sequential(nn.Linear(100, 32), nn.ReLU(), nn.Linear(32, 1)),
    ...                   init="orthogonal")
    >>> exp = TeacherStudentExperiment(
    ...     teacher=teacher,
    ...     student_factory=lambda: nn.Sequential(
    ...         nn.Linear(100, 32), nn.ReLU(), nn.Linear(32, 1)),
    ... )
    >>> result = exp.run_sample_complexity(alphas=[0.5, 1, 2, 4, 8], n_seeds=3)
    >>> result.plot()

"""

from statphys.experiment.dataset import TeacherStudentDataset
from statphys.experiment.metrics import (
    linear_cka,
    representation_similarity,
    test_error,
    weight_overlap,
)
from statphys.experiment.mixture import GaussianMixtureDataset, bayes_error
from statphys.experiment.observables import (
    binder_cumulant,
    function_order_params,
    generalization_error_decomposition,
    participation_ratio,
    replica_overlaps,
    specialization_index,
    subspace_overlap,
    susceptibility,
    vector_overlap,
)
from statphys.experiment.phase import PhaseDiagramResult, run_phase_diagram
from statphys.experiment.presets import PRESETS, get_preset
from statphys.experiment.protocol import ExperimentResult, TeacherStudentExperiment
from statphys.experiment.studies import STUDIES, run_study
from statphys.experiment.teacher import Teacher, init_weights_
from statphys.experiment.zoo import ARCHITECTURES, architecture_experiment, build_architecture

__all__ = [
    "Teacher",
    "init_weights_",
    "TeacherStudentDataset",
    "TeacherStudentExperiment",
    "ExperimentResult",
    "test_error",
    "weight_overlap",
    "linear_cka",
    "representation_similarity",
    "function_order_params",
    "replica_overlaps",
    "susceptibility",
    "binder_cumulant",
    "participation_ratio",
    "specialization_index",
    "subspace_overlap",
    "vector_overlap",
    "generalization_error_decomposition",
    "GaussianMixtureDataset",
    "bayes_error",
    "PhaseDiagramResult",
    "run_phase_diagram",
    "PRESETS",
    "get_preset",
    "STUDIES",
    "run_study",
    "ARCHITECTURES",
    "build_architecture",
    "architecture_experiment",
]
