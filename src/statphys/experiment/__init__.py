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
      Binder cumulant, participation ratio, specialization index).
    - TeacherStudentExperiment: sample-complexity sweeps (alpha = n/d),
      replica-resolved order-parameter sweeps, and online-SGD dynamics
      for arbitrary teacher-student pairs.
    - run_phase_diagram: 2D numerical phase diagrams
      (control parameter x alpha).
    - presets: ready-made interesting setups.

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
from statphys.experiment.observables import (
    binder_cumulant,
    function_order_params,
    participation_ratio,
    replica_overlaps,
    specialization_index,
    susceptibility,
)
from statphys.experiment.phase import PhaseDiagramResult, run_phase_diagram
from statphys.experiment.presets import PRESETS, get_preset
from statphys.experiment.protocol import ExperimentResult, TeacherStudentExperiment
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
    "PhaseDiagramResult",
    "run_phase_diagram",
    "PRESETS",
    "get_preset",
    "ARCHITECTURES",
    "build_architecture",
    "architecture_experiment",
]
