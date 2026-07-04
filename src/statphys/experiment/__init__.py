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
    - TeacherStudentExperiment: sample-complexity sweeps (alpha = n/d)
      and online-SGD dynamics for arbitrary teacher-student pairs.
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
from statphys.experiment.presets import PRESETS, get_preset
from statphys.experiment.protocol import ExperimentResult, TeacherStudentExperiment
from statphys.experiment.teacher import Teacher, init_weights_

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
    "PRESETS",
    "get_preset",
]
