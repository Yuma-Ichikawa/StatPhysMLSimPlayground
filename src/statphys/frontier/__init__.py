"""
statphys.frontier: modern learning paradigms as teacher-student physics.

Numerical experiments on settings *beyond* current exact theory --
minimal but faithful teacher-student reductions of SFT, RLHF, weak
supervision, synthetic-data loops, and in-context learning -- measured
with the same order parameters (function-space overlaps, generalization
error, phase diagrams) used in the exactly solvable settings elsewhere
in this package. The design goal is that when the theory side (replica
/ DMFT) reaches these paradigms, the numerical phenomenology is already
mapped out here.

Modules
-------
- common:         shared model factories, training loop, overlaps
- sft:            fine-tuning / transfer / catastrophic forgetting
- rlhf:           reward-model overoptimization (Goodhart transition)
- weak_to_strong: weak supervisor -> strong student generalization
- collapse:       model collapse under recursive synthetic data
- icl:            emergence of in-context learning vs task diversity
- studies:        ready-made sweeps + figures (CLI: `statphys study <name>`)

Quick start:

    >>> from statphys.frontier import run_overoptimization
    >>> res = run_overoptimization(alpha_r=8.0)
    >>> res["kl_star"]  # optimization budget where gold reward peaks

See docs/frontier.md for the mathematical definitions, order
parameters, and result gallery of every setting.
"""

from statphys.frontier.collapse import run_collapse, sweep_collapse
from statphys.frontier.common import (
    correlated_teacher,
    gaussian_sampler,
    mlp,
    model_overlap,
    output_overlap,
    train_regression,
)
from statphys.frontier.icl import (
    ICLTransformer,
    make_icl_batch,
    ridge_predictor,
    run_icl,
    sweep_icl,
)
from statphys.frontier.rlhf import (
    GoldReward,
    bon_kl,
    run_overoptimization,
    sweep_overoptimization,
    train_reward_model,
)
from statphys.frontier.sft import run_finetune, sweep_sft_phase_diagram
from statphys.frontier.studies import FRONTIER_STUDIES
from statphys.frontier.taxonomy import PARADIGM_PROBES, run_taxonomy, taxonomy_markdown
from statphys.frontier.teachers import TEACHER_TAXONOMY, TeacherSpec, make_teacher
from statphys.frontier.weak_to_strong import run_weak_to_strong, sweep_weak_to_strong

__all__ = [
    "FRONTIER_STUDIES",
    "GoldReward",
    "ICLTransformer",
    "PARADIGM_PROBES",
    "TEACHER_TAXONOMY",
    "TeacherSpec",
    "bon_kl",
    "correlated_teacher",
    "gaussian_sampler",
    "make_teacher",
    "run_taxonomy",
    "taxonomy_markdown",
    "make_icl_batch",
    "mlp",
    "model_overlap",
    "output_overlap",
    "ridge_predictor",
    "run_collapse",
    "run_finetune",
    "run_icl",
    "run_overoptimization",
    "run_weak_to_strong",
    "sweep_collapse",
    "sweep_icl",
    "sweep_overoptimization",
    "sweep_sft_phase_diagram",
    "sweep_weak_to_strong",
    "train_regression",
    "train_reward_model",
]
