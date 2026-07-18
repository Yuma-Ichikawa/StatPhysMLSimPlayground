"""Transformer thermodynamics phase-atlas research framework.

The public surface is intentionally small; data generators, architecture
ladders, observables, statistical analysis, and Slurm orchestration live in
separate subpackages so third-party studies can reuse them independently.
"""

from .schema import (
    ArchitectureStage,
    DataStage,
    InitStrategy,
    ObservableSpec,
    PhaseCard,
    ResourceSpec,
    RunSpec,
    ScalingPath,
    SeedPlan,
    TrainingSpec,
)

__all__ = [
    "ArchitectureStage",
    "DataStage",
    "InitStrategy",
    "ObservableSpec",
    "PhaseCard",
    "ResourceSpec",
    "RunSpec",
    "ScalingPath",
    "SeedPlan",
    "TrainingSpec",
]

