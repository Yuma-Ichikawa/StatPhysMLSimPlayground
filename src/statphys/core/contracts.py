"""Extension contracts for data, systems, and scientifically named observables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Batch:
    """A uniformly shaped sample returned by every supported learning system."""

    inputs: Tensor
    targets: Tensor
    mask: Tensor | None = None
    groups: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_ids: Tensor | None = None

    def __post_init__(self) -> None:
        if self.inputs.ndim == 0 or self.targets.ndim == 0:
            raise ValueError("inputs and targets must both have a batch dimension")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("inputs and targets must have equal batch size")
        for name, value in (
            ("mask", self.mask),
            ("groups", self.groups),
            ("sample_ids", self.sample_ids),
        ):
            if value is not None and value.shape[0] != self.inputs.shape[0]:
                raise ValueError(f"{name} must have the same batch size as inputs")


@dataclass(frozen=True)
class TaskSpec:
    """Human-readable semantics for a learning task."""

    name: str
    target_kind: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.target_kind.strip():
            raise ValueError("task name and target_kind must be non-empty")


@dataclass(frozen=True)
class ObservableSpec:
    """Semantic contract required before an observable enters a report.

    ``ensemble`` states the independent unit, preventing a training trajectory
    from being silently presented as a disorder ensemble. ``symmetry`` and
    ``null_model`` make the intended interpretation auditable.
    """

    name: str
    units: str
    ensemble: str
    interpretation: str
    symmetry: str = "none"
    normalization: str = "none"
    null_model: str | None = None
    valid_for: tuple[str, ...] = ()
    semantic_role: str = "diagnostic"
    quantity_kind: str = "scalar"
    intensive: bool = True
    independent_of_optimized_objective: bool = True
    requires_teacher_or_oracle: bool = False
    permutation_invariant: bool = True
    valid_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        required = {
            "name": self.name,
            "units": self.units,
            "ensemble": self.ensemble,
            "interpretation": self.interpretation,
            "semantic_role": self.semantic_role,
            "quantity_kind": self.quantity_kind,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"observable fields must be non-empty: {', '.join(missing)}")
        if self.ensemble in {"trajectory", "checkpoint", "time"} and self.name in {
            "binder_cumulant",
            "susceptibility",
            "seed_susceptibility",
            "disorder_binder",
        }:
            raise ValueError(
                "phase-ensemble observables require independent replicas; use "
                "trajectory_variance or temporal_kurtosis for checkpoint series"
            )
        if self.valid_range is not None and self.valid_range[0] > self.valid_range[1]:
            raise ValueError("valid_range must be ordered")


class LearningSystem(Protocol):
    """Minimal interface for adding a custom system without replacing a runner."""

    def sample(self, split: str, n: int, *, generator: torch.Generator) -> Batch:
        """Draw a batch using the named generator supplied by the runner."""
        ...

    def build_model(self, *, generator: torch.Generator) -> nn.Module:
        """Create a model using an initialization-only generator."""
        ...

    def task_spec(self) -> TaskSpec:
        """Describe target semantics for validation and reporting."""
        ...
