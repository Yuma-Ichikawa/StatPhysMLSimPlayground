"""Protocol identity, disorder hierarchy, and pairing declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class DisorderRole(str, Enum):
    TEACHER_OR_ENVIRONMENT = "teacher_or_environment_disorder"
    DATA = "data_disorder"
    INITIALIZATION = "initialization"
    MINIBATCH = "minibatch_order"
    DROPOUT = "dropout"
    DIFFUSION_NOISE = "diffusion_noise"
    ROLLOUT = "rollout"
    EVALUATION = "evaluation"
    INTERVENTION = "intervention"
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True)
class DisorderSpec:
    role: DisorderRole
    level: str
    paired_across_controls: bool = False
    paired_across_models: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, DisorderRole):
            object.__setattr__(self, "role", DisorderRole(str(self.role)))
        if self.level not in {"outer", "inner"}:
            raise ValueError("disorder level must be outer or inner")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "level": self.level,
            "paired_across_controls": self.paired_across_controls,
            "paired_across_models": self.paired_across_models,
        }


@dataclass(frozen=True)
class ProtocolSpec:
    """Measurement plan, independent of the scientific system and execution."""

    disorders: tuple[DisorderSpec, ...]
    outer_seed_ids: tuple[int, ...]
    inner_replicates: int = 1
    observables: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    holdout_rule: str = "none"
    censoring_rule: str = "explicit"
    statistical_plan: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.outer_seed_ids)
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            raise ValueError("outer_seed_ids must be distinct non-negative integers")
        if self.inner_replicates < 1:
            raise ValueError("inner_replicates must be positive")
        roles = [item.role for item in self.disorders]
        if len(roles) != len(set(roles)):
            raise ValueError("disorder roles must be unique")
        object.__setattr__(self, "outer_seed_ids", seeds)
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "statistical_plan", MappingProxyType(dict(self.statistical_plan)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "disorders": [item.to_dict() for item in self.disorders],
            "outer_seed_ids": list(self.outer_seed_ids),
            "inner_replicates": self.inner_replicates,
            "observables": list(self.observables),
            "interventions": list(self.interventions),
            "holdout_rule": self.holdout_rule,
            "censoring_rule": self.censoring_rule,
            "statistical_plan": dict(self.statistical_plan),
        }


__all__ = ["DisorderRole", "DisorderSpec", "ProtocolSpec"]
