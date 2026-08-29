"""Canonical scientific ontology shared by every StatPhys subsystem."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class Fidelity(_ValueEnum):
    """How directly a system represents an empirical learning endpoint."""

    ANALYTIC_ANCHOR = "analytic_anchor"
    REDUCED_DERIVED_MODEL = "reduced_derived_model"
    PHENOMENOLOGICAL_GENERATOR = "phenomenological_generator"
    TRAINABLE_SYNTHETIC = "trainable_synthetic"
    SEMI_NATURAL = "semi_natural"
    NATURAL_DATA = "natural_data"
    PRETRAINED_ENDPOINT = "pretrained_endpoint"


class TheoryStatus(_ValueEnum):
    """Epistemic status of a displayed theory or reference curve."""

    EXACT_FINITE = "exact_finite"
    ASYMPTOTICALLY_EXACT = "asymptotically_exact"
    STATE_EVOLUTION_OR_DMFT = "state_evolution_or_dmft"
    MEAN_FIELD_CLOSURE = "mean_field_closure"
    CONTROLLED_APPROXIMATION = "controlled_approximation"
    CALIBRATED_SURROGATE = "calibrated_surrogate"
    PHENOMENOLOGICAL = "phenomenological"
    SYNTHETIC_GROUND_TRUTH = "synthetic_ground_truth"
    EMPIRICAL_ONLY = "empirical_only"


class Outcome(_ValueEnum):
    """Continuation outcome, kept separate from evidence and artifact validity."""

    PRESERVED = "preserved"
    RENORMALIZED = "renormalized"
    ROUNDED = "rounded"
    SPLIT = "split"
    MERGED = "merged"
    NEW_REGIME = "new_regime"
    HYSTERETIC = "hysteretic"
    PATH_DEPENDENT = "path_dependent"
    STATISTICAL_COMPUTATIONAL_SEPARATION = "statistical_computational_separation"
    SEMANTIC_FAILURE = "semantic_failure"
    CENSORED_NO_CROSSING = "censored_no_crossing"
    UNRESOLVED = "unresolved"
    NOT_COMPARABLE = "not_comparable"


class PhenomenonType(_ValueEnum):
    THERMODYNAMIC_TRANSITION = "thermodynamic_phase_transition"
    NON_EQUILIBRIUM_TRANSITION = "non_equilibrium_dynamical_transition"
    ALGORITHMIC_THRESHOLD = "algorithmic_threshold"
    STATISTICAL_THRESHOLD = "statistical_threshold"
    BIFURCATION = "bifurcation"
    CROSSOVER = "crossover"
    OPERATIONAL_THRESHOLD = "operational_threshold"
    SEMANTIC_TRANSITION = "semantic_transition"


_SCALE_FIELDS = (
    "data_size",
    "input_dimension",
    "width",
    "depth",
    "context_length",
    "parameter_count",
    "horizon",
    "population",
    "resolution",
    "compute_flops",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ScaleSpec:
    """Vector-valued system scale and an explicit finite-size limit path."""

    data_size: int | None = None
    input_dimension: int | None = None
    width: int | None = None
    depth: int | None = None
    context_length: int | None = None
    parameter_count: int | None = None
    horizon: int | None = None
    population: int | None = None
    resolution: int | None = None
    compute_flops: float | None = None
    finite_size_coordinate: str = "width"
    scaling_path: Mapping[str, str] = field(default_factory=dict)
    coordinates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _SCALE_FIELDS:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"scale coordinate {name} must be finite and positive")
        coordinate = self.finite_size_coordinate.strip()
        if not coordinate:
            raise ValueError("finite_size_coordinate must be non-empty")
        object.__setattr__(self, "finite_size_coordinate", coordinate)
        scaling_path = {str(key): str(value) for key, value in self.scaling_path.items()}
        coordinates = {str(key): float(value) for key, value in self.coordinates.items()}
        if any(not math.isfinite(value) or value <= 0 for value in coordinates.values()):
            raise ValueError("custom scale coordinates must be finite and positive")
        object.__setattr__(self, "scaling_path", MappingProxyType(scaling_path))
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))
        if self.finite_size_value is None:
            raise ValueError(
                "finite_size_coordinate must identify a populated scale field or custom coordinate"
            )

    @property
    def finite_size_value(self) -> float | None:
        if self.finite_size_coordinate in _SCALE_FIELDS:
            value = getattr(self, self.finite_size_coordinate)
            return None if value is None else float(value)
        return self.coordinates.get(self.finite_size_coordinate)

    @classmethod
    def legacy(cls, size: int) -> ScaleSpec:
        """Represent the old scalar-size API without guessing its physical meaning."""
        return cls(finite_size_coordinate="legacy_size", coordinates={"legacy_size": float(size)})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleSpec:
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in _SCALE_FIELDS},
            "finite_size_coordinate": self.finite_size_coordinate,
            "scaling_path": dict(self.scaling_path),
            "coordinates": dict(self.coordinates),
        }


@dataclass(frozen=True)
class TheorySpec:
    """Validity metadata required for every theory-like curve."""

    status: TheoryStatus
    limit: str
    assumptions: tuple[str, ...] = ()
    validity_domain: Mapping[str, Any] = field(default_factory=dict)
    stability_checked: bool = False
    residual: float | None = None
    branch: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, TheoryStatus):
            object.__setattr__(self, "status", TheoryStatus(str(self.status)))
        if not self.limit.strip():
            raise ValueError("theory limit must be explicit")
        if self.residual is not None and not math.isfinite(self.residual):
            raise ValueError("theory residual must be finite")
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        object.__setattr__(self, "references", tuple(str(item) for item in self.references))
        object.__setattr__(self, "validity_domain", MappingProxyType(dict(self.validity_domain)))

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _plain(self.__dict__))


@dataclass(frozen=True)
class ScientificSpec:
    """Identity of a scientific system, independent of protocol and hardware."""

    domain: str
    task: str
    state_object: str
    fidelity: Fidelity
    teacher_or_reference: str
    data_or_environment: str
    model_or_interaction: str
    objective_or_feedback: str
    dynamics: str
    control_parameters: tuple[str, ...]
    scale: ScaleSpec
    deformation_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "domain",
            "task",
            "state_object",
            "teacher_or_reference",
            "data_or_environment",
            "model_or_interaction",
            "objective_or_feedback",
            "dynamics",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.fidelity, Fidelity):
            object.__setattr__(self, "fidelity", Fidelity(str(self.fidelity)))
        object.__setattr__(self, "control_parameters", tuple(self.control_parameters))
        object.__setattr__(self, "deformation_axes", tuple(self.deformation_axes))

    def to_dict(self) -> dict[str, Any]:
        return {
            **_plain(self.__dict__),
            "scale": self.scale.to_dict(),
        }


@dataclass(frozen=True)
class ObservableValue:
    """A value plus the semantics needed for valid cross-domain comparison."""

    name: str
    semantic_role: str
    quantity_kind: str
    value: float
    units: str
    normalization: str
    ensemble: str
    null_model: str
    valid_range: tuple[float, float] | None = None
    status: str = "valid"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValueError("observable value must be finite")
        for name in (
            "name",
            "semantic_role",
            "quantity_kind",
            "units",
            "normalization",
            "ensemble",
            "null_model",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.valid_range is not None:
            low, high = self.valid_range
            if low > high or not low <= self.value <= high:
                raise ValueError("observable value lies outside its registered range")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _plain(self.__dict__))


@dataclass(frozen=True)
class PhaseCardV3:
    """Domain-independent PhaseCard used by software, tables, and reports."""

    identity: Mapping[str, Any]
    system: Mapping[str, Any]
    deformation: Mapping[str, Any]
    scale: ScaleSpec
    disorder: Mapping[str, Any]
    observables: Mapping[str, Any]
    theory: TheorySpec
    evidence: Mapping[str, Any]
    result: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "3.0"

    def __post_init__(self) -> None:
        required_identity = {"domain", "task", "state_object", "fidelity"}
        missing = required_identity - set(self.identity)
        if missing:
            raise ValueError(f"PhaseCard identity is missing: {', '.join(sorted(missing))}")
        if not self.observables.get("primary"):
            raise ValueError("PhaseCard requires at least one primary observable")
        for name in (
            "identity",
            "system",
            "deformation",
            "disorder",
            "observables",
            "evidence",
            "result",
            "artifacts",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": _plain(self.identity),
            "system": _plain(self.system),
            "deformation": _plain(self.deformation),
            "scale": self.scale.to_dict(),
            "disorder": _plain(self.disorder),
            "observables": _plain(self.observables),
            "theory": self.theory.to_dict(),
            "evidence": _plain(self.evidence),
            "result": _plain(self.result),
            "artifacts": _plain(self.artifacts),
        }


def structural_features(values: Sequence[ObservableValue]) -> dict[str, bool]:
    """Compare domain observables by structural behavior, never raw magnitude."""
    return {
        "multiplicity_increases": any(
            value.semantic_role == "multiplicity" and value.value > 0 for value in values
        ),
        "has_semantic_validation": all(value.status == "valid" for value in values),
        "has_registered_null": all(bool(value.null_model) for value in values),
    }


__all__ = [
    "Fidelity",
    "ObservableValue",
    "Outcome",
    "PhaseCardV3",
    "PhenomenonType",
    "ScaleSpec",
    "ScientificSpec",
    "TheorySpec",
    "TheoryStatus",
    "structural_features",
]
