"""Immutable, serializable experiment schema for the Transformer phase atlas.

The schema is deliberately dependency-free. TOML sweeps, local runners and
Slurm workers all pass the same canonical JSON representation, so a run id is
a content hash rather than a mutable experiment name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping

import json
import math


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ArchitectureStage(_ValueEnum):
    M0_TIED_LOW_RANK = "m0_tied_low_rank"
    M1_UNTIED_LOW_RANK = "m1_untied_low_rank"
    M2_FULL_QKVO = "m2_full_qkvo"
    M3_MULTI_HEAD = "m3_multi_head"
    M4_RESIDUAL_NORM = "m4_residual_norm"
    M5_MLP = "m5_mlp"
    M6_CAUSAL_ROPE = "m6_causal_rope"
    M7_DEEP = "m7_deep"
    M8_AUTOREGRESSIVE = "m8_autoregressive"


class DataStage(_ValueEnum):
    D0_GAUSSIAN = "d0_gaussian"
    D1_ELLIPTICAL = "d1_elliptical"
    D2_HEAVY_TAILED = "d2_heavy_tailed"
    D3_FINITE_VOCABULARY = "d3_finite_vocabulary"
    D4_HMM = "d4_hmm"
    D5_GRAMMAR = "d5_grammar"


class InitStrategy(_ValueEnum):
    RANDOM = "random"
    SEMANTIC = "semantic"
    POSITIONAL = "positional"
    MIXED = "mixed"
    INTERPOLATED = "interpolated"


class Precision(_ValueEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"


class OptimizerName(_ValueEnum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAM = "adam"
    ADAMW = "adamw"


def _enum(value: Any, enum_type: type[_ValueEnum]) -> _ValueEnum:
    return value if isinstance(value, enum_type) else enum_type(str(value).strip().lower())


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ScalingPath:
    d_model: int = 64
    sequence_length: int = 2
    n_heads: int = 1
    head_dim: int | None = None
    n_layers: int = 1
    ff_dim: int | None = None
    teacher_rank: int = 1
    vocabulary_size: int = 32
    sample_coefficient: float = 1.0
    sample_exponent: float = 1.0

    def __post_init__(self) -> None:
        for name in ("d_model", "sequence_length", "n_heads", "n_layers", "teacher_rank"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.teacher_rank > self.d_model:
            raise ValueError("teacher_rank cannot exceed d_model")
        if self.head_dim is not None and self.head_dim < 1:
            raise ValueError("head_dim must be positive when supplied")
        if self.ff_dim is not None and self.ff_dim < 1:
            raise ValueError("ff_dim must be positive when supplied")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        _positive("sample_coefficient", float(self.sample_coefficient))
        _positive("sample_exponent", float(self.sample_exponent))

    @property
    def n_train(self) -> int:
        return max(1, int(round(self.sample_coefficient * self.d_model**self.sample_exponent)))

    @property
    def alpha(self) -> float:
        return self.n_train / float(self.d_model)

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)


@dataclass(frozen=True)
class PhaseCard:
    architecture: ArchitectureStage = ArchitectureStage.M0_TIED_LOW_RANK
    data: DataStage = DataStage.D0_GAUSSIAN
    scaling: ScalingPath = field(default_factory=ScalingPath)
    positional_mixture: float | None = None
    semantic_mixture: float | None = None
    input_noise: float = 0.5
    regularization: float = 0.0
    temperature: float = 1.0
    norm: str = "pre"
    covariance_condition: float = 8.0
    tail_degrees_freedom: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", _enum(self.architecture, ArchitectureStage))
        object.__setattr__(self, "data", _enum(self.data, DataStage))
        if isinstance(self.scaling, Mapping):
            object.__setattr__(self, "scaling", ScalingPath(**dict(self.scaling)))
        positional = self.positional_mixture
        semantic = self.semantic_mixture
        if positional is None and semantic is None:
            positional, semantic = 0.3, 0.7
        elif positional is None:
            positional = 1.0 - float(semantic)
        elif semantic is None:
            semantic = 1.0 - float(positional)
        positional, semantic = float(positional), float(semantic)
        if not 0.0 <= positional <= 1.0 or not 0.0 <= semantic <= 1.0:
            raise ValueError("mixture weights must lie in [0, 1]")
        if not math.isclose(positional + semantic, 1.0, abs_tol=1e-9):
            raise ValueError("positional_mixture and semantic_mixture must sum to one")
        object.__setattr__(self, "positional_mixture", positional)
        object.__setattr__(self, "semantic_mixture", semantic)
        _positive("input_noise", float(self.input_noise))
        if self.regularization < 0 or not math.isfinite(self.regularization):
            raise ValueError("regularization must be finite and non-negative")
        _positive("temperature", float(self.temperature))
        normalized_norm = self.norm.strip().lower()
        if normalized_norm not in {"pre", "post"}:
            raise ValueError("norm must be 'pre' or 'post'")
        object.__setattr__(self, "norm", normalized_norm)
        if self.covariance_condition < 1:
            raise ValueError("covariance_condition must be at least one")
        if self.tail_degrees_freedom <= 2:
            raise ValueError("tail_degrees_freedom must exceed two")

    @property
    def positional_strength(self) -> float:
        return float(self.positional_mixture)

    @property
    def semantic_strength(self) -> float:
        return float(self.semantic_mixture)

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)


@dataclass(frozen=True)
class TrainingSpec:
    optimizer: OptimizerName = OptimizerName.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    batch_size: int | None = None
    max_steps: int = 2000
    min_steps: int = 0
    checkpoint_interval: int = 200
    log_interval: int = 20
    patience: int = 400
    convergence_rtol: float = 1e-7
    gradient_clip: float | None = None
    precision: Precision = Precision.FLOAT32
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimizer", _enum(self.optimizer, OptimizerName))
        object.__setattr__(self, "precision", _enum(self.precision, Precision))
        _positive("learning_rate", float(self.learning_rate))
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must lie in [0, 1)")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive when supplied")
        if self.max_steps < 1 or not 0 <= self.min_steps <= self.max_steps:
            raise ValueError("require 0 <= min_steps <= max_steps and max_steps > 0")
        for name in ("checkpoint_interval", "log_interval", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.convergence_rtol < 0:
            raise ValueError("convergence_rtol must be non-negative")
        if self.gradient_clip is not None:
            _positive("gradient_clip", float(self.gradient_clip))

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)


@dataclass(frozen=True)
class ObservableSpec:
    heldout_size: int = 4096
    trajectory_interval: int = 20
    spectrum_interval: int = 200
    intervention_interval: int = 2000
    save_predictions: bool = False
    save_attention: bool = False

    def __post_init__(self) -> None:
        for name in ("heldout_size", "trajectory_interval", "spectrum_interval", "intervention_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)


@dataclass(frozen=True)
class ResourceSpec:
    device: str = "auto"
    gpus: int = 1
    cpus: int = 4
    memory_gb: int = 32

    def __post_init__(self) -> None:
        if self.gpus < 0 or self.cpus < 1 or self.memory_gb < 1:
            raise ValueError("resource counts must be non-negative and CPU/memory positive")

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)


_SEED_STREAMS = (
    "teacher",
    "data",
    "initialization",
    "minibatch",
    "heldout",
    "intervention",
)


@dataclass(frozen=True)
class SeedPlan:
    root: int = 0
    replica: int = 0
    teacher: int = 0
    data: int = 0
    initialization: int = 0
    minibatch: int = 0
    dropout: int = 0

    def __post_init__(self) -> None:
        if self.root < 0 or self.replica < 0:
            raise ValueError("root and replica seeds must be non-negative")

    def seed(self, stream: str) -> int:
        if not stream:
            raise ValueError("seed stream cannot be empty")
        digest = sha256(f"{self.root}:{self.replica}:{stream}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)

    def resolved(self) -> dict[str, int]:
        resolved = {name: self.seed(name) for name in _SEED_STREAMS}
        explicit = {
            "teacher": self.teacher,
            "data": self.data,
            "initialization": self.initialization,
            "minibatch": self.minibatch,
        }
        resolved.update({name: int(value) for name, value in explicit.items() if value != 0})
        resolved["dropout"] = int(self.dropout) if self.dropout != 0 else (self.seed("minibatch") ^ self.seed("intervention"))
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "replica": self.replica, "teacher": self.teacher, "data": self.data, "initialization": self.initialization, "minibatch": self.minibatch, "dropout": self.dropout}


@dataclass(frozen=True)
class RunSpec:
    experiment: str = "transformer_phase_atlas"
    phase: PhaseCard = field(default_factory=PhaseCard)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    observables: ObservableSpec = field(default_factory=ObservableSpec)
    seeds: SeedPlan = field(default_factory=SeedPlan)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    initialization: InitStrategy = InitStrategy.RANDOM
    replica: int = 0
    tags: tuple[str, ...] = ()
    notes: str = ""
    schema_version: str = "1.1"

    def __post_init__(self) -> None:
        if not self.experiment.strip():
            raise ValueError("experiment cannot be empty")
        if isinstance(self.phase, Mapping):
            object.__setattr__(self, "phase", _phase_from_dict(self.phase))
        if isinstance(self.training, Mapping):
            object.__setattr__(self, "training", TrainingSpec(**dict(self.training)))
        if isinstance(self.observables, Mapping):
            object.__setattr__(self, "observables", ObservableSpec(**dict(self.observables)))
        if isinstance(self.seeds, Mapping):
            object.__setattr__(self, "seeds", SeedPlan(**dict(self.seeds)))
        if isinstance(self.resources, Mapping):
            object.__setattr__(self, "resources", ResourceSpec(**dict(self.resources)))
        object.__setattr__(self, "initialization", _enum(self.initialization, InitStrategy))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        if self.replica < 0:
            raise ValueError("replica must be non-negative")
        if self.seeds.replica != self.replica:
            object.__setattr__(self, "seeds", SeedPlan(root=self.seeds.root, replica=self.replica, teacher=self.seeds.teacher, data=self.seeds.data, initialization=self.seeds.initialization, minibatch=self.seeds.minibatch, dropout=self.seeds.dropout))

    def to_dict(self) -> dict[str, Any]:
        return _encode_dataclass(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def run_id(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _encode_dataclass(value: Any) -> dict[str, Any]:
    return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}


def _phase_from_dict(value: Mapping[str, Any]) -> PhaseCard:
    payload = dict(value)
    payload["scaling"] = ScalingPath(**dict(payload.get("scaling", {})))
    return PhaseCard(**payload)


def run_spec_from_dict(value: Mapping[str, Any]) -> RunSpec:
    payload = dict(value)
    if "phase" not in payload:
        raise ValueError("run specification requires a phase table")
    payload["phase"] = _phase_from_dict(payload["phase"])
    payload["training"] = TrainingSpec(**dict(payload.get("training", {})))
    payload["observables"] = ObservableSpec(**dict(payload.get("observables", {})))
    payload["resources"] = ResourceSpec(**dict(payload.get("resources", {})))
    seed_payload = dict(payload.get("seeds", {}))
    replica = int(payload.get("replica", seed_payload.get("replica", 0)))
    seed_payload["replica"] = replica
    payload["seeds"] = SeedPlan(**seed_payload)
    payload["replica"] = replica
    return RunSpec(**payload)


__all__ = [
    "ArchitectureStage",
    "DataStage",
    "InitStrategy",
    "ObservableSpec",
    "OptimizerName",
    "PhaseCard",
    "Precision",
    "ResourceSpec",
    "RunSpec",
    "ScalingPath",
    "SeedPlan",
    "TrainingSpec",
    "run_spec_from_dict",
]
