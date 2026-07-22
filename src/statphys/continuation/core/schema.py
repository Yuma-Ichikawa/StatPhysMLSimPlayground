"""Immutable, portable manifests with a registered, at-least-five-seed contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

REQUIRED_SEED_COUNT = 5
SCHEMA_VERSION = "1.1"


class Domain(str, Enum):
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    RL = "reinforcement"
    MULTIAGENT = "multiagent"
    CROSS = "cross_domain"

    @classmethod
    def parse(cls, value: str | "Domain") -> "Domain":
        if isinstance(value, cls):
            return value
        aliases = {
            "rl": cls.RL,
            "reinforcement_learning": cls.RL,
            "reinforcement-learning": cls.RL,
            "multi_agent": cls.MULTIAGENT,
            "multi-agent": cls.MULTIAGENT,
            "cross": cls.CROSS,
            "cross-domain": cls.CROSS,
        }
        key = str(value).strip().lower()
        if key in aliases:
            return aliases[key]
        return cls(key)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _digest(payload: Mapping[str, Any], length: int = 20) -> str:
    encoded = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


def validate_seed_set(seeds: Sequence[int]) -> None:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < REQUIRED_SEED_COUNT or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"every study must use at least {REQUIRED_SEED_COUNT} distinct seeds"
        )
    if any(seed < 0 for seed in normalized):
        raise ValueError("seeds must be non-negative")


def derive_seed(root: int, namespace: str, index: int = 0) -> int:
    """Derive stable nested seeds without depending on Python's randomized hash."""
    payload = f"{int(root)}:{namespace}:{int(index)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class TaskSpec:
    study: str
    domain: Domain
    variant: str
    stage: str
    control_name: str
    control: float
    size: int
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    family: str = "anchor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", Domain.parse(self.domain))
        object.__setattr__(self, "study", str(self.study))
        object.__setattr__(self, "family", _slug(str(self.family)).lower())
        object.__setattr__(self, "variant", _slug(str(self.variant)).lower())
        object.__setattr__(self, "stage", str(self.stage).lower())
        object.__setattr__(self, "control_name", _slug(str(self.control_name)).lower())
        object.__setattr__(self, "control", float(self.control))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "parameters", MappingProxyType(dict(_plain(self.parameters))))
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.study:
            raise ValueError("study must be non-empty")

    @property
    def condition_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study": self.study,
            "domain": self.domain.value,
            "family": self.family,
            "variant": self.variant,
            "stage": self.stage,
            "control_name": self.control_name,
            "control": self.control,
            "size": self.size,
            "parameters": _plain(self.parameters),
        }

    @property
    def condition_id(self) -> str:
        return _digest(self.condition_payload)

    @property
    def task_id(self) -> str:
        return f"{self.domain.value}-{self.family}-{self.condition_id}-s{self.seed}"

    @property
    def run_id(self) -> str:
        return self.task_id

    @property
    def nested_seeds(self) -> dict[str, int]:
        return {
            name: derive_seed(self.seed, name)
            for name in ("disorder", "data", "initialization", "minibatch", "dropout", "evaluation")
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study": self.study,
            "domain": self.domain.value,
            "family": self.family,
            "variant": self.variant,
            "stage": self.stage,
            "control_name": self.control_name,
            "control": self.control,
            "size": self.size,
            "seed": self.seed,
            "nested_seeds": self.nested_seeds,
            "parameters": _plain(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        data = dict(payload)
        data.pop("nested_seeds", None)
        data["domain"] = Domain.parse(data["domain"])
        data.setdefault("family", "anchor")
        return cls(**data)


@dataclass(frozen=True)
class Manifest:
    study: str
    seeds: tuple[int, ...]
    tasks: tuple[TaskSpec, ...]
    config_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        tasks = tuple(self.tasks)
        validate_seed_set(seeds)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "metadata", MappingProxyType(dict(_plain(self.metadata))))
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest contains duplicate task ids")
        by_condition: dict[str, set[int]] = {}
        for task in tasks:
            by_condition.setdefault(task.condition_id, set()).add(task.seed)
        expected = set(seeds)
        incomplete = [condition for condition, found in by_condition.items() if found != expected]
        if incomplete:
            raise ValueError("every condition must contain the exact registered seed set")

    @property
    def n_conditions(self) -> int:
        return len({task.condition_id for task in self.tasks})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study": self.study,
            "seeds": list(self.seeds),
            "config_hash": self.config_hash,
            "metadata": _plain(self.metadata),
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Manifest":
        return cls(
            study=str(payload["study"]),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            tasks=tuple(TaskSpec.from_dict(item) for item in payload["tasks"]),
            config_hash=str(payload["config_hash"]),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


def _parameter_combinations(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(experiment.get("parameters", {}))
    grid = dict(experiment.get("parameter_grid", {}))
    if not grid:
        return [base]
    keys = sorted(grid)
    values = [value if isinstance(value, list) else [value] for value in (grid[key] for key in keys)]
    return [{**base, **dict(zip(keys, items, strict=True))} for items in product(*values)]


def expand_config(path: str | Path) -> Manifest:
    source = Path(path)
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    study_data = dict(raw.get("study", {}))
    name = str(study_data.pop("name"))
    seeds = tuple(int(seed) for seed in study_data.pop("seeds"))
    validate_seed_set(seeds)
    tasks: list[TaskSpec] = []
    for experiment in raw.get("experiments", []):
        domain = Domain.parse(experiment["domain"])
        family = str(experiment.get("family", "anchor"))
        variants = experiment.get("variants", [experiment.get("variant", "default")])
        controls = experiment.get("controls", [experiment.get("control", 0.0)])
        sizes = experiment.get("sizes", [experiment.get("size", 1)])
        stage = str(experiment.get("stage", "confirmatory"))
        control_name = str(experiment.get("control_name", "control"))
        for variant, control, size, seed, parameters in product(
            variants, controls, sizes, seeds, _parameter_combinations(experiment)
        ):
            tasks.append(
                TaskSpec(
                    study=name,
                    domain=domain,
                    family=family,
                    variant=str(variant),
                    stage=stage,
                    control_name=control_name,
                    control=float(control),
                    size=int(size),
                    seed=int(seed),
                    parameters=parameters,
                )
            )
    if not tasks:
        raise ValueError("configuration contains no experiments")
    tasks.sort(key=lambda task: task.task_id)
    return Manifest(
        study=name,
        seeds=seeds,
        tasks=tuple(tasks),
        config_hash=sha256(raw_bytes).hexdigest(),
        metadata={**study_data, "source_config": source.name},
    )


def compose_manifests(manifests: Sequence[Manifest | str | Path], study: str) -> Manifest:
    loaded = [read_manifest(item) if not isinstance(item, Manifest) else item for item in manifests]
    if not loaded:
        raise ValueError("at least one manifest is required")
    seeds = loaded[0].seeds
    if any(manifest.seeds != seeds for manifest in loaded[1:]):
        raise ValueError("composed manifests must share the same ordered seed set")
    # Preserve component task identities so composed manifests can aggregate artifacts
    # produced from the immutable component manifests.
    tasks = tuple(task for manifest in loaded for task in manifest.tasks)
    digest = sha256("".join(manifest.config_hash for manifest in loaded).encode("ascii")).hexdigest()
    return Manifest(
        study=study,
        seeds=seeds,
        tasks=tasks,
        config_hash=digest,
        metadata={"components": [manifest.study for manifest in loaded]},
    )


def write_manifest(path: str | Path, manifest: Manifest) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_manifest(path: str | Path) -> Manifest:
    return Manifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "Domain",
    "Manifest",
    "REQUIRED_SEED_COUNT",
    "SCHEMA_VERSION",
    "TaskSpec",
    "compose_manifests",
    "derive_seed",
    "expand_config",
    "read_manifest",
    "validate_seed_set",
    "write_manifest",
]
