"""Dependency-light public contracts shared by StatPhys workflows.

Scientific metadata remains importable on login and orchestration nodes that do
not provide PyTorch. Numerical contracts are loaded only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .scientific_spec import (
    Fidelity,
    ObservableValue,
    Outcome,
    PhaseCardV3,
    PhenomenonType,
    ScaleSpec,
    ScientificSpec,
    TheorySpec,
    TheoryStatus,
    structural_features,
)

_LAZY_IMPORTS = {
    "Batch": (".contracts", "Batch"),
    "LearningSystem": (".contracts", "LearningSystem"),
    "ObservableSpec": (".contracts", "ObservableSpec"),
    "TaskSpec": (".contracts", "TaskSpec"),
    "Estimate": (".estimates", "Estimate"),
    "EvidenceEngine": (".evidence", "EvidenceEngine"),
    "EvidenceVector": (".evidence", "EvidenceVector"),
    "ExecutionSpec": (".execution_spec", "ExecutionSpec"),
    "IdentityBundle": (".execution_spec", "IdentityBundle"),
    "content_id": (".execution_spec", "content_id"),
    "DisorderRole": (".protocol_spec", "DisorderRole"),
    "DisorderSpec": (".protocol_spec", "DisorderSpec"),
    "ProtocolSpec": (".protocol_spec", "ProtocolSpec"),
    "ArtifactValidity": (".provenance", "ArtifactValidity"),
    "InvalidationRecord": (".provenance", "InvalidationRecord"),
    "RunnerContract": (".registry", "RunnerContract"),
    "RNG_STREAMS": (".rng", "RNG_STREAMS"),
    "SeedStreams": (".rng", "SeedStreams"),
}

__all__ = [
    "Batch",
    "DisorderRole",
    "DisorderSpec",
    "Estimate",
    "EvidenceEngine",
    "EvidenceVector",
    "ExecutionSpec",
    "Fidelity",
    "IdentityBundle",
    "InvalidationRecord",
    "LearningSystem",
    "ObservableSpec",
    "ObservableValue",
    "Outcome",
    "PhaseCardV3",
    "PhenomenonType",
    "ProtocolSpec",
    "RNG_STREAMS",
    "RunnerContract",
    "ScaleSpec",
    "ScientificSpec",
    "SeedStreams",
    "TaskSpec",
    "TheorySpec",
    "TheoryStatus",
    "ArtifactValidity",
    "content_id",
    "structural_features",
]


def __getattr__(name: str) -> Any:
    """Load numerical public contracts only when callers request them."""

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
