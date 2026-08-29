"""Small, dependency-light contracts shared by StatPhys workflows.

The legacy namespaces remain available for compatibility. New systems and user
extensions should depend on this module rather than on a runner-specific API.
"""

from .contracts import Batch, LearningSystem, ObservableSpec, TaskSpec
from .estimates import Estimate
from .evidence import EvidenceEngine, EvidenceVector
from .execution_spec import ExecutionSpec, IdentityBundle, content_id
from .protocol_spec import DisorderRole, DisorderSpec, ProtocolSpec
from .provenance import ArtifactValidity, InvalidationRecord
from .registry import RunnerContract
from .rng import RNG_STREAMS, SeedStreams
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
