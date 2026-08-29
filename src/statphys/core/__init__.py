"""Small, dependency-light contracts shared by StatPhys workflows.

The legacy namespaces remain available for compatibility. New systems and user
extensions should depend on this module rather than on a runner-specific API.
"""

from .contracts import Batch, LearningSystem, ObservableSpec, TaskSpec
from .rng import RNG_STREAMS, SeedStreams

__all__ = [
    "Batch",
    "LearningSystem",
    "ObservableSpec",
    "RNG_STREAMS",
    "SeedStreams",
    "TaskSpec",
]
