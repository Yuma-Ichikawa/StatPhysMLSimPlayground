"""Artifact invalidation records that preserve unaffected measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ArtifactValidity(str, Enum):
    VALID = "valid"
    PARTIALLY_INVALIDATED = "partially_invalidated"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class InvalidationRecord:
    run_id: str
    validity: ArtifactValidity
    invalid_metrics: tuple[str, ...]
    retained_metrics: tuple[str, ...]
    reason: str
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "validity": self.validity.value,
            "invalid_metrics": list(self.invalid_metrics),
            "retained_metrics": list(self.retained_metrics),
            "reason": self.reason,
            "superseded_by": self.superseded_by,
        }


__all__ = ["ArtifactValidity", "InvalidationRecord"]
