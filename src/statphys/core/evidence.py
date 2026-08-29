"""Canonical evidence vector and conservative summary label."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceVector:
    semantics: int = 0
    replication: int = 0
    finite_size: int = 0
    prediction: int = 0
    intervention: int = 0
    external_validity: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value not in {0, 1, 2}:
                raise ValueError(f"evidence component {name} must be 0, 1, or 2")

    @property
    def grade(self) -> str:
        values = tuple(self.__dict__.values())
        if min(self.semantics, self.replication) == 2 and sum(value >= 2 for value in values) >= 4:
            return "A"
        if min(self.semantics, self.replication) >= 1 and sum(value >= 1 for value in values) >= 4:
            return "B"
        if self.replication >= 1 and sum(value >= 1 for value in values) >= 2:
            return "C"
        return "insufficient"

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "grade": self.grade}


class EvidenceEngine:
    """Translate detailed diagnostics into one stable evidence vector."""

    @staticmethod
    def assess(diagnostics: Mapping[str, Any]) -> EvidenceVector:
        semantics = (
            2
            if diagnostics.get("semantic_null_passed")
            else int(bool(diagnostics.get("observable_registered")))
        )
        replication = (
            2
            if int(diagnostics.get("outer_seeds", 0)) >= 12
            else int(int(diagnostics.get("outer_seeds", 0)) >= 5)
        )
        finite_size = (
            2
            if diagnostics.get("prospective_largest_size")
            else int(
                int(diagnostics.get("n_sizes", 0)) >= 3
                and bool(diagnostics.get("finite_size_diagnostic"))
            )
        )
        prediction = (
            2
            if diagnostics.get("untouched_holdout")
            else int(bool(diagnostics.get("frozen_comparison")))
        )
        intervention = (
            2 if diagnostics.get("matched_intervention") else int(bool(diagnostics.get("ablation")))
        )
        external = (
            2
            if diagnostics.get("pretrained_endpoint")
            else int(bool(diagnostics.get("natural_or_realistic_endpoint")))
        )
        return EvidenceVector(
            semantics=semantics,
            replication=replication,
            finite_size=finite_size,
            prediction=prediction,
            intervention=intervention,
            external_validity=external,
        )


__all__ = ["EvidenceEngine", "EvidenceVector"]
