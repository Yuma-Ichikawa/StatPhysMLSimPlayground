"""Contracts for runner outputs and their scientific interpretation."""

from __future__ import annotations

from dataclasses import dataclass

from .scientific_spec import Fidelity, TheoryStatus


@dataclass(frozen=True)
class RunnerContract:
    required_metrics: frozenset[str]
    optional_metrics: frozenset[str] = frozenset()
    required_arrays: frozenset[str] = frozenset()
    theory_status: TheoryStatus = TheoryStatus.EMPIRICAL_ONLY
    fidelity: Fidelity = Fidelity.TRAINABLE_SYNTHETIC
    phase_ensemble: str = "outer_seed"

    def validate_metric_names(self, names: set[str], *, context: str = "runner") -> None:
        missing = self.required_metrics - names
        if missing:
            raise ValueError(f"{context} is missing required metrics: {', '.join(sorted(missing))}")


__all__ = ["RunnerContract"]
