"""One uncertainty schema for all public aggregates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Estimate:
    mean: float
    interval_low: float | None
    interval_high: float | None
    interval_level: float | None
    uncertainty_method: str
    n_outer: int
    n_inner: int | None
    outer_seed_ids: tuple[int, ...]
    raw_outer_values: tuple[float, ...]
    units: str
    status: str = "valid"
    censoring: str | None = None
    standard_deviation: float | None = None
    standard_error: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("estimate mean must be finite")
        if self.n_outer != len(self.outer_seed_ids) or self.n_outer != len(self.raw_outer_values):
            raise ValueError("n_outer, seed ids, and raw values must agree")
        if len(set(self.outer_seed_ids)) != len(self.outer_seed_ids):
            raise ValueError("outer seed ids must be unique")
        if any(not math.isfinite(value) for value in self.raw_outer_values):
            raise ValueError("raw outer values must be finite")
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("both interval endpoints must be present or absent")
        if (
            self.interval_low is not None
            and self.interval_high is not None
            and (self.interval_low > self.mean or self.mean > self.interval_high)
        ):
            raise ValueError("interval must contain the estimate mean")

    def to_dict(self, *, compatibility: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mean": self.mean,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "interval_level": self.interval_level,
            "uncertainty_method": self.uncertainty_method,
            "n_outer": self.n_outer,
            "n_inner": self.n_inner,
            "outer_seed_ids": list(self.outer_seed_ids),
            "raw_outer_values": list(self.raw_outer_values),
            "units": self.units,
            "status": self.status,
            "censoring": self.censoring,
            "standard_deviation": self.standard_deviation,
            "standard_error": self.standard_error,
        }
        if compatibility:
            half_width = (
                None
                if self.interval_low is None or self.interval_high is None
                else (self.interval_high - self.interval_low) / 2.0
            )
            result.update(
                {
                    "n": self.n_outer,
                    "sd": self.standard_deviation,
                    "sem": self.standard_error,
                    "ci95": half_width if self.interval_level == 0.95 else None,
                    "ci95_low": self.interval_low if self.interval_level == 0.95 else None,
                    "ci95_high": self.interval_high if self.interval_level == 0.95 else None,
                }
            )
        return result


__all__ = ["Estimate"]
