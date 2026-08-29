"""Structural audit for the phase-continuation taxonomy and realism tiers."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from ..core.registry import is_supported
from ..core.schema import validate_seed_set

COORDINATES = ("data", "architecture", "objective", "dynamics", "scale", "lifecycle")
OUTCOMES = (
    "preserved",
    "renormalized",
    "rounded",
    "split",
    "merged",
    "new_regime",
    "hysteretic",
    "path_dependent",
    "statistical_computational_separation",
    "semantic_failure",
    "censored_no_crossing",
    "unresolved",
    "not_comparable",
)
TIERS = ("A", "B", "B+", "C")
THEORY_STATUSES = {f"T{index}" for index in range(7)}
REALISM_LEVELS = {f"R{index}" for index in range(6)}
EVIDENCE_STATUSES = {f"E{index}" for index in range(7)}


def validate_taxonomy(path: str | Path, config_paths: Sequence[str | Path]) -> dict[str, Any]:
    registry = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if tuple(registry.get("coordinates", ())) != COORDINATES:
        errors.append("coordinates must contain the canonical six solvability axes in order")
    if tuple(registry.get("outcomes", ())) != OUTCOMES:
        errors.append("outcomes must contain all 13 continuation outcomes in order")
    if set(registry.get("tier_definitions", {})) != set(TIERS):
        errors.append("tier_definitions must define A, B, B+, and C")
    if set(registry.get("theory_status", {})) != THEORY_STATUSES:
        errors.append("theory_status must define T0 through T6")
    if set(registry.get("realism_level", {})) != REALISM_LEVELS:
        errors.append("realism_level must define R0 through R5")
    if set(registry.get("evidence_status", {})) != EVIDENCE_STATUSES:
        errors.append("evidence_status must define E0 through E6")

    configured: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pair_values: set[str] = set()
    reports = []
    for config_path in config_paths:
        raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        seeds = tuple(raw.get("study", {}).get("seeds", ()))
        validate_seed_set(seeds)
        for experiment in raw.get("experiments", []):
            key = (str(experiment["domain"]), str(experiment.get("family", "anchor")))
            configured.setdefault(key, []).append(dict(experiment))
            if key == ("cross_domain", "assumption_pairs"):
                pair_values.update(
                    str(value) for value in experiment.get("parameter_grid", {}).get("pair", ())
                )
        reports.append({"path": str(config_path), "seeds": len(seeds)})

    endpoint_ids: set[str] = set()
    runnable = 0
    protocols = 0
    for endpoint in registry.get("endpoints", []):
        identifier = str(endpoint.get("id", ""))
        if not identifier or identifier in endpoint_ids:
            errors.append(f"duplicate or empty endpoint id: {identifier!r}")
        endpoint_ids.add(identifier)
        tier = str(endpoint.get("tier", ""))
        status = str(endpoint.get("status", ""))
        if tier not in TIERS:
            errors.append(f"{identifier}: invalid tier {tier!r}")
        if endpoint.get("theory_status") not in THEORY_STATUSES:
            errors.append(f"{identifier}: invalid theory status")
        if endpoint.get("realism_level") not in REALISM_LEVELS:
            errors.append(f"{identifier}: invalid realism level")
        if endpoint.get("evidence_status") not in EVIDENCE_STATUSES:
            errors.append(f"{identifier}: invalid evidence status")
        if not endpoint.get("state_object") or not endpoint.get("phenomenon_type"):
            errors.append(f"{identifier}: state_object and phenomenon_type are required")
        if status == "runnable":
            runnable += 1
            domain = str(endpoint.get("domain", ""))
            family = str(endpoint.get("family", ""))
            aliases = {domain, "reinforcement" if domain == "rl" else domain}
            matches = [item for alias in aliases for item in configured.get((alias, family), [])]
            if not is_supported(domain, family):
                errors.append(f"{identifier}: unsupported runner {domain}/{family}")
            if not matches:
                errors.append(f"{identifier}: no matching configuration")
            if not endpoint.get("observables") or not endpoint.get("falsifier"):
                errors.append(f"{identifier}: observables and falsifier are required")
            if not endpoint.get("finite_size_coordinate") or not endpoint.get("scaling_path"):
                errors.append(f"{identifier}: finite_size_coordinate and scaling_path are required")
        elif status == "protocol":
            protocols += 1
            if tier != "C" or not endpoint.get("limitation"):
                errors.append(f"{identifier}: Tier C protocol requires an explicit limitation")
        else:
            errors.append(f"{identifier}: status must be runnable or protocol")

    expected_pairs = {"__".join(pair) for pair in combinations(COORDINATES, 2)}
    missing_pairs = sorted(expected_pairs - pair_values)
    if missing_pairs:
        errors.append("assumption-pair grid is incomplete")
    return {
        "ok": not errors,
        "runnable_complete": not errors,
        "full_realism_complete": protocols == 0,
        "runnable_endpoints": runnable,
        "tier_c_protocols": protocols,
        "coordinates": len(COORDINATES),
        "outcomes": len(OUTCOMES),
        "assumption_pairs": len(pair_values & expected_pairs),
        "missing_assumption_pairs": missing_pairs,
        "errors": errors,
        "configs": reports,
    }


__all__ = ["COORDINATES", "OUTCOMES", "TIERS", "validate_taxonomy"]
