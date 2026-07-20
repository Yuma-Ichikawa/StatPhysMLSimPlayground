"""Strict five-seed aggregation for the empirical phase-continuation tensor."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from statphys.continuation.core.schema import REQUIRED_SEED_COUNT, read_manifest


T95_FOUR_DOF = 2.7764451051977987


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(tuple(values), dtype=np.float64)
    if data.size != REQUIRED_SEED_COUNT:
        raise ValueError(f"expected exactly {REQUIRED_SEED_COUNT} observations, found {data.size}")
    mean = float(data.mean())
    standard_deviation = float(data.std(ddof=1))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(data.size),
        "ci95": T95_FOUR_DOF * standard_deviation / math.sqrt(data.size),
        "n": int(data.size),
    }


def _interpolated_crossing(controls: np.ndarray, values: np.ndarray, target: float = 0.5) -> float:
    order = np.argsort(controls)
    x = controls[order]
    y = values[order] - target
    for left in range(len(x) - 1):
        if y[left] == 0.0:
            return float(x[left])
        if y[left] * y[left + 1] <= 0.0:
            fraction = -y[left] / (y[left + 1] - y[left] + 1e-12)
            return float(x[left] + fraction * (x[left + 1] - x[left]))
    return float(x[int(np.argmin(np.abs(y)))])


def aggregate_phase_tensor(
    manifest_path: str | Path,
    runs_root: str | Path,
    output_path: str | Path,
) -> Path:
    manifest = read_manifest(manifest_path)
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    def read_task(task: Any) -> tuple[dict[str, Any] | None, str | None]:
        directory = root / "runs" / task.run_id
        status_path = directory / "status.json"
        metrics_path = directory / "metrics.json"
        if not status_path.is_file() or not metrics_path.is_file():
            return None, task.run_id
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            return None, task.run_id
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        trajectories: dict[str, np.ndarray] = {}
        arrays_path = directory / "arrays.npz"
        if arrays_path.is_file():
            with np.load(arrays_path) as arrays:
                for name in (
                    "history_step",
                    "history_train_risk",
                    "history_test_risk",
                    "history_generalization_gap",
                ):
                    if name in arrays:
                        trajectories[name] = arrays[name]
        return {
            "condition_id": task.condition_id,
            "run_id": task.run_id,
            "study": task.study,
            "domain": task.domain.value,
            "family": task.family,
            "variant": task.variant,
            "stage": task.stage,
            "control_name": task.control_name,
            "control": task.control,
            "size": task.size,
            "seed": task.seed,
            "parameters": dict(task.parameters),
            "metrics": metrics,
            "trajectories": trajectories,
        }, None

    with ThreadPoolExecutor(max_workers=32) as pool:
        for row, absent in pool.map(read_task, manifest.tasks):
            if row is not None:
                rows.append(row)
            if absent is not None:
                missing.append(absent)
    if missing:
        sample = ", ".join(missing[:8])
        raise RuntimeError(f"refusing incomplete aggregation: {len(missing)} missing/failed runs ({sample})")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_id"]].append(row)
    conditions: list[dict[str, Any]] = []
    for condition_id, members in sorted(grouped.items()):
        seeds = {int(member["seed"]) for member in members}
        if seeds != set(manifest.seeds) or len(members) != REQUIRED_SEED_COUNT:
            raise RuntimeError(f"condition {condition_id} violates the exact five-seed contract")
        first = members[0]
        metric_names = sorted(set.intersection(*(set(member["metrics"]) for member in members)))
        summaries: dict[str, Any] = {}
        for metric_name in metric_names:
            values = [_finite_float(member["metrics"][metric_name]) for member in members]
            if all(value is not None for value in values):
                summaries[metric_name] = _summary(value for value in values if value is not None)
        conditions.append({
            key: first[key]
            for key in ("condition_id", "study", "domain", "family", "variant", "stage", "control_name", "control", "size", "parameters")
        } | {"metrics": summaries})

    dynamics: list[dict[str, Any]] = []
    for condition_id, members in sorted(grouped.items()):
        required = {"history_step", "history_train_risk", "history_test_risk", "history_generalization_gap"}
        if not all(required <= set(member["trajectories"]) for member in members):
            continue
        reference_steps = members[0]["trajectories"]["history_step"]
        if not all(np.array_equal(reference_steps, member["trajectories"]["history_step"]) for member in members[1:]):
            raise RuntimeError(f"condition {condition_id} has incompatible trajectory checkpoints")
        summaries: dict[str, dict[str, list[float]]] = {}
        for name in sorted(required - {"history_step"}):
            stacked = np.stack([member["trajectories"][name] for member in members], axis=0)
            summaries[name] = {
                "mean": [float(value) for value in stacked.mean(axis=0)],
                "ci95": [float(T95_FOUR_DOF * value / math.sqrt(REQUIRED_SEED_COUNT)) for value in stacked.std(axis=0, ddof=1)],
            }
        first = members[0]
        dynamics.append({
            key: first[key]
            for key in ("condition_id", "family", "variant", "control", "size", "parameters")
        } | {"steps": [int(value) for value in reference_steps], "metrics": summaries})

    boundary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = {
            "study": row["study"],
            "domain": row["domain"],
            "family": row["family"],
            "variant": row["variant"],
            "stage": row["stage"],
            "control_name": row["control_name"],
            "size": row["size"],
            "parameters": row["parameters"],
        }
        boundary_groups[json.dumps(identity, sort_keys=True, separators=(",", ":"))].append(row)
    boundaries: list[dict[str, Any]] = []
    for key, members in sorted(boundary_groups.items()):
        identity = json.loads(key)
        crossings: list[float] = []
        for seed in manifest.seeds:
            seed_rows = [row for row in members if row["seed"] == seed]
            valid = []
            for row in seed_rows:
                value = _finite_float(row["metrics"].get("semantic_order"))
                if value is not None:
                    valid.append((row["control"], value))
            if len(valid) >= 2:
                crossings.append(_interpolated_crossing(
                    np.asarray([item[0] for item in valid]),
                    np.asarray([item[1] for item in valid]),
                ))
        if len(crossings) == REQUIRED_SEED_COUNT:
            boundaries.append(identity | {"semantic_order_half": _summary(crossings)})

    intensive_metrics = (
        "normalized_generalization_error",
        "normalized_train_risk",
        "normalized_test_risk",
        "normalized_ood_risk",
        "normalized_generalization_gap",
        "normalized_ce",
        "normalized_brier",
        "attention_entropy",
        "mlp_participation_fraction",
        "mlp_activation_entropy",
    )
    intensive_groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    intensive_identities: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        for metric_name in intensive_metrics:
            estimate = condition["metrics"].get(metric_name)
            if estimate is None:
                continue
            identity = {
                "family": condition["family"],
                "variant": condition["variant"],
                "control": condition["control"],
                "parameters": condition["parameters"],
                "metric": metric_name,
            }
            key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            intensive_identities[key] = identity
            intensive_groups[key].append((int(condition["size"]), float(estimate["mean"])))
    intensive_checks = []
    for key, values in sorted(intensive_groups.items()):
        unique = sorted(set(values))
        if len(unique) < 3:
            continue
        x = np.log(np.asarray([item[0] for item in unique], dtype=np.float64))
        y = np.asarray([item[1] for item in unique], dtype=np.float64)
        slope = float(np.polyfit(x, y, deg=1)[0])
        intensive_checks.append(
            intensive_identities[key]
            | {
                "sizes": [item[0] for item in unique],
                "log_size_slope": slope,
                "relative_log_size_slope": slope / max(abs(float(y.mean())), 1e-12),
            }
        )

    payload = {
        "schema_version": "1.0",
        "study": manifest.study,
        "seeds": list(manifest.seeds),
        "uncertainty": "two-sided 95% Student-t interval across five full-pipeline seeds",
        "tasks": len(rows),
        "conditions": conditions,
        "dynamics": dynamics,
        "boundaries": boundaries,
        "intensive_scaling_checks": intensive_checks,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = ["aggregate_phase_tensor"]
