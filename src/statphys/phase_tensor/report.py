"""Strict manifest-seed aggregation for the empirical phase tensor."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import t as student_t

from statphys.continuation.core.schema import read_manifest


LEGACY_CAUSAL_METRICS = {
    "attention_contribution": "attention",
    "mlp_contribution": "mlp",
    "mlp_causal_contribution": "mlp",
    "attention_mlp_synergy": "synergy",
}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summary(values: Iterable[float], seed_ids: Iterable[int]) -> dict[str, Any]:
    data = np.asarray(tuple(values), dtype=np.float64)
    seeds = tuple(int(seed) for seed in seed_ids)
    if data.size != len(seeds) or data.size < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("summary values must have one unique seed ID per observation")
    if not np.isfinite(data).all():
        raise ValueError("summary values must all be finite")
    mean = float(data.mean())
    standard_deviation = float(data.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=data.size - 1))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(data.size),
        "ci95": critical * standard_deviation / math.sqrt(data.size),
        "n": int(data.size),
        "seed_ids": list(seeds),
        "raw_values": [float(value) for value in data],
    }


def _bounded_effect(delta: float, full: float) -> float:
    return float(delta / (abs(delta) + max(full, 0.0) + 1e-12))


def _canonical_causal_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add stable causal metrics, reconstructing historical runs when possible."""
    result = dict(metrics)
    full = _finite_float(result.get("full_risk"))
    if full is None:
        full = _finite_float(result.get("normalized_test_risk"))
    if full is None:
        full = _finite_float(result.get("normalized_generalization_error"))
    if full is None:
        return result
    result.setdefault("full_risk", full)
    legacy_scale = max(1.0 - full, 1e-8)
    branches = (
        ("attention", "attention_contribution"),
        ("mlp", "mlp_causal_contribution" if "mlp_causal_contribution" in result else "mlp_contribution"),
    )
    for branch, legacy_name in branches:
        risk_name = f"{branch}_ablated_risk"
        delta_name = f"{branch}_risk_delta"
        effect_name = f"{branch}_causal_effect"
        risk = _finite_float(result.get(risk_name))
        delta = _finite_float(result.get(delta_name))
        if delta is None and risk is not None:
            delta = risk - full
        if delta is None:
            legacy = _finite_float(result.get(legacy_name))
            if legacy is not None:
                delta = legacy * legacy_scale
        if delta is not None:
            result.setdefault(delta_name, delta)
            result.setdefault(risk_name, full + delta)
            result.setdefault(effect_name, _bounded_effect(delta, full))

    synergy_delta = _finite_float(result.get("attention_mlp_synergy_risk_delta"))
    if synergy_delta is None:
        legacy_synergy = _finite_float(result.get("attention_mlp_synergy"))
        if legacy_synergy is not None:
            synergy_delta = legacy_synergy * legacy_scale
    attention_risk = _finite_float(result.get("attention_ablated_risk"))
    mlp_risk = _finite_float(result.get("mlp_ablated_risk"))
    joint_risk = _finite_float(result.get("attention_mlp_ablated_risk"))
    if joint_risk is None and None not in (attention_risk, mlp_risk, synergy_delta):
        joint_risk = attention_risk + mlp_risk - full - synergy_delta
        result["attention_mlp_ablated_risk"] = joint_risk
    if joint_risk is not None:
        joint_delta = joint_risk - full
        result.setdefault("attention_mlp_risk_delta", joint_delta)
        result.setdefault("attention_mlp_causal_effect", _bounded_effect(joint_delta, full))
    if synergy_delta is None and None not in (attention_risk, mlp_risk, joint_risk):
        synergy_delta = attention_risk + mlp_risk - joint_risk - full
    if synergy_delta is not None:
        result.setdefault("attention_mlp_synergy_risk_delta", synergy_delta)
        result.setdefault("attention_mlp_synergy_effect", _bounded_effect(synergy_delta, full))
    return result


def _condition_metric_summaries(
    members: list[dict[str, Any]], seed_ids: tuple[int, ...]
) -> dict[str, Any]:
    metric_names = sorted(set().union(*(set(member["metrics"]) for member in members)))
    summaries: dict[str, Any] = {}
    members_by_seed = {int(member["seed"]): member for member in members}
    for metric_name in metric_names:
        values: list[float] = []
        for seed in seed_ids:
            member = members_by_seed[seed]
            if metric_name not in member["metrics"]:
                raise RuntimeError(f"required metric {metric_name!r} missing for seed {seed}")
            value = _finite_float(member["metrics"][metric_name])
            if value is None:
                raise RuntimeError(f"required metric {metric_name!r} is non-finite for seed {seed}")
            values.append(value)
        summaries[metric_name] = _summary(values, seed_ids)
    return summaries


def _threshold_estimate(
    controls: np.ndarray, values: np.ndarray, target: float = 0.5
) -> dict[str, float | str]:
    order = np.argsort(controls)
    x = controls[order]
    y = values[order] - target
    for left in range(len(x) - 1):
        if y[left] == 0.0:
            return {"value": float(x[left]), "status": "observed"}
        if y[left] * y[left + 1] < 0.0:
            fraction = -y[left] / (y[left + 1] - y[left] + 1e-12)
            return {
                "value": float(x[left] + fraction * (x[left + 1] - x[left])),
                "status": "interpolated",
            }
    if y[-1] == 0.0:
        return {"value": float(x[-1]), "status": "observed"}
    if np.all(y < 0.0):
        return {"value": float(x[-1]), "status": "right_censored"}
    if np.all(y > 0.0):
        return {"value": float(x[0]), "status": "left_censored"}
    return {"value": float(x[int(np.argmin(np.abs(y)))]), "status": "nonmonotone_no_crossing"}


def _boundary_summary(
    estimates: list[dict[str, Any]], seed_ids: tuple[int, ...]
) -> dict[str, Any] | None:
    genuine = {"interpolated", "observed"}
    if len(estimates) != len(seed_ids) or any(item["status"] not in genuine for item in estimates):
        return None
    by_seed = {int(item["seed_id"]): float(item["value"]) for item in estimates}
    return _summary([by_seed[seed] for seed in seed_ids], seed_ids)


def aggregate_phase_tensor(
    manifest_path: str | Path,
    runs_root: str | Path,
    output_path: str | Path,
) -> Path:
    manifest = read_manifest(manifest_path)
    manifest_seeds = tuple(int(seed) for seed in manifest.seeds)
    if len(manifest_seeds) < 5 or len(set(manifest_seeds)) != len(manifest_seeds):
        raise ValueError("phase-tensor aggregation requires at least five unique manifest seeds")
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
        metrics = _canonical_causal_metrics(
            json.loads(metrics_path.read_text(encoding="utf-8"))
        )
        trajectories: dict[str, np.ndarray] = {}
        arrays_path = directory / "arrays.npz"
        required_trajectories = (
            "history_step",
            "history_train_risk",
            "history_test_risk",
            "history_generalization_gap",
        )
        if not arrays_path.is_file():
            return None, f"{task.run_id}:arrays.npz"
        with np.load(arrays_path) as arrays:
            absent = [name for name in required_trajectories if name not in arrays]
            if absent:
                return None, f"{task.run_id}:missing-{','.join(absent)}"
            for name in required_trajectories:
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
        if seeds != set(manifest_seeds) or len(members) != len(manifest_seeds):
            raise RuntimeError(f"condition {condition_id} violates the exact manifest-seed contract")
        first = members[0]
        summaries = _condition_metric_summaries(members, manifest_seeds)
        conditions.append({
            key: first[key]
            for key in ("condition_id", "study", "domain", "family", "variant", "stage", "control_name", "control", "size", "parameters")
        } | {"metrics": summaries})

    dynamics: list[dict[str, Any]] = []
    for condition_id, members in sorted(grouped.items()):
        required = {"history_step", "history_train_risk", "history_test_risk", "history_generalization_gap"}
        if not all(required <= set(member["trajectories"]) for member in members):
            raise RuntimeError(f"condition {condition_id} lacks a required trajectory")
        reference_steps = members[0]["trajectories"]["history_step"]
        if not all(np.array_equal(reference_steps, member["trajectories"]["history_step"]) for member in members[1:]):
            raise RuntimeError(f"condition {condition_id} has incompatible trajectory checkpoints")
        summaries: dict[str, dict[str, list[float]]] = {}
        for name in sorted(required - {"history_step"}):
            members_by_seed = {int(member["seed"]): member for member in members}
            stacked = np.stack([members_by_seed[seed]["trajectories"][name] for seed in manifest_seeds], axis=0)
            if not np.isfinite(stacked).all():
                raise RuntimeError(f"trajectory {name!r} contains a non-finite value")
            critical = float(student_t.ppf(0.975, df=len(manifest_seeds) - 1))
            summaries[name] = {
                "mean": [float(value) for value in stacked.mean(axis=0)],
                "ci95": [float(critical * value / math.sqrt(len(manifest_seeds))) for value in stacked.std(axis=0, ddof=1)],
                "n": len(manifest_seeds),
                "seed_ids": list(manifest_seeds),
                "raw_values": [[float(value) for value in row] for row in stacked],
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
    thresholds: list[dict[str, Any]] = []
    for key, members in sorted(boundary_groups.items()):
        identity = json.loads(key)
        estimates: list[dict[str, Any]] = []
        for seed in manifest_seeds:
            seed_rows = [row for row in members if row["seed"] == seed]
            valid = []
            for row in seed_rows:
                value = _finite_float(row["metrics"].get("semantic_order"))
                if value is not None:
                    valid.append((row["control"], value))
            if len(valid) >= 2:
                estimate = _threshold_estimate(
                    np.asarray([item[0] for item in valid]),
                    np.asarray([item[1] for item in valid]),
                )
                estimates.append({"seed_id": seed} | estimate)
            else:
                estimates.append({"seed_id": seed, "value": None, "status": "insufficient_grid"})
        counts = {status: sum(item["status"] == status for item in estimates) for status in (
            "interpolated", "observed", "left_censored", "right_censored", "nonmonotone_no_crossing", "insufficient_grid"
        )}
        boundary = _boundary_summary(estimates, manifest_seeds)
        thresholds.append(identity | {
            "target": 0.5,
            "estimates": estimates,
            "status_counts": counts,
            "boundary_status": "identified" if boundary is not None else "censored_or_unidentified",
        })
        if boundary is not None:
            boundaries.append(identity | {"status": "identified", "semantic_order_half": boundary})

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
        "schema_version": "1.1",
        "study": manifest.study,
        "seeds": list(manifest_seeds),
        "seed_count": len(manifest_seeds),
        "uncertainty": f"two-sided 95% Student-t interval across {len(manifest_seeds)} full-pipeline seeds",
        "causal_metric_contract": {
            "effect": "delta_R / (abs(delta_R) + R_full + 1e-12)",
            "range": [-1.0, 1.0],
            "legacy_metrics_deprecated": sorted(LEGACY_CAUSAL_METRICS),
        },
        "tasks": len(rows),
        "conditions": conditions,
        "dynamics": dynamics,
        "thresholds": thresholds,
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
