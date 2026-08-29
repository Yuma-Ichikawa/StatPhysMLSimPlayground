"""Strict registered-seed aggregation with Student-t uncertainty and evidence grades."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from statphys.core import Estimate, EvidenceEngine

from ..core.registry import runner_contract
from ..schema import REQUIRED_SEED_COUNT, Manifest, TaskSpec, read_manifest

T95_DF4 = 2.7764451051977987


def _read_completed(
    task: TaskSpec, root: Path
) -> tuple[dict[str, float], dict[str, np.ndarray]] | None:
    directory = root / "runs" / task.run_id
    status_path = directory / "status.json"
    metrics_path = directory / "metrics.json"
    spec_path = directory / "spec.json"
    if not status_path.exists() or not metrics_path.exists() or not spec_path.exists():
        return None
    status = json.loads(status_path.read_text())
    if status.get("state") != "completed":
        return None
    registered = TaskSpec.from_dict(json.loads(spec_path.read_text()))
    if registered.run_id != task.run_id:
        raise ValueError(f"artifact spec mismatch for {task.run_id}")
    metrics = json.loads(metrics_path.read_text())
    result = {name: float(value) for name, value in metrics.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite metric in {task.run_id}")
    contract = runner_contract(task)
    contract.validate_metric_names(set(result), context=task.run_id)
    arrays: dict[str, np.ndarray] = {}
    if contract.required_arrays:
        array_path = directory / "arrays.npz"
        if not array_path.exists():
            raise ValueError(f"required arrays are absent in {task.run_id}")
        with np.load(array_path, allow_pickle=False) as stored:
            missing = contract.required_arrays - set(stored.files)
            if missing:
                raise ValueError(
                    f"{task.run_id} is missing required arrays: {', '.join(sorted(missing))}"
                )
            arrays = {name: np.asarray(stored[name]) for name in contract.required_arrays}
    return result, arrays


def _metric_units(name: str) -> str:
    if name.endswith("_seconds") or name == "wall_seconds":
        return "seconds"
    if "flops" in name:
        return "FLOP"
    if name.endswith("_count") or name in {"parameter_count", "tokens_seen"}:
        return "count"
    return "dimensionless"


def _interval(
    values: Iterable[float], seeds: Iterable[int] | None = None, *, metric_name: str = "metric"
) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    seed_ids = (
        tuple(range(int(array.size))) if seeds is None else tuple(int(seed) for seed in seeds)
    )
    if array.size < REQUIRED_SEED_COUNT or not np.isfinite(array).all():
        raise ValueError(f"uncertainty requires at least {REQUIRED_SEED_COUNT} finite seeds")
    if array.size != len(seed_ids):
        raise ValueError("each outer value must have one registered seed id")
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(array.size)
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    mean = float(array.mean())
    half_width = critical * standard_error
    return Estimate(
        mean=mean,
        interval_low=mean - half_width,
        interval_high=mean + half_width,
        interval_level=0.95,
        uncertainty_method="Student-t interval over independent outer seeds",
        n_outer=int(array.size),
        n_inner=None,
        outer_seed_ids=seed_ids,
        raw_outer_values=tuple(float(value) for value in array),
        units=_metric_units(metric_name),
        standard_deviation=standard_deviation,
        standard_error=standard_error,
    ).to_dict()


def _outer_phase_estimates(
    runs: list[tuple[TaskSpec, dict[str, float], dict[str, np.ndarray]]],
) -> dict[str, dict[str, Any]]:
    if not all("signed_order_parameter" in metrics for _, metrics, _ in runs):
        return {}
    ordered = sorted(runs, key=lambda item: item[0].seed)
    seeds = [task.seed for task, _, _ in ordered]
    samples = np.asarray(
        [metrics["signed_order_parameter"] for _, metrics, _ in ordered], dtype=np.float64
    )
    n_eff = float(ordered[0][0].finite_size_value)

    def susceptibility(values: np.ndarray) -> float:
        return n_eff * float(np.mean(values**2) - np.mean(values) ** 2)

    def binder(values: np.ndarray) -> float:
        second = float(np.mean(values**2))
        fourth = float(np.mean(values**4))
        return 0.0 if second <= np.finfo(float).eps else 1.0 - fourth / (3.0 * second**2)

    def sign_entropy(values: np.ndarray) -> float:
        positive = float(np.mean(values > 0.0))
        if positive <= 0.0 or positive >= 1.0:
            return 0.0
        return float(
            -(positive * math.log(positive) + (1.0 - positive) * math.log(1.0 - positive))
            / math.log(2.0)
        )

    def jackknife(statistic: Any, metric_name: str) -> dict[str, Any]:
        point = float(statistic(samples))
        n_outer = len(samples)
        leave_one_out = np.asarray(
            [statistic(np.delete(samples, index)) for index in range(n_outer)], dtype=float
        )
        pseudo = n_outer * point - (n_outer - 1) * leave_one_out
        standard_deviation = float(pseudo.std(ddof=1))
        standard_error = standard_deviation / math.sqrt(n_outer)
        critical = float(student_t.ppf(0.975, df=n_outer - 1))
        return Estimate(
            mean=point,
            interval_low=point - critical * standard_error,
            interval_high=point + critical * standard_error,
            interval_level=0.95,
            uncertainty_method="delete-one outer-seed jackknife",
            n_outer=n_outer,
            n_inner=None,
            outer_seed_ids=tuple(seeds),
            raw_outer_values=tuple(float(value) for value in pseudo),
            units=_metric_units(metric_name),
            standard_deviation=standard_deviation,
            standard_error=standard_error,
        ).to_dict()

    return {
        "susceptibility": jackknife(susceptibility, "susceptibility"),
        "binder_cumulant": jackknife(binder, "binder_cumulant"),
        "disorder_sign_entropy": jackknife(sign_entropy, "disorder_sign_entropy"),
    }


def _evidence(records: list[dict[str, Any]], registered_seed_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        parameter_key = json.dumps(record["parameters"], sort_keys=True, separators=(",", ":"))
        groups[(record["domain"], record["family"], record["variant"], parameter_key)].append(
            record
        )
    result: list[dict[str, Any]] = []
    for (domain, family, variant, parameter_key), values in sorted(groups.items()):
        controls = sorted({float(value["control"]) for value in values})
        sizes = sorted({int(value["size"]) for value in values})
        largest = max(sizes)
        large_records = sorted(
            (value for value in values if int(value["size"]) == largest),
            key=lambda value: float(value["control"]),
        )
        susceptibility = [
            float(value["metrics"]["susceptibility"]["mean"]) for value in large_records
        ]
        peak_index = int(np.argmax(susceptibility))
        interior_peak = 0 < peak_index < len(large_records) - 1
        complete = all(
            int(metric["n"]) == registered_seed_count
            for value in values
            for metric in value["metrics"].values()
        )
        confirmatory = all(value["stage"] == "confirmatory" for value in values)
        peak_by_size = []
        for size in sizes:
            candidates = [item for item in values if int(item["size"]) == size]
            peak_by_size.append(
                max(float(item["metrics"]["susceptibility"]["mean"]) for item in candidates)
            )
        susceptibility_growth = (
            peak_by_size[-1] > peak_by_size[0] if len(peak_by_size) > 1 else False
        )
        parameters = json.loads(parameter_key)
        evidence_vector = EvidenceEngine.assess(
            {
                "observable_registered": True,
                "semantic_null_passed": bool(parameters.get("semantic_null_passed", False)),
                "outer_seeds": registered_seed_count if complete else 0,
                "n_sizes": len(sizes),
                "finite_size_diagnostic": interior_peak and susceptibility_growth,
                "prospective_largest_size": bool(parameters.get("largest_size_holdout", False)),
                "frozen_comparison": confirmatory,
                "untouched_holdout": bool(parameters.get("untouched_holdout", False)),
                "matched_intervention": bool(parameters.get("matched_intervention", False)),
                "natural_or_realistic_endpoint": bool(parameters.get("natural_endpoint", False)),
                "pretrained_endpoint": bool(parameters.get("pretrained_endpoint", False)),
            }
        )
        result.append(
            {
                "domain": domain,
                "family": family,
                "variant": variant,
                "parameters": parameters,
                "grade": evidence_vector.grade,
                "evidence_vector": evidence_vector.to_dict(),
                "complete_registered_seed": complete,
                "complete_five_seed": complete and registered_seed_count == REQUIRED_SEED_COUNT,
                "confirmatory": confirmatory,
                "n_sizes": len(sizes),
                "n_controls": len(controls),
                "interior_susceptibility_peak": interior_peak,
                "susceptibility_peak_growth": susceptibility_growth,
                "peak_control_largest_size": float(large_records[peak_index]["control"]),
                "largest_size": largest,
            }
        )
    return result


def aggregate_manifest(
    manifest: Manifest | str | Path,
    run_root: str | Path,
    output_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    registered = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    root = Path(run_root)
    missing: list[str] = []
    grouped: dict[str, list[tuple[TaskSpec, dict[str, float], dict[str, np.ndarray]]]] = (
        defaultdict(list)
    )
    for task in registered.tasks:
        completed = _read_completed(task, root)
        if completed is None:
            missing.append(task.run_id)
            continue
        metrics, arrays = completed
        grouped[task.condition_id].append((task, metrics, arrays))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} of {len(registered.tasks)} registered runs are incomplete"
        )

    records: list[dict[str, Any]] = []
    expected_seeds = set(registered.seeds)
    for condition_id, runs in sorted(grouped.items()):
        run_seed_ids = [task.seed for task, _, _ in runs]
        seeds = set(run_seed_ids)
        if len(run_seed_ids) != len(seeds):
            raise RuntimeError(f"condition {condition_id} contains duplicate outer seeds")
        if seeds != expected_seeds:
            if allow_incomplete:
                continue
            raise RuntimeError(f"condition {condition_id} does not contain the registered seeds")
        runs.sort(key=lambda item: item[0].seed)
        first = runs[0][0]
        contract = runner_contract(first)
        schemas = {tuple(sorted(metrics)) for _, metrics, _ in runs}
        if len(schemas) != 1:
            by_seed = {task.seed: sorted(metrics) for task, metrics, _ in runs}
            raise RuntimeError(
                f"condition {condition_id} has inconsistent metric schemas across seeds: {by_seed}"
            )
        metric_names = set(next(iter(schemas)))
        contract.validate_metric_names(metric_names, context=condition_id)
        has_outer_phase = all("signed_order_parameter" in metrics for _, metrics, _ in runs)
        for array_name in contract.required_arrays:
            reference = runs[0][2][array_name]
            for task, _, arrays in runs[1:]:
                candidate = arrays[array_name]
                if candidate.shape != reference.shape:
                    raise RuntimeError(
                        f"condition {condition_id} has incompatible {array_name} shape "
                        f"for seed {task.seed}"
                    )
                if array_name.endswith("step") and not np.array_equal(candidate, reference):
                    raise RuntimeError(
                        f"condition {condition_id} has incompatible checkpoint grid "
                        f"for seed {task.seed}"
                    )
        metric_intervals = {
            (
                "inner_susceptibility"
                if name == "susceptibility"
                and contract.phase_ensemble == "inner_replica"
                and has_outer_phase
                else (
                    "inner_binder_cumulant"
                    if name == "binder_cumulant"
                    and contract.phase_ensemble == "inner_replica"
                    and has_outer_phase
                    else name
                )
            ): _interval(
                (metrics[name] for _, metrics, _ in runs),
                (task.seed for task, _, _ in runs),
                metric_name=name,
            )
            for name in sorted(metric_names)
        }
        metric_intervals.update(_outer_phase_estimates(runs))
        records.append(
            {
                "condition_id": condition_id,
                "study": first.study,
                "domain": first.domain.value,
                "family": first.family,
                "variant": first.variant,
                "stage": first.stage,
                "control_name": first.control_name,
                "control": float(first.control),
                "size": int(first.size),
                "scale": first.scale.to_dict(),
                "finite_size_coordinate": first.finite_size_coordinate,
                "finite_size_value": first.finite_size_value,
                "parameters": dict(first.parameters),
                "seeds": sorted(seeds),
                "runner_contract": {
                    "fidelity": contract.fidelity.value,
                    "theory_status": contract.theory_status.value,
                    "phase_ensemble": contract.phase_ensemble,
                    "required_metrics": sorted(contract.required_metrics),
                },
                "metrics": metric_intervals,
            }
        )

    aggregate = {
        "schema_version": "2.0",
        "study": registered.study,
        "manifest_config_hash": registered.config_hash,
        "minimum_seed_count": REQUIRED_SEED_COUNT,
        "required_seed_count": len(registered.seeds),
        "registered_runs": len(registered.tasks),
        "completed_runs": len(registered.tasks) - len(missing),
        "missing_run_ids": missing,
        "records": records,
        "evidence": _evidence(records, len(registered.seeds)) if records else [],
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "aggregate.json.tmp"
    temporary.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    temporary.replace(target / "aggregate.json")

    metric_names = sorted({name for record in records for name in record["metrics"]})
    with (target / "summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "condition_id",
            "study",
            "domain",
            "family",
            "variant",
            "stage",
            "control_name",
            "control",
            "size",
        ]
        for name in metric_names:
            fieldnames.extend((f"{name}_mean", f"{name}_ci95", f"{name}_sd"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {name: record[name] for name in fieldnames if name in record}
            for name, interval in record["metrics"].items():
                row[f"{name}_mean"] = interval["mean"]
                row[f"{name}_ci95"] = interval["ci95"]
                row[f"{name}_sd"] = interval["sd"]
            writer.writerow(row)
    return aggregate


def read_aggregate(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "aggregate.json"
    return json.loads(source.read_text())


__all__ = ["T95_DF4", "aggregate_manifest", "read_aggregate"]
