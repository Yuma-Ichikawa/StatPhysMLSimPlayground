"""Strict registered-seed aggregation with Student-t uncertainty and evidence grades."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import csv
import json
import math

import numpy as np
from scipy.stats import t as student_t

from ..schema import Manifest, REQUIRED_SEED_COUNT, TaskSpec, read_manifest

T95_DF4 = 2.7764451051977987


def _read_completed(task: TaskSpec, root: Path) -> dict[str, float] | None:
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
    return result


def _interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size < REQUIRED_SEED_COUNT or not np.isfinite(array).all():
        raise ValueError(
            f"uncertainty requires at least {REQUIRED_SEED_COUNT} finite seeds"
        )
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(array.size)
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "mean": float(array.mean()),
        "sd": standard_deviation,
        "sem": standard_error,
        "ci95": critical * standard_error,
        "n": int(array.size),
    }


def _evidence(
    records: list[dict[str, Any]], registered_seed_count: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        parameter_key = json.dumps(record["parameters"], sort_keys=True, separators=(",", ":"))
        groups[(record["domain"], record["family"], record["variant"], parameter_key)].append(record)
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
        if (
            complete
            and confirmatory
            and len(sizes) >= 6
            and len(controls) >= 7
            and interior_peak
            and susceptibility_growth
        ):
            grade = "A"
        elif complete and len(sizes) >= 6 and len(controls) >= 5:
            grade = "B"
        elif complete and len(sizes) >= 3:
            grade = "C"
        else:
            grade = "insufficient"
        result.append(
            {
                "domain": domain,
                "family": family,
                "variant": variant,
                "parameters": json.loads(parameter_key),
                "grade": grade,
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
    grouped: dict[str, list[tuple[TaskSpec, dict[str, float]]]] = defaultdict(list)
    for task in registered.tasks:
        metrics = _read_completed(task, root)
        if metrics is None:
            missing.append(task.run_id)
            continue
        grouped[task.condition_id].append((task, metrics))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} of {len(registered.tasks)} registered runs are incomplete"
        )

    records: list[dict[str, Any]] = []
    expected_seeds = set(registered.seeds)
    for condition_id, runs in sorted(grouped.items()):
        seeds = {task.seed for task, _ in runs}
        if seeds != expected_seeds:
            if allow_incomplete:
                continue
            raise RuntimeError(f"condition {condition_id} does not contain the registered seeds")
        first = runs[0][0]
        metric_names = set.intersection(*(set(metrics) for _, metrics in runs))
        metric_intervals = {
            name: _interval(metrics[name] for _, metrics in runs)
            for name in sorted(metric_names)
        }
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
                "parameters": dict(first.parameters),
                "seeds": sorted(seeds),
                "metrics": metric_intervals,
            }
        )

    aggregate = {
        "schema_version": "1.1",
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
            "condition_id", "study", "domain", "family", "variant", "stage",
            "control_name", "control", "size",
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
