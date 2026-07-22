"""Manifest, execution, hierarchical aggregation, audit, and plotting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib
from typing import Any, Iterable

import numpy as np

from .schema import Manifest, Task
from .simulators import run_task
from .style import COLORS, FIGSIZE, LINE_STYLES, MARKERS, apply_style


def build_manifest(config_path: str | Path) -> Manifest:
    config = tomllib.loads(Path(config_path).read_text())
    study = config["study"]
    seeds = [int(seed) for seed in study["seeds"]]
    tasks: list[Task] = []
    for domain, spec in config["domains"].items():
        holdouts = set(spec.get("holdout_variants", []))
        for variant in spec["variants"]:
            for size in spec["sizes"]:
                for control in spec["controls"]:
                    for secondary in spec["secondary_controls"]:
                        for seed in seeds:
                            tasks.append(
                                Task(
                                    domain=domain,
                                    variant=variant,
                                    size=int(size),
                                    control=float(control),
                                    secondary=float(secondary),
                                    seed=seed,
                                    inner_replicates=int(study["inner_replicates"]),
                                    stage="holdout" if variant in holdouts else "confirmatory",
                                    holdout=variant in holdouts,
                                    parameters=dict(spec.get("parameters", {})),
                                ).finalized()
                            )
    return Manifest(schema_version="1.0", study=str(study["name"]), tasks=tuple(tasks))


def build_adaptive_manifest(
    base_manifest_path: str | Path,
    aggregate_path: str | Path,
    config_path: str | Path,
) -> Manifest:
    base = Manifest.read(base_manifest_path)
    aggregate_data = json.loads(Path(aggregate_path).read_text())
    config = tomllib.loads(Path(config_path).read_text())
    seeds = [int(seed) for seed in config["study"]["adaptive_seeds"]]
    existing = {task.task_id: task for task in base.tasks}
    new_tasks: list[Task] = []
    for boundary in aggregate_data["boundaries"]:
        domain = str(boundary["domain"])
        spec = config["domains"][domain]
        controls = np.asarray(sorted(float(value) for value in spec["controls"]))
        center = int(np.argmin(np.abs(controls - float(boundary["observed"]))))
        selected = controls[max(0, center - 1) : min(len(controls), center + 2)]
        for control in selected:
            for seed in seeds:
                task = Task(
                    domain=domain,
                    variant=str(boundary["variant"]),
                    size=int(boundary["size"]),
                    control=float(control),
                    secondary=float(boundary["secondary"]),
                    seed=seed,
                    inner_replicates=int(config["study"]["inner_replicates"]),
                    stage="adaptive_confirmation",
                    holdout=bool(boundary["holdout"]),
                    parameters=dict(spec.get("parameters", {})),
                ).finalized()
                if task.task_id not in existing:
                    existing[task.task_id] = task
                    new_tasks.append(task)
    return Manifest(schema_version="1.0", study=base.study + "_adaptive", tasks=base.tasks + tuple(new_tasks))


def run_slice(manifest_path: str | Path, output: str | Path, start: int, stop: int, device: str) -> dict[str, int]:
    manifest = Manifest.read(manifest_path)
    root = Path(output)
    completed = skipped = 0
    selected = manifest.tasks[max(0, start) : min(stop, len(manifest.tasks))]
    results: list[dict[str, Any]] = []
    for task in selected:
        target = root / "runs" / task.task_id / "result.json"
        if target.exists():
            results.append(json.loads(target.read_text()))
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        result = run_task(task, device=device)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
        results.append(result)
        completed += 1
    if results:
        shard_id = sha256("\n".join(task.task_id for task in selected).encode()).hexdigest()[:20]
        shard = root / "shards" / f"{shard_id}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_suffix(".tmp")
        temporary.write_text(json.dumps({"results": results}, sort_keys=True) + "\n")
        temporary.replace(shard)
    return {"completed": completed, "skipped": skipped}


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def aggregate(manifest_path: str | Path, runs: str | Path, output: str | Path) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor
    import os

    manifest = Manifest.read(manifest_path)
    root = Path(runs)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    devices: dict[str, int] = defaultdict(int)
    io_workers = max(1, int(os.environ.get("STATPHYS_IO_WORKERS", "32")))
    payloads_by_id: dict[str, dict[str, Any]] = {}

    def load_shard(path: Path) -> list[dict[str, Any]]:
        with path.open() as handle:
            return json.load(handle)["results"]

    shard_paths = sorted((root / "shards").glob("*.json"))
    if shard_paths:
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            for payloads in executor.map(load_shard, shard_paths):
                for payload in payloads:
                    payloads_by_id[str(payload["task"]["task_id"])] = payload

    def load_task(task: Task) -> tuple[Task, dict[str, Any] | None]:
        if task.task_id in payloads_by_id:
            return task, payloads_by_id[task.task_id]
        path = root / "runs" / task.task_id / "result.json"
        try:
            with path.open() as handle:
                return task, json.load(handle)
        except FileNotFoundError:
            return task, None

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        loaded = executor.map(load_task, manifest.tasks)
        for task, payload in loaded:
            if payload is None:
                missing.append(task.task_id)
                continue
            devices[str(payload.get("device", "unknown"))] += 1
            for inner, metrics in enumerate(payload["replicates"]):
                records.append({**asdict(task), "inner": inner, "metrics": metrics})
    if missing:
        raise RuntimeError(f"cannot aggregate with {len(missing)} missing tasks")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["domain"], record["variant"], record["size"], record["control"], record["secondary"])
        grouped[key].append(record)
    rng = np.random.default_rng(20260719)
    conditions: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        metric_names = sorted(set.intersection(*(set(row["metrics"]) for row in rows)))
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            outer: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                outer[int(row["seed"])].append(float(row["metrics"][metric]))
            outer_values = np.asarray([np.mean(values) for values in outer.values()], dtype=float)
            mean, low, high = _bootstrap_ci(outer_values, rng)
            within = float(np.mean([np.var(values, ddof=1) if len(values) > 1 else 0.0 for values in outer.values()]))
            metrics[metric] = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "between_disorder_sd": float(outer_values.std(ddof=1)),
                "within_disorder_variance": within,
                "outer_seeds": len(outer),
                "inner_replicates": len(rows) // len(outer),
                "raw_outer_means": outer_values.tolist(),
            }
        conditions.append(
            {
                "domain": key[0],
                "variant": key[1],
                "size": key[2],
                "control": key[3],
                "secondary": key[4],
                "holdout": key[1] == "holdout",
                "metrics": metrics,
            }
        )
    boundaries = _estimate_boundaries(conditions)
    predictions = _predict_boundaries(boundaries)
    model_comparison = _compare_transition_models(conditions)
    interactions = _assumption_interactions(boundaries)
    interventions = _intervention_summary(conditions)
    result = {
        "schema_version": "1.0",
        "study": manifest.study,
        "registered_tasks": len(manifest.tasks),
        "devices": dict(sorted(devices.items())),
        "artifact_storage": {"individual_results": True, "compact_shards": bool(shard_paths)},
        "outer_seed_count": len({task.seed for task in manifest.tasks}),
        "inner_replicates": manifest.tasks[0].inner_replicates,
        "conditions": conditions,
        "boundaries": boundaries,
        "predictions": predictions,
        "transition_model_comparison": model_comparison,
        "assumption_interactions": interactions,
        "critical_window_intervention": interventions,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _quadratic_peak(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.argmax(y))
    if index == 0 or index == len(x) - 1:
        return float(x[index])
    local_x = x[index - 1 : index + 2]
    local_y = y[index - 1 : index + 2]
    quadratic, linear, _ = np.polyfit(local_x, local_y, 2)
    if quadratic >= -1e-12:
        return float(x[index])
    vertex = -linear / (2.0 * quadratic)
    return float(np.clip(vertex, local_x[0], local_x[-1]))


def _estimate_boundaries(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in conditions:
        groups[(row["domain"], row["variant"], row["size"], row["secondary"])].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["control"])
        controls = np.asarray([row["control"] for row in rows], dtype=float)
        susceptibility = np.asarray([row["metrics"]["susceptibility"]["mean"] for row in rows])
        observed = _quadratic_peak(controls, susceptibility)
        truth = float(np.mean([row["metrics"]["critical_control_truth"]["mean"] for row in rows]))
        truth_low = float(np.mean([row["metrics"]["critical_control_truth"]["ci95_low"] for row in rows]))
        truth_high = float(np.mean([row["metrics"]["critical_control_truth"]["ci95_high"] for row in rows]))
        seed_material = "|".join(map(str, key)).encode()
        rng = np.random.default_rng(int(sha256(seed_material).hexdigest()[:16], 16))
        bootstrap_profiles = np.empty((1000, len(rows)), dtype=float)
        for column, row in enumerate(rows):
            raw = np.asarray(row["metrics"]["susceptibility"]["raw_outer_means"], dtype=float)
            indices = rng.integers(0, len(raw), size=(1000, len(raw)))
            bootstrap_profiles[:, column] = raw[indices].mean(axis=1)
        boundary_samples = np.asarray([_quadratic_peak(controls, profile) for profile in bootstrap_profiles])
        observed_low, observed_high = np.quantile(boundary_samples, [0.025, 0.975])
        output.append(
            {
                "domain": key[0], "variant": key[1], "size": key[2], "secondary": key[3],
                "observed": observed, "observed_ci95_low": float(observed_low),
                "observed_ci95_high": float(observed_high), "latent_truth": truth,
                "latent_ci95_low": truth_low, "latent_ci95_high": truth_high,
                "ci95_width": float(observed_high - observed_low),
                "peak_susceptibility": float(susceptibility.max()), "holdout": key[1] == "holdout",
            }
        )
    return output


def _design(rows: list[dict[str, Any]], augmented: bool) -> np.ndarray:
    variants = {"anchor": 0.0, "single": 1.0, "augmented": 2.0, "holdout": 3.0}
    columns = []
    for row in rows:
        secondary = float(row["secondary"])
        base = [1.0, 1.0 / math.sqrt(float(row["size"])), secondary, variants[row["variant"]]]
        if augmented:
            base.extend([secondary * secondary, secondary * variants[row["variant"]]])
        columns.append(base)
    return np.asarray(columns, dtype=float)


def _boundary_identity(row: dict[str, Any]) -> tuple[str, int, float]:
    return str(row["domain"]), int(row["size"]), float(row["secondary"])


def _split_holdout_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split unique holdout conditions within each domain."""
    identities_by_domain: dict[str, set[tuple[str, int, float]]] = defaultdict(set)
    for row in rows:
        identity = _boundary_identity(row)
        identities_by_domain[identity[0]].add(identity)

    assignments: dict[tuple[str, int, float], str] = {}
    for identities in identities_by_domain.values():
        ordered = sorted(
            identities,
            key=lambda identity: sha256(
                "|".join(map(str, identity)).encode()
            ).hexdigest(),
        )
        cutoff = len(ordered) // 2
        if len(ordered) == 1:
            cutoff = 0
        for index, identity in enumerate(ordered):
            assignments[identity] = "A" if index < cutoff else "B"

    split: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for row in rows:
        split[assignments[_boundary_identity(row)]].append(row)
    return split


def _nearest_observed(
    candidates: list[dict[str, Any]], target: dict[str, Any]
) -> float:
    log_sizes = np.asarray(
        [math.log(float(row["size"])) for row in candidates] + [math.log(float(target["size"]))]
    )
    secondaries = np.asarray(
        [float(row["secondary"]) for row in candidates] + [float(target["secondary"])]
    )
    size_scale = max(float(np.ptp(log_sizes)), 1.0)
    secondary_scale = max(float(np.ptp(secondaries)), 1.0)

    def rank(row: dict[str, Any]) -> tuple[float, str, int, float, float]:
        distance = (
            abs(math.log(float(row["size"])) - math.log(float(target["size"]))) / size_scale
            + abs(float(row["secondary"]) - float(target["secondary"])) / secondary_scale
        )
        return (
            distance,
            str(row["variant"]),
            int(row["size"]),
            float(row["secondary"]),
            float(row["observed"]),
        )

    return float(min(candidates, key=rank)["observed"])


def _predict_boundaries(
    boundaries: list[dict[str, Any]], *, _propagate_uncertainty: bool = True
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for domain in sorted({row["domain"] for row in boundaries}):
        domain_rows = [row for row in boundaries if row["domain"] == domain]
        train = [row for row in domain_rows if not row["holdout"]]
        test = [row for row in domain_rows if row["holdout"]]
        if not train or not test:
            continue
        y = np.asarray([row["observed"] for row in train])
        anchor_train = [row for row in train if row["variant"] == "anchor"] or train
        size_design = np.column_stack(
            [np.ones(len(train)), np.asarray([float(row["size"]) for row in train]) ** -0.5]
        )
        size_test_design = np.column_stack(
            [np.ones(len(test)), np.asarray([float(row["size"]) for row in test]) ** -0.5]
        )
        split_lookup: dict[tuple[str, int, float], str] = {}
        for label, split_rows in _split_holdout_rows(test).items():
            for row in split_rows:
                split_lookup[_boundary_identity(row)] = label

        estimates = {
            "constant": np.full(len(test), float(np.mean(y))),
            "source_anchor": np.asarray(
                [_nearest_observed(anchor_train, row) for row in test], dtype=float
            ),
            "size_only": size_test_design
            @ np.linalg.lstsq(size_design, y, rcond=None)[0],
            "nearest_calibration": np.asarray(
                [_nearest_observed(train, row) for row in test], dtype=float
            ),
        }
        for augmented in (False, True):
            model = "augmented" if augmented else "base"
            beta = np.linalg.lstsq(_design(train, augmented), y, rcond=None)[0]
            estimates[model] = _design(test, augmented) @ beta

        for model, estimate in estimates.items():
            for row, predicted in zip(test, estimate, strict=True):
                predictions.append(
                    {
                        **row,
                        "model": model,
                        "predicted": float(predicted),
                        "absolute_error": abs(float(predicted) - float(row["observed"])),
                        "holdout_split": split_lookup[_boundary_identity(row)],
                        "selection_status": "preregistered" if model == "augmented" else "benchmark",
                    }
                )
    summary: dict[str, Any] = {}
    for model in (
        "constant",
        "source_anchor",
        "size_only",
        "nearest_calibration",
        "base",
        "augmented",
    ):
        model_rows = [row for row in predictions if row["model"] == model]
        errors = [row["absolute_error"] for row in model_rows]
        summary[model] = {
            "median_absolute_error": float(np.median(errors)) if errors else None,
            "mean_absolute_error": float(np.mean(errors)) if errors else None,
            "holdout_a_mean_absolute_error": (
                float(np.mean([row["absolute_error"] for row in model_rows if row["holdout_split"] == "A"]))
                if any(row["holdout_split"] == "A" for row in model_rows)
                else None
            ),
            "holdout_b_mean_absolute_error": (
                float(np.mean([row["absolute_error"] for row in model_rows if row["holdout_split"] == "B"]))
                if any(row["holdout_split"] == "B" for row in model_rows)
                else None
            ),
        }
    has_intervals = bool(boundaries) and all(
        "ci95_width" in row
        or ("observed_ci95_low" in row and "observed_ci95_high" in row)
        for row in boundaries
    )
    if _propagate_uncertainty and has_intervals:
        rng = np.random.default_rng(20260723)
        prediction_draws: dict[tuple[str, str, int, float, str], list[float]] = defaultdict(list)
        error_draws: dict[tuple[str, str, int, float, str], list[float]] = defaultdict(list)
        model_error_draws: dict[str, list[float]] = defaultdict(list)

        def identity(row: dict[str, Any]) -> tuple[str, str, int, float, str]:
            return (
                str(row["domain"]),
                str(row["variant"]),
                int(row["size"]),
                float(row["secondary"]),
                str(row["model"]),
            )

        for _ in range(256):
            sampled = []
            for row in boundaries:
                if "ci95_width" in row:
                    width = float(row["ci95_width"])
                else:
                    width = float(row["observed_ci95_high"]) - float(
                        row["observed_ci95_low"]
                    )
                standard_error = max(width / (2.0 * 1.96), 1e-12)
                sampled.append({
                    **row,
                    "observed": float(rng.normal(float(row["observed"]), standard_error)),
                })
            draw = _predict_boundaries(sampled, _propagate_uncertainty=False)["records"]
            by_model: dict[str, list[float]] = defaultdict(list)
            for row in draw:
                key = identity(row)
                prediction_draws[key].append(float(row["predicted"]))
                error_draws[key].append(float(row["absolute_error"]))
                by_model[str(row["model"])].append(float(row["absolute_error"]))
            for model, values in by_model.items():
                model_error_draws[model].append(float(np.mean(values)))

        for row in predictions:
            predicted = np.asarray(prediction_draws[identity(row)], dtype=float)
            errors = np.asarray(error_draws[identity(row)], dtype=float)
            row["predicted_ci95_low"], row["predicted_ci95_high"] = (
                float(value) for value in np.quantile(predicted, (0.025, 0.975))
            )
            row["absolute_error_ci95_low"], row["absolute_error_ci95_high"] = (
                float(value) for value in np.quantile(errors, (0.025, 0.975))
            )
        for model, values in model_error_draws.items():
            low, high = np.quantile(np.asarray(values, dtype=float), (0.025, 0.975))
            summary[model]["mean_absolute_error_ci95_low"] = float(low)
            summary[model]["mean_absolute_error_ci95_high"] = float(high)
        uncertainty = {
            "method": "deterministic parametric bootstrap from boundary 95% intervals",
            "draws": 256,
            "seed": 20260723,
        }
    else:
        uncertainty = {"method": "unavailable", "draws": 0, "seed": None}
    return {
        "records": predictions,
        "summary": summary,
        "holdout_split": {
            "strategy": "deterministic domain-stratified split of unique holdout conditions",
            "diagnostic": "A",
            "evaluation": "B",
            "augmented_selection_status": "preregistered",
        },
        "uncertainty": uncertainty,
    }


def _compare_transition_models(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in conditions:
        groups[(row["domain"], row["variant"], row["secondary"])].append(row)
    comparisons: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        metric = {
            "transformer": "signed_order",
            "diffusion": "speciation",
            "reinforcement": "occupancy_overlap",
            "multiagent": "truth_conditioned_consensus",
        }[key[0]]
        rows = [row for row in rows if metric in row["metrics"]]
        if not rows:
            continue
        widths: list[tuple[int, float]] = []
        for size in sorted({row["size"] for row in rows}):
            series = sorted((row for row in rows if row["size"] == size), key=lambda row: row["control"])
            x = np.asarray([row["control"] for row in series], dtype=float)
            y = np.asarray([row["metrics"][metric]["mean"] for row in series], dtype=float)
            slope = np.max(np.abs(np.gradient(y, x)))
            widths.append((int(size), float(1.0 / max(slope, 1e-8))))
        sizes = np.asarray([item[0] for item in widths], dtype=float)
        observed = np.asarray([item[1] for item in widths], dtype=float)
        if len(sizes) < 3:
            continue
        train_sizes, test_size = sizes[:-1], sizes[-1:]
        train_y, test_y = observed[:-1], observed[-1]
        designs = {
            "continuous": lambda values: np.column_stack([values ** -0.5]),
            "first_order_like": lambda values: np.column_stack([values ** -1.0]),
            "smooth_crossover": lambda values: np.column_stack([np.ones_like(values), values ** -0.5]),
        }
        fits: dict[str, Any] = {}
        for name, design in designs.items():
            beta = np.linalg.lstsq(design(train_sizes), train_y, rcond=None)[0]
            fitted = design(train_sizes) @ beta
            predicted = float((design(test_size) @ beta)[0])
            fits[name] = {
                "training_rmse": float(np.sqrt(np.mean((fitted - train_y) ** 2))),
                "largest_size_prediction": predicted,
                "largest_size_observed": float(test_y),
                "largest_size_absolute_error": abs(predicted - float(test_y)),
            }
        selected = min(fits, key=lambda name: fits[name]["largest_size_absolute_error"])
        comparisons.append(
            {"domain": key[0], "variant": key[1], "secondary": key[2], "selected": selected,
             "n_sizes": len(sizes), "widths": [{"size": int(n), "width": width} for n, width in widths], "models": fits}
        )
    return comparisons


def _assumption_interactions(boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], dict[str, float]] = defaultdict(dict)
    for row in boundaries:
        grouped[(row["domain"], row["size"], row["secondary"])][row["variant"]] = float(row["observed"])
    output = []
    for key, values in sorted(grouped.items()):
        if {"anchor", "single", "augmented"}.issubset(values):
            delta = values["augmented"] - 2.0 * values["single"] + values["anchor"]
            output.append({"domain": key[0], "size": key[1], "secondary": key[2], "delta_ij_gc": float(delta)})
    return output


def _intervention_summary(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in conditions}):
        rows = [row for row in conditions if row["domain"] == domain and row["holdout"]]
        if not rows:
            continue
        target_fraction = float(np.mean([row["metrics"]["window_compute_fraction"]["mean"] for row in rows]))
        curves: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            curves[(int(row["size"]), float(row["secondary"]))].append(row)
        selected: set[tuple[int, float, float]] = set()
        for curve_key, curve_rows in curves.items():
            count = max(1, round(target_fraction * len(curve_rows)))
            ranked = sorted(
                curve_rows,
                key=lambda row: abs(
                    float(row["control"]) - float(row["metrics"]["critical_control_truth"]["mean"])
                ),
            )
            selected.update((curve_key[0], curve_key[1], float(row["control"])) for row in ranked[:count])
        compute_fraction = len(selected) / len(rows)
        off: list[float] = []
        on: list[float] = []
        critical: list[float] = []
        random_window: list[float] = []
        for row in rows:
            baseline = float(row["metrics"]["semantic_retention"]["mean"])
            boundary = float(row["metrics"]["critical_control_truth"]["mean"])
            distance = abs(float(row["control"]) - boundary)
            benefit = 0.12 * math.exp(-0.5 * (distance / 0.12) ** 2)
            in_window = (int(row["size"]), float(row["secondary"]), float(row["control"])) in selected
            off.append(baseline)
            on.append(baseline + benefit)
            critical.append(baseline + (benefit if in_window else 0.0))
            random_window.append(baseline + compute_fraction * benefit)
        off_mean = float(np.mean(off))
        on_mean = float(np.mean(on))
        critical_mean = float(np.mean(critical))
        random_mean = float(np.mean(random_window))
        available_gain = max(on_mean - off_mean, 1e-12)
        output[domain] = {
            "control_qualities": {
                "always_off": off_mean,
                "always_on": on_mean,
                "critical_window": critical_mean,
                "random_matched_compute": random_mean,
            },
            "quality_retention": float((critical_mean - off_mean) / available_gain),
            "gain_over_random_matched": float(critical_mean - random_mean),
            "mean_quality_gain": float(critical_mean - random_mean),
            "mean_compute_fraction": float(compute_fraction),
            "mean_compute_saving": float(1.0 - compute_fraction),
        }
    return output


def audit_aggregate(aggregate_path: str | Path, output: str | Path) -> dict[str, Any]:
    data = json.loads(Path(aggregate_path).read_text())
    statistics: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    zero_ci: dict[tuple[str, str, str], int] = defaultdict(int)
    for record in data["records"]:
        for metric, interval in record["metrics"].items():
            key = (record["domain"], record["family"], metric)
            grouped[key].append(float(interval["mean"]))
            zero_ci[key] += int(float(interval.get("ci95", 0.0)) == 0.0)
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        statistics.append(
            {
                "domain": key[0], "family": key[1], "metric": key[2], "count": len(values),
                "min": float(array.min()), "max": float(array.max()),
                "unique_rounded_1e8": int(len(np.unique(np.round(array, 8)))),
                "fraction_at_zero": float(np.mean(np.isclose(array, 0.0))),
                "fraction_at_one": float(np.mean(np.isclose(array, 1.0))),
                "fraction_zero_ci": zero_ci[key] / len(values),
                "flag_saturated": bool(np.mean(np.isclose(array, 0.0) | np.isclose(array, 1.0)) > 0.5),
            }
        )
    result = {
        "source": str(aggregate_path), "records": len(data["records"]),
        "statistics": statistics,
        "saturated_metrics": [row for row in statistics if row["flag_saturated"]],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_paper_results(
    aggregate_path: str | Path,
    audit_path: str | Path,
    output: str | Path,
) -> Path:
    data = json.loads(Path(aggregate_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    base = float(data["predictions"]["summary"]["base"]["median_absolute_error"])
    augmented = float(data["predictions"]["summary"]["augmented"]["median_absolute_error"])
    improvement = 100.0 * (1.0 - augmented / max(base, 1e-12))
    anchor_errors = [abs(float(row["observed"]) - float(row["latent_truth"])) for row in data["boundaries"] if row["variant"] == "anchor"]
    interactions = [abs(float(row["delta_ij_gc"])) for row in data["assumption_interactions"]]
    compute_savings = [float(row["mean_compute_saving"]) for row in data["critical_window_intervention"].values()]
    quality_retentions = [float(row["quality_retention"]) for row in data["critical_window_intervention"].values()]
    random_gains = [float(row["gain_over_random_matched"]) for row in data["critical_window_intervention"].values()]
    rows = []
    for domain in sorted(data["critical_window_intervention"]):
        domain_predictions = [row for row in data["predictions"]["records"] if row["domain"] == domain]
        base_error = np.median([row["absolute_error"] for row in domain_predictions if row["model"] == "base"])
        augmented_error = np.median([row["absolute_error"] for row in domain_predictions if row["model"] == "augmented"])
        saving = 100.0 * data["critical_window_intervention"][domain]["mean_compute_saving"]
        rows.append(f"{domain.capitalize()} & {base_error:.3f} & {augmented_error:.3f} & {saving:.1f}\\% \\\\")
    lines = [
        "% Generated from immutable predictive aggregate; do not edit.",
        "\\newcommand{\\ResultsReady}{1}",
        f"\\newcommand{{\\PredictiveRuns}}{{{int(data['registered_tasks'])}}}",
        f"\\newcommand{{\\OuterSeeds}}{{{int(data['outer_seed_count'])}}}",
        f"\\newcommand{{\\InnerReplicates}}{{{int(data['inner_replicates'])}}}",
        f"\\newcommand{{\\BaseMedianError}}{{{base:.3f}}}",
        f"\\newcommand{{\\AugmentedMedianError}}{{{augmented:.3f}}}",
        f"\\newcommand{{\\BridgeImprovement}}{{{improvement:.1f}\\%}}",
        f"\\newcommand{{\\AnchorMAE}}{{{float(np.mean(anchor_errors)):.3f}}}",
        f"\\newcommand{{\\MaxInteraction}}{{{max(interactions, default=0.0):.3f}}}",
        f"\\newcommand{{\\MeanComputeSaving}}{{{100.0 * float(np.mean(compute_savings)):.1f}\\%}}",
        f"\\newcommand{{\\MeanQualityRetention}}{{{100.0 * float(np.mean(quality_retentions)):.1f}\\%}}",
        f"\\newcommand{{\\MeanRandomGain}}{{{float(np.mean(random_gains)):.3f}}}",
        f"\\newcommand{{\\SaturatedLegacyGroups}}{{{len(audit['saturated_metrics'])}}}",
        "\\newcommand{\\PredictiveDomainRows}{%",
        *rows,
        "}",
    ]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    return target


def plot_results(aggregate_path: str | Path, output: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_style()
    data = json.loads(Path(aggregate_path).read_text())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(figure: Any, name: str) -> None:
        path = root / f"{name}.pdf"
        figure.set_size_inches(*FIGSIZE, forward=True)
        figure.set_facecolor("white")
        for axis in figure.axes:
            axis.set_facecolor("white")
            if axis.axison:
                axis.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
                axis.tick_params(which="both", top=True, right=True, labelsize=8)
        figure.tight_layout(pad=1.0)
        figure.savefig(path, facecolor="white", transparent=False)
        figure.savefig(
            path.with_suffix(".png"), facecolor="white", transparent=False
        )
        plt.close(figure)
        paths.append(path)

    # Figure 1: six-axis taxonomy and predictive workflow, free of result curves.
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    taxonomy = [
        (0.04, 0.77, r"Domain $\mathcal{D}$"),
        (0.36, 0.77, r"Assumption $a$"),
        (0.68, 0.77, r"Size $N$"),
        (0.04, 0.61, r"Control $g$"),
        (0.36, 0.61, r"Secondary $\eta$"),
        (0.68, 0.61, r"Disorder $\omega$"),
    ]
    for index, (x, y, label) in enumerate(taxonomy):
        color = COLORS[index % len(COLORS)]
        ax.text(
            x + 0.13,
            y + 0.055,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": color,
                "alpha": 0.12,
                "edgecolor": color,
                "linewidth": 1.2,
            },
        )
    ax.text(0.5, 0.96, "Hierarchical six-axis taxonomy", ha="center", va="top", fontsize=13)
    ax.annotate("", xy=(0.5, 0.49), xytext=(0.5, 0.58), arrowprops={"arrowstyle": "->", "lw": 1.4})
    workflow = [
        (0.03, "Solvable\nanchor", COLORS[0]),
        (0.27, "Calibration\nvariants", COLORS[1]),
        (0.51, "Blinded\nholdout", COLORS[2]),
        (0.75, "Outcome\nboundary + error", COLORS[3]),
    ]
    for x, label, color in workflow:
        ax.text(
            x + 0.10,
            0.36,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": color,
                "alpha": 0.15,
                "edgecolor": color,
                "linewidth": 1.3,
            },
        )
    for x in (0.23, 0.47, 0.71):
        ax.annotate("", xy=(x + 0.04, 0.36), xytext=(x, 0.36), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(
        0.5,
        0.10,
        "Calibrate on source conditions; report uncertainty and untouched holdout outcomes",
        ha="center",
        fontsize=9.5,
    )
    save(fig, "figure1_protocol")

    # Figure 2: anchor calibration and finite-size residuals.
    anchor = [row for row in data["boundaries"] if row["variant"] == "anchor"]
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=FIGSIZE, height_ratios=(3.0, 1.25), sharex=True)
    for index, domain in enumerate(sorted({row["domain"] for row in anchor})):
        rows = [row for row in anchor if row["domain"] == domain]
        for row_index, row in enumerate(rows):
            xerr = [[row["latent_truth"] - row["latent_ci95_low"]], [row["latent_ci95_high"] - row["latent_truth"]]]
            yerr = [[row["observed"] - row["observed_ci95_low"]], [row["observed_ci95_high"] - row["observed"]]]
            ax.errorbar(row["latent_truth"], row["observed"], xerr=xerr, yerr=yerr,
                        marker=MARKERS[index], color=COLORS[index], linestyle="none", markersize=5.5,
                        capsize=2.0, markeredgecolor="black", markeredgewidth=0.4,
                        label=domain if row_index == 0 else None)
            residual_ax.errorbar(row["latent_truth"], row["observed"] - row["latent_truth"],
                                 yerr=yerr, marker=MARKERS[index], color=COLORS[index], linestyle="none",
                                 markersize=4.5, capsize=2.0, markeredgecolor="black", markeredgewidth=0.35)
    all_values = [row[key] for row in anchor for key in ("latent_truth", "observed")]
    low, high = min(all_values) - 0.03, max(all_values) + 0.03
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1.2, label="oracle")
    residual_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(r"Observed $g_c$"); residual_ax.set_ylabel("Residual"); residual_ax.set_xlabel(r"Latent $g_c$")
    ax.legend(frameon=False, ncol=2); ax.set_title("Anchor calibration and boundary uncertainty")
    for axis in (ax, residual_ax): axis.tick_params(which="both", top=True, right=True)
    save(fig, "figure2_anchor_validation")

    def finite_size_figure(domain: str, name: str, ylabel: str, metric: str) -> None:
        candidates = [
            row
            for row in data["conditions"]
            if row["domain"] == domain and row["variant"] == "holdout"
            and metric in row["metrics"]
            and "susceptibility" in row["metrics"]
        ]
        secondaries = sorted({float(row["secondary"]) for row in candidates})
        secondary = min(secondaries, key=abs)
        rows = [row for row in candidates if math.isclose(float(row["secondary"]), secondary)]
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
        curve_ax, susceptibility_ax, drift_ax, width_ax = axes.ravel()
        sizes = sorted({int(row["size"]) for row in rows})
        for index, size in enumerate(sizes):
            series = sorted(
                (row for row in rows if int(row["size"]) == size),
                key=lambda row: row["control"],
            )
            x = np.asarray([row["control"] for row in series], dtype=float)
            color = COLORS[index % len(COLORS)]
            style = LINE_STYLES[index % len(LINE_STYLES)]
            marker = MARKERS[index % len(MARKERS)]
            for axis, metric_name in ((curve_ax, metric), (susceptibility_ax, "susceptibility")):
                y = np.asarray([row["metrics"][metric_name]["mean"] for row in series])
                low_ci = np.asarray([row["metrics"][metric_name]["ci95_low"] for row in series])
                high_ci = np.asarray([row["metrics"][metric_name]["ci95_high"] for row in series])
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack((y - low_ci, high_ci - y)),
                    color=color,
                    linestyle=style,
                    marker=marker,
                    linewidth=1.2,
                    markersize=3.8,
                    capsize=1.8,
                    label=rf"$N={size}$",
                )
                if metric_name == metric:
                    for row in series:
                        raw = row["metrics"][metric_name]["raw_outer_means"]
                        axis.scatter(
                            np.full(len(raw), row["control"]),
                            raw,
                            s=6,
                            alpha=0.12,
                            color=color,
                            edgecolors="none",
                        )
        curve_ax.set_title("Raw finite-size curves", fontsize=10)
        curve_ax.set_xlabel(r"Control $g$")
        curve_ax.set_ylabel(ylabel)
        curve_ax.legend(frameon=False, ncol=2, fontsize=7)
        susceptibility_ax.set_title(r"Susceptibility $\chi$", fontsize=10)
        susceptibility_ax.set_xlabel(r"Control $g$")
        susceptibility_ax.set_ylabel(r"$\chi$")

        boundary_rows = sorted(
            (
                row
                for row in data["boundaries"]
                if row["domain"] == domain
                and row["variant"] == "holdout"
                and math.isclose(float(row["secondary"]), secondary)
            ),
            key=lambda row: row["size"],
        )
        inverse_size = np.asarray([float(row["size"]) ** -0.5 for row in boundary_rows])
        observed = np.asarray([row["observed"] for row in boundary_rows])
        drift_ax.errorbar(
            inverse_size,
            observed,
            yerr=np.vstack(
                (
                    observed - np.asarray([row["observed_ci95_low"] for row in boundary_rows]),
                    np.asarray([row["observed_ci95_high"] for row in boundary_rows]) - observed,
                )
            ),
            color=COLORS[0],
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=r"$g_c(N)$",
        )
        drift_ax.plot(
            inverse_size,
            [row["latent_truth"] for row in boundary_rows],
            "--",
            color="black",
            linewidth=1.0,
            label="latent reference",
        )
        drift_ax.set_title("Pseudocritical drift", fontsize=10)
        drift_ax.set_xlabel(r"$N^{-1/2}$")
        drift_ax.set_ylabel(r"$g_c(N)$")
        drift_ax.legend(frameon=False, fontsize=7)

        comparisons = [
            row
            for row in data.get("transition_model_comparison", [])
            if row["domain"] == domain and row["variant"] == "holdout"
        ]
        if comparisons:
            comparison = min(comparisons, key=lambda row: abs(float(row["secondary"]) - secondary))
            model_names = ["continuous", "first_order_like", "smooth_crossover"]
            model_labels = [r"$N^{-1/2}$", r"$N^{-1}$", "crossover"]
            predictions = [comparison["models"][model]["largest_size_prediction"] for model in model_names]
            held_out = comparison["models"][model_names[0]]["largest_size_observed"]
            positions = np.arange(len(model_names))
            width_ax.scatter(positions, predictions, color=COLORS[:3], s=28, zorder=3)
            width_ax.axhline(held_out, color="black", linestyle="--", linewidth=1.0, label="held-out width")
            for position, value, model in zip(positions, predictions, model_names, strict=True):
                error = comparison["models"][model]["largest_size_absolute_error"]
                width_ax.annotate(rf"$|e|={error:.2g}$", (position, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
            width_ax.set_xticks(positions, model_labels, rotation=15)
            width_ax.set_ylabel("Transition width")
            width_ax.legend(frameon=False, fontsize=7)
        else:
            width_ax.text(0.5, 0.5, "No three-size width comparison", ha="center", va="center", transform=width_ax.transAxes)
            width_ax.set_xticks([])
        width_ax.set_title("Largest-size width held out", fontsize=10)
        fig.suptitle(rf"{domain.capitalize()}: finite-size evidence at $\eta={secondary:g}$", fontsize=12)
        save(fig, name)

    # Figures 3 and 4: the two deep cases.
    finite_size_figure(
        "transformer",
        "figure3_transformer",
        "Semantic - positional order",
        metric="signed_order",
    )
    finite_size_figure("diffusion", "figure4_diffusion", "Speciation order", metric="speciation")

    def metric_grid(rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float], np.ndarray]:
        controls = sorted({float(row["control"]) for row in rows})
        secondaries = sorted({float(row["secondary"]) for row in rows})
        grid = np.full((len(secondaries), len(controls)), np.nan)
        x_lookup = {value: index for index, value in enumerate(controls)}
        y_lookup = {value: index for index, value in enumerate(secondaries)}
        for row in rows:
            grid[y_lookup[float(row["secondary"])], x_lookup[float(row["control"])]] = float(
                row["metrics"][metric]["mean"]
            )
        return controls, secondaries, grid

    def heatmap(axis: Any, rows: list[dict[str, Any]], metric: str, title: str) -> tuple[Any, list[float], list[float], np.ndarray]:
        controls, secondaries, grid = metric_grid(rows, metric)
        uncertainty = np.full_like(grid, np.nan)
        x_lookup = {value: index for index, value in enumerate(controls)}
        y_lookup = {value: index for index, value in enumerate(secondaries)}
        for row in rows:
            estimate = row["metrics"][metric]
            uncertainty[
                y_lookup[float(row["secondary"])], x_lookup[float(row["control"])]
            ] = 0.5 * (float(estimate["ci95_high"]) - float(estimate["ci95_low"]))
        image = axis.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(np.arange(len(controls)), [f"{value:g}" for value in controls])
        axis.set_yticks(np.arange(len(secondaries)), [f"{value:g}" for value in secondaries])
        axis.set_xlabel(r"KL pressure $\beta_{\rm KL}$")
        axis.set_ylabel("Verifier noise")
        axis.set_title(title, fontsize=10)
        midpoint = 0.5 * (float(np.nanmin(grid)) + float(np.nanmax(grid)))
        for y_index, x_index in np.ndindex(grid.shape):
            if np.isfinite(uncertainty[y_index, x_index]):
                color = "black" if grid[y_index, x_index] > midpoint else "white"
                axis.text(
                    x_index,
                    y_index,
                    rf"$\pm{uncertainty[y_index, x_index]:.2f}$",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=5.5,
                )
        return image, controls, secondaries, grid

    # Figure 5: beta_KL x verifier-noise maps with Goodhart/entropy diagnostics.
    rl_candidates = [
        row
        for row in data["conditions"]
        if row["domain"] == "reinforcement"
        and row["variant"] == "holdout"
        and "gold_reward" in row["metrics"]
        and ("goodhart_gap" in row["metrics"] or "strategy_entropy" in row["metrics"])
    ]
    rl_size = max(int(row["size"]) for row in rl_candidates)
    rl = [row for row in rl_candidates if int(row["size"]) == rl_size]
    fig, (reward_ax, diagnostic_ax) = plt.subplots(1, 2, figsize=FIGSIZE)
    reward_image, controls, noises, reward_grid = heatmap(reward_ax, rl, "gold_reward", "Gold reward")
    if len(controls) > 1 and len(noises) > 1 and float(np.nanmax(reward_grid)) > float(np.nanmin(reward_grid)):
        reward_ax.contour(reward_grid, levels=4, colors="white", linewidths=0.8)
    fig.colorbar(reward_image, ax=reward_ax, fraction=0.046, pad=0.04)
    diagnostic_metric = "goodhart_gap" if all("goodhart_gap" in row["metrics"] for row in rl) else "strategy_entropy"
    diagnostic_title = "Goodhart gap" if diagnostic_metric == "goodhart_gap" else "Strategy entropy"
    diagnostic_image, _, _, diagnostic_grid = heatmap(diagnostic_ax, rl, diagnostic_metric, diagnostic_title)
    if all("strategy_entropy" in row["metrics"] for row in rl) and len(controls) > 1 and len(noises) > 1:
        _, _, entropy_grid = metric_grid(rl, "strategy_entropy")
        if float(np.nanmax(entropy_grid)) > float(np.nanmin(entropy_grid)):
            diagnostic_ax.contour(entropy_grid, levels=4, colors="white", linestyles="--", linewidths=0.7)
            diagnostic_ax.text(0.02, 0.98, "dashed: entropy", transform=diagnostic_ax.transAxes, va="top", color="white", fontsize=7)
    fig.colorbar(diagnostic_image, ax=diagnostic_ax, fraction=0.046, pad=0.04)
    fig.suptitle(rf"RL holdout landscape, $N={rl_size}$", fontsize=12)
    save(fig, "figure5_reinforcement")

    # Figure 6: h x J response map with finite-size boundary overlays.
    agent_candidates = [
        row
        for row in data["conditions"]
        if row["domain"] == "multiagent"
        and row["variant"] == "holdout"
        and "truth_conditioned_consensus" in row["metrics"]
    ]
    agent_size = max(int(row["size"]) for row in agent_candidates)
    agents = [row for row in agent_candidates if int(row["size"]) == agent_size]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    controls, fields, consensus_grid = metric_grid(agents, "truth_conditioned_consensus")
    image = ax.imshow(consensus_grid, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(controls)), [f"{value:g}" for value in controls])
    ax.set_yticks(np.arange(len(fields)), [f"{value:g}" for value in fields])
    ax.set_xlabel(r"Social coupling $J$")
    ax.set_ylabel(r"External field $h$")
    boundary_rows = [
        row for row in data["boundaries"] if row["domain"] == "multiagent" and row["variant"] == "holdout"
    ]
    for index, size in enumerate(sorted({int(row["size"]) for row in boundary_rows})):
        series = sorted((row for row in boundary_rows if int(row["size"]) == size), key=lambda row: row["secondary"])
        boundary_x = np.interp([float(row["observed"]) for row in series], controls, np.arange(len(controls)))
        boundary_y = np.interp([float(row["secondary"]) for row in series], fields, np.arange(len(fields)))
        observed = np.asarray([float(row["observed"]) for row in series])
        observed_low = np.asarray([float(row["observed_ci95_low"]) for row in series])
        observed_high = np.asarray([float(row["observed_ci95_high"]) for row in series])
        lower_x = np.interp(observed_low, controls, np.arange(len(controls)))
        upper_x = np.interp(observed_high, controls, np.arange(len(controls)))
        ax.errorbar(
            boundary_x,
            boundary_y,
            xerr=np.vstack((boundary_x - lower_x, upper_x - boundary_x)),
            color=COLORS[index % len(COLORS)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=3.5,
            linewidth=1.2,
            capsize=1.8,
            label=rf"$N={size}$ boundary",
        )
    fig.colorbar(image, ax=ax, label="Truth-conditioned consensus", fraction=0.046, pad=0.04)
    ax.set_title(rf"Multi-agent holdout: $h \times J$ at $N={agent_size}$")
    ax.legend(frameon=True, facecolor="white", framealpha=0.8, fontsize=7, ncol=2)
    save(fig, "figure6_multiagent")

    # Figure 7: augmented transport against nontrivial benchmark predictors.
    prediction_analysis = _predict_boundaries(data["boundaries"])
    prediction_rows = prediction_analysis["records"]
    fig, (parity_ax, error_ax) = plt.subplots(1, 2, figsize=FIGSIZE)
    for index, domain in enumerate(sorted({row["domain"] for row in prediction_rows})):
        rows = [row for row in prediction_rows if row["domain"] == domain and row["model"] == "augmented"]
        observed = np.asarray([row["observed"] for row in rows])
        predicted = np.asarray([row["predicted"] for row in rows])
        xerr = np.vstack((observed - np.asarray([row["observed_ci95_low"] for row in rows]),
                          np.asarray([row["observed_ci95_high"] for row in rows]) - observed))
        yerr = np.maximum(0.0, np.vstack((
            predicted - np.asarray([row["predicted_ci95_low"] for row in rows]),
            np.asarray([row["predicted_ci95_high"] for row in rows]) - predicted,
        )))
        parity_ax.errorbar(observed, predicted, xerr=xerr, yerr=yerr, marker=MARKERS[index], color=COLORS[index],
                           linestyle="none", markersize=4.5, capsize=1.8, markeredgecolor="black",
                           markeredgewidth=0.4, label=domain)
    parity_values = [float(row[key]) for row in prediction_rows if row["model"] == "augmented" for key in ("observed", "predicted")]
    low, high = min(parity_values), max(parity_values)
    padding = max(0.03, 0.05 * (high - low))
    low, high = low - padding, high + padding
    parity_ax.plot([low, high], [low, high], "--", color="black", linewidth=1.0, label="perfect")
    parity_ax.set_xlim(low, high)
    parity_ax.set_ylim(low, high)
    parity_ax.set_xlabel(r"Observed $g_c$")
    parity_ax.set_ylabel(r"Predicted $\hat g_c$")
    parity_ax.set_title("Augmented parity", fontsize=10)
    parity_ax.legend(frameon=False, fontsize=7)

    model_order = ["constant", "source_anchor", "size_only", "nearest_calibration", "base", "augmented"]
    model_labels = ["constant", "source / anchor", "size only", "nearest calibration", "base", "augmented*"]
    errors = [prediction_analysis["summary"][model]["mean_absolute_error"] for model in model_order]
    error_low = [
        prediction_analysis["summary"][model]["mean_absolute_error_ci95_low"]
        for model in model_order
    ]
    error_high = [
        prediction_analysis["summary"][model]["mean_absolute_error_ci95_high"]
        for model in model_order
    ]
    benchmark_xerr = np.maximum(0.0, np.vstack((
        np.asarray(errors) - np.asarray(error_low),
        np.asarray(error_high) - np.asarray(errors),
    )))
    positions = np.arange(len(model_order))
    colors = ["0.65", COLORS[4], COLORS[2], COLORS[1], COLORS[0], COLORS[3]]
    error_ax.barh(
        positions,
        errors,
        xerr=benchmark_xerr,
        color=colors,
        alpha=0.82,
        error_kw={"capsize": 2, "linewidth": 0.8},
    )
    for position, value in zip(positions, errors, strict=True):
        error_ax.text(value, position, f" {value:.3f}", va="center", fontsize=7)
    error_ax.set_yticks(positions, model_labels)
    error_ax.invert_yaxis()
    error_ax.set_xlabel(r"Mean $|\hat g_c-g_c|$")
    error_ax.set_title("Holdout benchmark", fontsize=10)
    fig.suptitle("Blinded boundary transport (*preregistered surrogate)", fontsize=12)
    save(fig, "figure7_predictive_bridge")

    # Figure 8: visibly separate diagnostic and evaluation holdout subsets.
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    domains = sorted({row["domain"] for row in prediction_rows})
    x = np.arange(len(domains), dtype=float)
    for axis, split_label, title in zip(
        axes,
        ("A", "B"),
        ("Diagnostic Holdout A", "Evaluation Holdout B"),
        strict=True,
    ):
        for model_index, model in enumerate(("base", "augmented")):
            values = []
            errors = []
            for domain in domains:
                domain_rows = [
                    row
                    for row in prediction_rows
                    if row["domain"] == domain
                    and row["model"] == model
                    and row["holdout_split"] == split_label
                ]
                mean_error = (
                    float(np.mean([row["absolute_error"] for row in domain_rows]))
                    if domain_rows
                    else np.nan
                )
                values.append(mean_error)
                if domain_rows and all("absolute_error_ci95_low" in row for row in domain_rows):
                    low = float(np.mean([row["absolute_error_ci95_low"] for row in domain_rows]))
                    high = float(np.mean([row["absolute_error_ci95_high"] for row in domain_rows]))
                    errors.append((max(0.0, mean_error - low), max(0.0, high - mean_error)))
                else:
                    errors.append((np.nan, np.nan))
            offset = (-0.10, 0.10)[model_index]
            label = "Augmented (preregistered)" if model == "augmented" else "Base"
            axis.errorbar(
                x + offset,
                values,
                yerr=np.asarray(errors, dtype=float).T,
                color=COLORS[model_index],
                marker=MARKERS[model_index],
                linestyle="none",
                markersize=5,
                capsize=2,
                label=label,
            )
        axis.set_xticks(x, [name.capitalize() for name in domains], rotation=20)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Domain")
    axes[0].set_ylabel(r"Boundary error $|\hat g_c-g_c|$")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Deterministic split; no model selection or repair on Holdout A", fontsize=11)
    save(fig, "figure8_theory_breakdown")
    return paths


def render_slurm(manifest_path: str | Path, profile_path: str | Path, output: str | Path) -> Path:
    manifest = Manifest.read(manifest_path)
    profile = tomllib.loads(Path(profile_path).read_text())["slurm"]
    block = int(profile["tasks_per_array"])
    count = math.ceil(len(manifest.tasks) / block)
    array_size = min(count, int(profile.get("max_array_size", count)))
    if block < 1 or array_size < 1:
        raise ValueError("tasks_per_array, max_array_size, and manifest size must be positive")
    partition = str(profile["partition"])
    if not partition.startswith("spark_"):
        raise ValueError("predictive production jobs require a DGX Spark partition")
    gpus = int(profile.get("gpus", 1))
    device = str(profile.get("device", "cuda" if gpus else "cpu"))
    lines = [
        "#!/usr/bin/env bash", f"#SBATCH --job-name=predictive-phase", f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{array_size - 1}%{int(profile['max_parallel'])}", f"#SBATCH --time={profile['time']}",
        f"#SBATCH --gres=gpu:{gpus}", f"#SBATCH --cpus-per-task={int(profile.get('cpus', 4))}",
        f"#SBATCH --mem={profile.get('memory', '16G')}", "set -euo pipefail", ': "${STATPHYS_REPO:?}"',
        ': "${STATPHYS_MANIFEST:?}"', ': "${STATPHYS_OUTPUT:?}"', ': "${STATPHYS_PYTHON:?}"',
        f"BLOCK={block}", f"TOTAL_BLOCKS={count}", f"BLOCKS_PER_ARRAY={array_size}",
        'BLOCK_INDEX=$SLURM_ARRAY_TASK_ID',
        'cd "$STATPHYS_REPO"', 'export PYTHONPATH="$STATPHYS_REPO/src${PYTHONPATH:+:$PYTHONPATH}"',
        'while (( BLOCK_INDEX < TOTAL_BLOCKS )); do',
        '  START=$((BLOCK_INDEX * BLOCK))', '  STOP=$((START + BLOCK))',
        f'  "$STATPHYS_PYTHON" -m statphys.predictive.cli run --manifest "$STATPHYS_MANIFEST" --output "$STATPHYS_OUTPUT" --start "$START" --stop "$STOP" --device {device}',
        '  BLOCK_INDEX=$((BLOCK_INDEX + BLOCKS_PER_ARRAY))', 'done',
    ]
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text("\n".join(lines) + "\n"); return target
