"""Tidy, hierarchy-preserving aggregation of phase-atlas artifacts.

Every registered run remains visible, including never-started and failed runs.
Scientific eligibility is deliberately stricter than the artifact store's
terminal status: provenance, summary, trajectories, diagnostics, and recorded
checksums must all validate before a run contributes to ensemble claims.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    binder_crossing,
    binder_summary,
    classify_phase,
    classify_transition_evidence,
    finite_difference_response,
    fit_finite_size_scaling_grid,
    susceptibility,
)
from .artifacts import sha256_file
from .schema import RunSpec, run_spec_from_dict


_BASE_TRAJECTORIES = {
    "data_loss",
    "objective",
    "gradient_norm",
    "weight_norm",
    "relative_update_norm",
}
_REQUIRED_FILES = (
    "spec.json",
    "provenance.json",
    "status.json",
    "summary.json",
    "trajectories.npz",
    "diagnostics.npz",
)
_CHECKSUM_FILES = ("trajectories.npz", "diagnostics.npz")
_CLAIM_GROUP_COLUMNS = (
    "spec.experiment",
    "spec.phase.architecture",
    "spec.phase.data",
    "spec.initialization",
    "spec.phase.positional_strength",
    "spec.phase.semantic_strength",
    "spec.phase.input_noise",
    "spec.phase.regularization",
    "spec.phase.temperature",
    "spec.phase.scaling.sequence_length",
    "spec.phase.scaling.n_heads",
    "spec.phase.scaling.n_layers",
    "spec.phase.scaling.teacher_rank",
    "spec.training.optimizer",
)


def _json_value(value: Any) -> Any:
    """Recursively convert NumPy/path values to strict JSON values."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _flatten(mapping: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, name))
        else:
            result[name] = _json_value(value)
    return result


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: expected a JSON object")
        return None
    return value


def _coerce_spec(value: RunSpec | Mapping[str, Any]) -> RunSpec:
    return value if isinstance(value, RunSpec) else run_spec_from_dict(value)


def _read_expected_manifest(path: str | Path) -> list[RunSpec]:
    result: list[RunSpec] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON on line {line_number}") from exc
        if record.get("kind") != "run":
            continue
        spec = run_spec_from_dict(record["spec"])
        expected_id = record.get("run_id")
        if expected_id is not None and expected_id != spec.run_id:
            raise ValueError(f"manifest content hash mismatch on line {line_number}")
        result.append(spec)
    return result


def _artifact_events(
    root: Path,
) -> tuple[set[str], dict[tuple[str, str], str], list[str]]:
    """Index run IDs and the latest registered checksum for each artifact."""

    path = root / "manifest.jsonl"
    if not path.is_file():
        return set(), {}, ["manifest.jsonl is missing; checksums cannot be audited"]
    run_ids: set[str] = set()
    checksums: dict[tuple[str, str], str] = {}
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"manifest.jsonl line {line_number} is invalid JSON")
            continue
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        run_ids.add(run_id)
        checksum = record.get("sha256")
        artifact_path = record.get("path")
        if record.get("event") == "artifact" and isinstance(checksum, str) and isinstance(artifact_path, str):
            checksums[(run_id, Path(artifact_path).name)] = checksum.lower()
    return run_ids, checksums, warnings


def _hierarchy(spec: RunSpec) -> dict[str, Any]:
    seeds = spec.seeds.resolved()
    return {
        "replica": int(spec.replica),
        "teacher_seed": int(seeds["teacher"]),
        "data_seed": int(seeds["data"]),
        "initialization_seed": int(seeds["initialization"]),
        "optimizer_seed": int(seeds["minibatch"]),
        "dropout_seed": int(seeds["dropout"]),
        "teacher_key": f"teacher:{seeds['teacher']}",
        "data_key": f"teacher:{seeds['teacher']}/data:{seeds['data']}",
        "optimizer_key": (
            f"teacher:{seeds['teacher']}/data:{seeds['data']}"
            f"/optimizer:{seeds['minibatch']}/init:{seeds['initialization']}"
        ),
    }


def _step_axis(
    arrays: Mapping[str, np.ndarray], name: str, length: int
) -> tuple[str | None, np.ndarray | None]:
    step = arrays.get("step")
    probe = arrays.get("probe_step")
    if name in {"step", "probe_step"}:
        values = arrays[name]
        return name, values if values.ndim == 1 and values.size == length else None
    step_matches = step is not None and step.ndim == 1 and step.size == length
    probe_matches = probe is not None and probe.ndim == 1 and probe.size == length
    if probe_matches and (not step_matches or name not in _BASE_TRAJECTORIES):
        return "probe_step", probe
    if step_matches:
        return "step", step
    return None, None


def _load_npz(path: Path, errors: list[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return {}


def _trajectory_rows(
    run_id: str,
    run_fields: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hierarchy = {
        name: run_fields.get(name)
        for name in (
            "replica",
            "teacher_seed",
            "data_seed",
            "initialization_seed",
            "optimizer_seed",
            "dropout_seed",
            "teacher_key",
            "data_key",
            "optimizer_key",
        )
    }
    for name in sorted(arrays):
        array = arrays[name]
        shape = list(array.shape)
        first_axis = int(array.shape[0]) if array.ndim else 1
        step_name, step_values = _step_axis(arrays, name, first_axis)
        coordinates = np.ndindex(array.shape) if array.ndim else [()]
        for coordinate in coordinates:
            index = int(coordinate[0]) if coordinate else 0
            value = array[coordinate] if coordinate else array.item()
            rows.append(
                {
                    "run_id": run_id,
                    **hierarchy,
                    "array_name": name,
                    "array_dtype": str(array.dtype),
                    "array_shape": shape,
                    "trajectory_index": index,
                    "coordinate": list(coordinate),
                    "step_axis": step_name,
                    "step": _json_value(step_values[index]) if step_values is not None else None,
                    "value": _json_value(value),
                }
            )
    return rows


def _diagnostic_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    fields: dict[str, Any] = {"diagnostics.array_names": sorted(arrays)}
    for name, array in arrays.items():
        fields[f"diagnostics.{name}.shape"] = list(array.shape)
        fields[f"diagnostics.{name}.dtype"] = str(array.dtype)
    spectra = arrays.get("qk_top_singular_values")
    if spectra is not None and spectra.ndim == 2 and spectra.size:
        finite_count = np.sum(np.isfinite(spectra), axis=0)
        total = np.nansum(spectra, axis=0)
        mean = np.divide(total, finite_count, out=np.full(spectra.shape[1], np.nan), where=finite_count > 0)
        maximum = np.max(np.where(np.isfinite(spectra), spectra, -np.inf), axis=0)
        maximum[~np.isfinite(maximum)] = np.nan
        fields["diagnostics.qk_singular_values_mean"] = _json_value(mean)
        fields["diagnostics.qk_singular_values_max"] = _json_value(maximum)
    overlap = arrays.get("head_latent_overlap")
    if overlap is not None and overlap.ndim == 2:
        fields["diagnostics.head_latent_overlap_frobenius"] = float(np.linalg.norm(overlap))
    return fields


def _number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return float(value)
    return None


def _summary_order(row: Mapping[str, Any], component: str) -> float | None:
    return _number(
        row.get(
            f"summary.functional_m_{component}",
            row.get(f"summary.m_{component}"),
        )
    )


def _state_signature(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    ignored = {"spec.replica", "spec.notes", "spec.tags"}
    return tuple(
        sorted(
            (key, json.dumps(_json_value(value), sort_keys=True))
            for key, value in row.items()
            if key.startswith("spec.") and not key.startswith("spec.seeds.") and key not in ignored
        )
    )


def _teacher_samples(
    rows: Sequence[Mapping[str, Any]], extractor: Any
) -> np.ndarray:
    hierarchy: dict[Any, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = extractor(row)
        if value is None or not math.isfinite(float(value)):
            continue
        teacher = row.get("teacher_seed", row.get("run_id"))
        data = row.get("data_seed", row.get("run_id"))
        hierarchy[teacher][data].append(float(value))
    result: list[float] = []
    for datasets in hierarchy.values():
        data_means = [float(np.mean(values)) for values in datasets.values() if values]
        if data_means:
            result.append(float(np.mean(data_means)))
    return np.asarray(result)


def build_ensemble_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate optimizer→data→teacher at each registered state point."""

    grouped: dict[tuple[tuple[str, str], ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(key.startswith("spec.") for key in row):
            grouped[_state_signature(row)].append(row)
    ensembles: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        eligible = [row for row in group if row.get("eligible_for_claims")]
        first = group[0]
        state_fields = {
            key: value
            for key, value in first.items()
            if key.startswith("spec.") and not key.startswith("spec.seeds.")
        }
        teacher_ids = {row.get("teacher_seed") for row in eligible}
        data_ids = {row.get("data_key") for row in eligible}
        optimizer_ids = {row.get("optimizer_key") for row in eligible}
        ensemble: dict[str, Any] = {
            **state_fields,
            "n_expected_runs": len(group),
            "n_eligible_runs": len(eligible),
            "n_teachers": len(teacher_ids),
            "n_datasets": len(data_ids),
            "n_optimizers": len(optimizer_ids),
            "complete": len(eligible) == len(group) and bool(eligible),
            "missing_run_ids": [row.get("run_id") for row in group if row not in eligible],
            "fluctuation_unit": "teacher_mean_after_optimizer_and_data_averaging",
        }
        size = _number(first.get("spec.phase.scaling.d_model"))
        n_eff_values = [
            _number(row.get("summary.n_eff")) for row in eligible if _number(row.get("summary.n_eff"))
        ]
        n_eff = float(np.median(n_eff_values)) if n_eff_values else size
        ensemble["n_eff"] = n_eff
        ensemble["n_eff_definition"] = "summary.n_eff" if n_eff_values else "d_model"
        extractors = {
            "m_pos": lambda row: _summary_order(row, "pos"),
            "m_sem": lambda row: _summary_order(row, "sem"),
            "competition": lambda row: (
                _summary_order(row, "sem") - _summary_order(row, "pos")
                if _summary_order(row, "sem") is not None
                and _summary_order(row, "pos") is not None
                else None
            ),
        }
        for name, extractor in extractors.items():
            samples = _teacher_samples(eligible, extractor)
            ensemble[f"mean_{name}"] = float(samples.mean()) if samples.size else None
            ensemble[f"std_{name}"] = float(samples.std(ddof=1)) if samples.size > 1 else None
            ensemble[f"n_teacher_{name}"] = int(samples.size)
            if samples.size > 1 and n_eff is not None and n_eff > 0:
                ensemble[f"susceptibility_{name}"] = float(susceptibility(samples, n_eff=n_eff))
                binders = binder_summary(samples)
                ensemble[f"binder_{name}_raw"] = _json_value(binders["raw"])
                ensemble[f"binder_{name}_centered"] = _json_value(binders["centered"])
            else:
                ensemble[f"susceptibility_{name}"] = None
                ensemble[f"binder_{name}_raw"] = None
                ensemble[f"binder_{name}_centered"] = None
        mean_pos, mean_sem = ensemble["mean_m_pos"], ensemble["mean_m_sem"]
        if mean_pos is not None and mean_sem is not None:
            ensemble["phase_label"] = classify_phase(mean_pos, mean_sem)["label"]
        else:
            ensemble["phase_label"] = "unresolved"
        ensemble["eligible_for_transition"] = bool(ensemble["complete"] and ensemble["n_teachers"] >= 2)
        ensembles.append(ensemble)
    return ensembles


def _claim_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in _CLAIM_GROUP_COLUMNS)


def _power_growth(sizes: Sequence[float], peaks: Sequence[float]) -> float | None:
    valid = [(size, peak) for size, peak in zip(sizes, peaks) if size > 0 and peak > 0]
    if len(valid) < 3:
        return None
    x, y = np.log(np.asarray(valid, dtype=float)).T
    return float(np.polyfit(x, y, 1)[0])


def _rectangular_metric(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    controls = sorted({_number(row.get("spec.phase.semantic_mixture")) for row in rows} - {None})
    sizes = sorted({_number(row.get("spec.phase.scaling.d_model")) for row in rows} - {None})
    if len(controls) < 2 or len(sizes) < 2:
        return None
    lookup = {
        (_number(row.get("spec.phase.scaling.d_model")), _number(row.get("spec.phase.semantic_mixture"))): _number(row.get(metric))
        for row in rows
    }
    matrix = np.full((len(sizes), len(controls)), np.nan)
    for i, size in enumerate(sizes):
        for j, control in enumerate(controls):
            value = lookup.get((size, control))
            if value is None:
                return None
            matrix[i, j] = value
    return np.asarray(controls), np.asarray(sizes), matrix


def evaluate_claims(
    rows: Sequence[Mapping[str, Any]],
    ensembles: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build conservative transition claims from hierarchy-aware ensembles."""

    ensemble_rows = list(ensembles) if ensembles is not None else build_ensemble_table(rows)
    run_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    ensemble_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        run_groups[_claim_key(row)].append(row)
    for row in ensemble_rows:
        ensemble_groups[_claim_key(row)].append(row)
    claims: list[dict[str, Any]] = []
    for key in sorted(set(run_groups) | set(ensemble_groups), key=str):
        run_group = run_groups.get(key, [])
        group = ensemble_groups.get(key, [])
        eligible = [row for row in group if row.get("eligible_for_transition")]
        sizes = sorted({_number(row.get("spec.phase.scaling.d_model")) for row in eligible} - {None})
        susceptibility_peaks: list[float] = []
        response_peaks: list[float] = []
        peak_sizes: list[float] = []
        for size in sizes:
            size_rows = [row for row in eligible if _number(row.get("spec.phase.scaling.d_model")) == size]
            susceptibilities = [_number(row.get("susceptibility_competition")) for row in size_rows]
            finite_susceptibilities = [value for value in susceptibilities if value is not None]
            grid = _rectangular_metric(size_rows, "mean_competition")
            if finite_susceptibilities and len(size_rows) >= 2:
                susceptibility_peaks.append(max(finite_susceptibilities))
                peak_sizes.append(float(size))
                controls = np.asarray(
                    sorted(_number(row.get("spec.phase.semantic_mixture")) for row in size_rows)
                )
                values_by_control = {
                    _number(row.get("spec.phase.semantic_mixture")): _number(row.get("mean_competition"))
                    for row in size_rows
                }
                values = np.asarray([values_by_control[control] for control in controls], dtype=float)
                response = finite_difference_response(controls, values)
                response_peaks.append(float(np.max(response["absolute_response"])))
            del grid
        susceptibility_growth = _power_growth(peak_sizes, susceptibility_peaks)
        response_growth = _power_growth(peak_sizes, response_peaks)

        binder_spread: float | None = None
        binder_grid = _rectangular_metric(eligible, "binder_competition_raw")
        if binder_grid is not None:
            controls, binder_sizes, binder_values = binder_grid
            crossing = binder_crossing(controls, binder_sizes, binder_values)
            binder_spread = _number(crossing["crossing_spread"])

        collapse_score: float | None = None
        collapse_fit: dict[str, Any] | None = None
        order_grid = _rectangular_metric(eligible, "mean_competition")
        if order_grid is not None and len(sizes) >= 5:
            controls, collapse_sizes, orders = order_grid
            critical_grid = np.unique(
                np.concatenate((controls, 0.5 * (controls[:-1] + controls[1:])))
            )
            fit = fit_finite_size_scaling_grid(
                controls,
                collapse_sizes,
                orders,
                critical_control_grid=critical_grid,
                observable_exponent_grid=np.linspace(0.0, 1.0, 9),
                inverse_nu_grid=np.linspace(0.25, 2.0, 8),
                n_grid=48,
            )
            collapse_score = float(fit["score"])
            collapse_fit = {
                name: _json_value(fit[name])
                for name in ("critical_control", "observable_exponent", "inverse_nu", "nu", "score")
            }
        transition = classify_transition_evidence(
            n_sizes=max(1, len(sizes)),
            susceptibility_peak_growth=susceptibility_growth,
            binder_crossing_spread=binder_spread,
            data_collapse_score=collapse_score,
            response_peak_growth=response_growth,
        )
        complete = bool(run_group) and all(row.get("eligible_for_claims") for row in run_group)
        enough_teachers = bool(group) and all(row.get("n_teachers", 0) >= 2 for row in group)
        label = transition["label"] if complete and enough_teachers else "insufficient_evidence"
        identifying = dict(zip(_CLAIM_GROUP_COLUMNS, key))
        claims.append(
            {
                **identifying,
                "label": label,
                "analysis_label": transition["label"],
                "n_expected_runs": len(run_group),
                "n_eligible_runs": sum(bool(row.get("eligible_for_claims")) for row in run_group),
                "n_state_points": len(group),
                "n_sizes": len(sizes),
                "complete_artifacts": complete,
                "enough_teachers": enough_teachers,
                "susceptibility_peak_growth": susceptibility_growth,
                "response_peak_growth": response_growth,
                "binder_crossing_spread": binder_spread,
                "data_collapse_score": collapse_score,
                "data_collapse_fit": collapse_fit,
                "evidence": transition,
            }
        )
    return claims


@dataclass(frozen=True)
class AtlasAggregate:
    """In-memory tidy tables plus audit metadata and conservative claims."""

    runs: list[dict[str, Any]]
    trajectories: list[dict[str, Any]]
    ensembles: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    metadata: dict[str, Any]

    @property
    def completed_runs(self) -> list[dict[str, Any]]:
        return [row for row in self.runs if row.get("is_completed")]

    @property
    def eligible_runs(self) -> list[dict[str, Any]]:
        return [row for row in self.runs if row.get("eligible_for_claims")]

    @property
    def missing_runs(self) -> list[dict[str, Any]]:
        return [row for row in self.runs if row.get("missing_components")]


def aggregate_artifacts(
    artifact_root: str | Path,
    *,
    expected_specs: Iterable[RunSpec | Mapping[str, Any]] | None = None,
    expected_manifest: str | Path | None = None,
    manifest: str | Path | None = None,
    destination: str | Path | None = None,
    require_trajectories: bool = True,
    include_incomplete_trajectories: bool = False,
) -> AtlasAggregate | dict[str, Any]:
    """Read, audit, aggregate, and optionally write one registered artifact tree."""

    if manifest is not None:
        if expected_manifest is not None:
            raise ValueError("provide expected_manifest or manifest, not both")
        expected_manifest = manifest
    if expected_specs is not None and expected_manifest is not None:
        raise ValueError("provide expected_specs or expected_manifest, not both")
    root = Path(artifact_root)
    runs_root = root / "runs"
    registered: dict[str, RunSpec] = {}
    if expected_specs is not None:
        for value in expected_specs:
            spec = _coerce_spec(value)
            registered[spec.run_id] = spec
    elif expected_manifest is not None:
        for spec in _read_expected_manifest(expected_manifest):
            registered[spec.run_id] = spec

    event_run_ids, checksums, manifest_warnings = _artifact_events(root)
    discovered: set[str] = set(registered) | event_run_ids
    if runs_root.is_dir():
        discovered.update(path.name for path in runs_root.iterdir() if path.is_dir())
    run_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for run_id in sorted(discovered):
        directory = runs_root / run_id
        errors: list[str] = []
        raw_spec = _read_json(directory / "spec.json", errors)
        provenance = _read_json(directory / "provenance.json", errors)
        status = _read_json(directory / "status.json", errors)
        summary = _read_json(directory / "summary.json", errors)
        spec = registered.get(run_id)
        if raw_spec is not None:
            try:
                stored_spec = run_spec_from_dict(raw_spec)
                if stored_spec.run_id != run_id:
                    errors.append("spec.json: content hash does not match run directory")
                elif spec is not None and stored_spec.canonical_json() != spec.canonical_json():
                    errors.append("spec.json: differs from registered expected specification")
                else:
                    spec = stored_spec
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(f"spec.json: {type(exc).__name__}: {exc}")

        missing = [name for name in _REQUIRED_FILES if not (directory / name).is_file()]
        if not require_trajectories and "trajectories.npz" in missing:
            missing.remove("trajectories.npz")
        checksum_targets = list(_CHECKSUM_FILES)
        if not require_trajectories:
            checksum_targets.remove("trajectories.npz")
        for filename in checksum_targets:
            path = directory / filename
            if not path.is_file():
                continue
            expected_checksum = checksums.get((run_id, filename))
            if expected_checksum is None:
                missing.append(f"checksum:{filename}")
            else:
                actual_checksum = sha256_file(path)
                if actual_checksum.lower() != expected_checksum:
                    errors.append(
                        f"{filename}: checksum mismatch {actual_checksum} != {expected_checksum}"
                    )
                    missing.append(f"checksum:{filename}")

        state = status.get("state") if status is not None else "missing"
        is_completed = state == "completed"
        if not is_completed:
            missing.append("completed_status")
        if spec is None:
            missing.append("valid_spec")
        if status is not None and status.get("run_id") not in {None, run_id}:
            errors.append("status.json: run_id does not match directory")
        if summary is not None and summary.get("run_id") not in {None, run_id}:
            errors.append("summary.json: run_id does not match directory")

        row: dict[str, Any] = {
            "run_id": run_id,
            "run_directory": str(directory),
            "artifact_present": directory.is_dir(),
            "status_state": state,
            "is_completed": is_completed,
        }
        if spec is not None:
            row.update(_hierarchy(spec))
            row.update(_flatten(spec.to_dict(), "spec"))
        if provenance is not None:
            row.update(_flatten(provenance, "provenance"))
        if status is not None:
            row.update(_flatten(status, "status"))
        if summary is not None:
            row.update(_flatten(summary, "summary"))
            m_pos, m_sem = _summary_order(row, "pos"), _summary_order(row, "sem")
            if m_pos is not None and m_sem is not None:
                phase = classify_phase(m_pos, m_sem)
                row["phase_label"] = phase["label"]
                row["phase_classification"] = phase
            else:
                row["phase_label"] = str(row.get("summary.phase_label") or "unresolved")
        else:
            row["phase_label"] = "unresolved"

        if is_completed or include_incomplete_trajectories:
            trajectory_arrays = _load_npz(directory / "trajectories.npz", errors)
            trajectories.extend(_trajectory_rows(run_id, row, trajectory_arrays))
            diagnostics = _load_npz(directory / "diagnostics.npz", errors)
            row.update(_diagnostic_fields(diagnostics))
        if errors:
            missing.append("read_errors")
        row["read_errors"] = errors
        row["missing_components"] = sorted(set(missing))
        row["eligible_for_claims"] = is_completed and not row["missing_components"]
        run_rows.append(row)

    ensembles = build_ensemble_table(run_rows)
    claims = evaluate_claims(run_rows, ensembles)
    metadata = {
        "artifact_root": str(root),
        "n_expected": len(registered) if registered else None,
        "n_runs": len(run_rows),
        "n_completed": sum(bool(row["is_completed"]) for row in run_rows),
        "n_eligible": sum(bool(row["eligible_for_claims"]) for row in run_rows),
        "n_missing_or_incomplete": sum(bool(row["missing_components"]) for row in run_rows),
        "n_trajectory_rows": len(trajectories),
        "n_ensemble_rows": len(ensembles),
        "require_trajectories": require_trajectories,
        "claim_requirements": list(_REQUIRED_FILES) + [f"sha256:{name}" for name in _CHECKSUM_FILES],
        "manifest_warnings": manifest_warnings,
    }
    aggregate = AtlasAggregate(run_rows, trajectories, ensembles, claims, metadata)
    if destination is None:
        return aggregate
    requested = Path(destination)
    output_dir = requested.parent if requested.suffix else requested
    prefix = requested.stem if requested.suffix else "atlas"
    paths = write_tidy_aggregate(aggregate, output_dir, prefix=prefix)
    bundle_path = (
        requested
        if requested.suffix.lower() == ".json"
        else output_dir / f"{prefix}_aggregate.json"
    )
    bundle = {
        "runs": aggregate.runs,
        "trajectories": aggregate.trajectories,
        "ensembles": aggregate.ensembles,
        "claims": aggregate.claims,
        "metadata": aggregate.metadata,
    }
    _atomic_text(bundle_path, json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n")
    if requested.suffix.lower() == ".csv":
        _atomic_text(requested, _csv_text(aggregate.runs))
    return {
        "aggregate": str(bundle_path),
        "files": {name: str(path) for name, path in paths.items()},
        "metadata": _json_value(metadata),
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields = sorted({str(key) for row in rows for key in row})
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    json.dumps(_json_value(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (Mapping, list, tuple, np.ndarray))
                    else _json_value(row.get(field))
                )
                for field in fields
            }
        )
    return buffer.getvalue()


def write_tidy_aggregate(
    aggregate: AtlasAggregate,
    output_dir: str | Path,
    *,
    prefix: str = "atlas",
) -> dict[str, Path]:
    """Atomically write run, trajectory, ensemble, claim, and audit tables."""

    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a simple filename component")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_payloads = {
        "runs": aggregate.runs,
        "trajectories": aggregate.trajectories,
        "ensembles": aggregate.ensembles,
    }
    payloads: dict[str, tuple[Path, str]] = {}
    for name, rows in table_payloads.items():
        payloads[f"{name}_csv"] = (destination / f"{prefix}_{name}.csv", _csv_text(rows))
        payloads[f"{name}_json"] = (
            destination / f"{prefix}_{name}.json",
            json.dumps(_json_value(rows), indent=2, sort_keys=True) + "\n",
        )
    payloads["claims_json"] = (
        destination / f"{prefix}_claims.json",
        json.dumps(_json_value(aggregate.claims), indent=2, sort_keys=True) + "\n",
    )
    payloads["metadata_json"] = (
        destination / f"{prefix}_metadata.json",
        json.dumps(_json_value(aggregate.metadata), indent=2, sort_keys=True) + "\n",
    )
    for path, text in payloads.values():
        _atomic_text(path, text)
    return {name: path for name, (path, _) in payloads.items()}


__all__ = [
    "AtlasAggregate",
    "aggregate_artifacts",
    "build_ensemble_table",
    "evaluate_claims",
    "write_tidy_aggregate",
]

