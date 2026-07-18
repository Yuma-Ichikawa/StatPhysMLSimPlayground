"""Tests for audited, hierarchy-preserving atlas aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from statphys.atlas.aggregate import (
    aggregate_artifacts,
    build_ensemble_table,
    evaluate_claims,
    write_tidy_aggregate,
)
from statphys.atlas.schema import PhaseCard, RunSpec, ScalingPath, SeedPlan


def _spec(*, replica: int = 0, size: int = 8, mixture: float = 0.25) -> RunSpec:
    return RunSpec(
        phase=PhaseCard(scaling=ScalingPath(d_model=size), semantic_mixture=mixture),
        seeds=SeedPlan(
            root=replica,
            teacher=100 + replica,
            data=200 + replica,
            initialization=300 + replica,
            minibatch=400 + replica,
            dropout=500 + replica,
        ),
        replica=replica,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _append_checksum(root: Path, run_id: str, path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "artifact",
                    "run_id": run_id,
                    "path": str(path),
                    "sha256": digest,
                }
            )
            + "\n"
        )


def _required_support_files(root: Path, spec: RunSpec) -> Path:
    directory = root / "runs" / spec.run_id
    _write_json(directory / "spec.json", spec.to_dict())
    _write_json(directory / "provenance.json", {"git_commit": "abc", "git_dirty": False})
    np.savez_compressed(
        directory / "diagnostics.npz",
        qk_top_singular_values=np.array([[2.0, 0.5]]),
        head_latent_overlap=np.array([[1.0, 0.0]]),
    )
    _append_checksum(root, spec.run_id, directory / "diagnostics.npz")
    return directory


def _complete_run(root: Path, spec: RunSpec) -> None:
    directory = _required_support_files(root, spec)
    _write_json(directory / "status.json", {"run_id": spec.run_id, "state": "completed"})
    _write_json(
        directory / "summary.json",
        {
            "run_id": spec.run_id,
            "functional_m_pos": 0.8,
            "functional_m_sem": 0.1,
            "qk_spectral_norm_max": 1.7,
            "specialization_strength": 0.4,
        },
    )
    np.savez_compressed(
        directory / "trajectories.npz",
        step=np.array([0, 10]),
        data_loss=np.array([2.0, 1.0]),
        probe_step=np.array([0]),
        functional_m_pos=np.array([0.8]),
        qk_singular_values=np.array([[2.0, 0.5]]),
    )
    _append_checksum(root, spec.run_id, directory / "trajectories.npz")


def test_expected_failed_and_never_started_runs_remain_explicit(tmp_path: Path) -> None:
    completed, failed, never_started = _spec(replica=0), _spec(replica=1), _spec(replica=2)
    _complete_run(tmp_path, completed)
    failed_directory = tmp_path / "runs" / failed.run_id
    _write_json(failed_directory / "spec.json", failed.to_dict())
    _write_json(failed_directory / "status.json", {"state": "failed", "error": "OOM"})
    aggregate = aggregate_artifacts(
        tmp_path,
        expected_specs=[completed, failed, never_started],
    )
    assert not isinstance(aggregate, dict)
    assert aggregate.metadata["n_runs"] == 3
    assert aggregate.metadata["n_completed"] == 1
    assert aggregate.metadata["n_eligible"] == 1
    by_id = {row["run_id"]: row for row in aggregate.runs}
    completed_row = by_id[completed.run_id]
    assert completed_row["eligible_for_claims"]
    assert completed_row["phase_label"] == "positional"
    assert completed_row["summary.qk_spectral_norm_max"] == 1.7
    assert completed_row["diagnostics.qk_singular_values_mean"] == [2.0, 0.5]
    assert completed_row["teacher_seed"] == 100
    assert completed_row["optimizer_key"].endswith("optimizer:400/init:300")
    assert "completed_status" in by_id[failed.run_id]["missing_components"]
    assert "provenance.json" in by_id[failed.run_id]["missing_components"]
    assert not by_id[never_started.run_id]["artifact_present"]
    assert "status.json" in by_id[never_started.run_id]["missing_components"]
    assert aggregate.claims[0]["label"] == "insufficient_evidence"


def test_trajectory_table_preserves_axes_steps_and_hierarchy(tmp_path: Path) -> None:
    spec = _spec()
    _complete_run(tmp_path, spec)
    aggregate = aggregate_artifacts(tmp_path, expected_specs=[spec])
    assert not isinstance(aggregate, dict)
    loss_rows = [row for row in aggregate.trajectories if row["array_name"] == "data_loss"]
    assert [row["step"] for row in loss_rows] == [0, 10]
    assert all(row["step_axis"] == "step" for row in loss_rows)
    probe_rows = [
        row for row in aggregate.trajectories if row["array_name"] == "functional_m_pos"
    ]
    assert probe_rows[0]["step_axis"] == "probe_step"
    spectral_rows = [
        row for row in aggregate.trajectories if row["array_name"] == "qk_singular_values"
    ]
    assert [row["coordinate"] for row in spectral_rows] == [[0, 0], [0, 1]]
    assert [row["value"] for row in spectral_rows] == [2.0, 0.5]
    assert all(row["teacher_seed"] == 100 for row in aggregate.trajectories)


def test_tidy_outputs_include_run_trajectory_and_ensemble_tables(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    spec = _spec()
    _complete_run(root, spec)
    aggregate = aggregate_artifacts(root, expected_specs=[spec])
    assert not isinstance(aggregate, dict)
    paths = write_tidy_aggregate(aggregate, tmp_path / "tidy", prefix="paper")
    assert set(paths) == {
        "runs_csv",
        "runs_json",
        "trajectories_csv",
        "trajectories_json",
        "ensembles_csv",
        "ensembles_json",
        "claims_json",
        "metadata_json",
    }
    assert all(path.is_file() for path in paths.values())
    with paths["runs_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == spec.run_id
    assert rows[0]["missing_components"] == "[]"
    ensembles = json.loads(paths["ensembles_json"].read_text(encoding="utf-8"))
    assert ensembles[0]["mean_m_pos"] == 0.8
    assert ensembles[0]["fluctuation_unit"].startswith("teacher_mean")


def test_cli_compatible_manifest_destination_writes_bundle(tmp_path: Path) -> None:
    spec = _spec()
    _complete_run(tmp_path / "artifacts", spec)
    manifest = tmp_path / "sweep.jsonl"
    manifest.write_text(
        json.dumps({"kind": "metadata", "name": "test"})
        + "\n"
        + json.dumps({"kind": "run", "run_id": spec.run_id, "spec": spec.to_dict()})
        + "\n",
        encoding="utf-8",
    )
    report = aggregate_artifacts(
        tmp_path / "artifacts",
        manifest=manifest,
        destination=tmp_path / "tidy",
    )
    assert isinstance(report, dict)
    assert Path(report["aggregate"]).is_file()
    bundle = json.loads(Path(report["aggregate"]).read_text(encoding="utf-8"))
    assert bundle["runs"][0]["run_id"] == spec.run_id
    assert "ensembles" in bundle


def test_completed_run_can_make_trajectories_optional(tmp_path: Path) -> None:
    spec = _spec()
    directory = _required_support_files(tmp_path, spec)
    _write_json(directory / "status.json", {"state": "completed"})
    _write_json(
        directory / "summary.json",
        {"functional_m_pos": 0.1, "functional_m_sem": 0.8},
    )
    strict = aggregate_artifacts(tmp_path, expected_specs=[spec])
    relaxed = aggregate_artifacts(tmp_path, expected_specs=[spec], require_trajectories=False)
    assert not isinstance(strict, dict) and not isinstance(relaxed, dict)
    assert not strict.runs[0]["eligible_for_claims"]
    assert relaxed.runs[0]["eligible_for_claims"]
    assert relaxed.runs[0]["phase_label"] == "semantic"


def test_missing_checksum_and_corrupt_npz_are_claim_blockers(tmp_path: Path) -> None:
    spec = _spec()
    directory = _required_support_files(tmp_path, spec)
    _write_json(directory / "status.json", {"state": "completed"})
    _write_json(directory / "summary.json", {"functional_m_pos": 0.5, "functional_m_sem": 0.5})
    (directory / "trajectories.npz").write_bytes(b"not an npz file")
    _append_checksum(tmp_path, spec.run_id, directory / "trajectories.npz")
    aggregate = aggregate_artifacts(tmp_path, expected_specs=[spec])
    assert not isinstance(aggregate, dict)
    row = aggregate.runs[0]
    assert not row["eligible_for_claims"]
    assert "read_errors" in row["missing_components"]
    assert "trajectories.npz" in row["read_errors"][0]

    second = _spec(replica=1)
    _complete_run(tmp_path, second)
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    retained = [
        line
        for line in lines
        if not (
            json.loads(line).get("run_id") == second.run_id
            and Path(json.loads(line).get("path", "")).name == "diagnostics.npz"
        )
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(retained) + "\n", encoding="utf-8")
    audited = aggregate_artifacts(tmp_path, expected_specs=[second])
    assert not isinstance(audited, dict)
    second_row = next(row for row in audited.runs if row["run_id"] == second.run_id)
    assert "checksum:diagnostics.npz" in second_row["missing_components"]


def test_ensemble_table_uses_teacher_as_outer_fluctuation_unit() -> None:
    rows = []
    for teacher, value in [(1, 0.2), (2, 0.8)]:
        for data in (10, 11):
            for optimizer in (20, 21):
                rows.append(
                    {
                        "run_id": f"{teacher}-{data}-{optimizer}",
                        "eligible_for_claims": True,
                        "teacher_seed": teacher,
                        "data_seed": data,
                        "data_key": f"{teacher}-{data}",
                        "optimizer_key": f"{teacher}-{data}-{optimizer}",
                        "spec.experiment": "x",
                        "spec.phase.scaling.d_model": 10,
                        "spec.phase.semantic_mixture": 0.5,
                        "summary.functional_m_pos": 1.0 - value,
                        "summary.functional_m_sem": value,
                    }
                )
    ensemble = build_ensemble_table(rows)[0]
    assert ensemble["n_expected_runs"] == 8
    assert ensemble["n_teachers"] == 2
    assert ensemble["mean_m_sem"] == 0.5
    assert np.isclose(ensemble["susceptibility_m_sem"], 0.9)
    assert ensemble["n_eff"] == 10
    assert ensemble["n_eff_definition"] == "d_model"


def test_transition_claim_requires_complete_five_size_ensemble_evidence() -> None:
    run_rows: list[dict[str, object]] = []
    ensembles: list[dict[str, object]] = []
    for size in (8.0, 16.0, 32.0, 64.0, 128.0):
        for control in (-0.5, 0.0, 0.5):
            common = {
                "spec.experiment": "atlas",
                "spec.phase.architecture": "m3_multi_head",
                "spec.phase.data": "d0_gaussian",
                "spec.initialization": "random",
                "spec.phase.semantic_mixture": control,
                "spec.phase.scaling.d_model": size,
            }
            run_rows.append({**common, "eligible_for_claims": True})
            scaled = size**-0.25 * (1.0 + control * size**0.5)
            ensembles.append(
                {
                    **common,
                    "eligible_for_transition": True,
                    "n_teachers": 3,
                    "mean_competition": scaled,
                    "susceptibility_competition": size**0.5 * np.exp(-(control**2)),
                    "binder_competition_raw": 0.4 + control * size**0.2,
                }
            )
    claim = evaluate_claims(run_rows, ensembles)[0]
    assert claim["label"] == "continuous_transition_candidate"
    assert claim["susceptibility_peak_growth"] > 0
    assert claim["binder_crossing_spread"] == 0.0
    assert claim["data_collapse_score"] < 1e-20
    run_rows[-1]["eligible_for_claims"] = False
    assert evaluate_claims(run_rows, ensembles)[0]["label"] == "insufficient_evidence"

