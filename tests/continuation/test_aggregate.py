from pathlib import Path

import pytest

from statphys.continuation.aggregate import T95_DF4, aggregate_manifest
from statphys.continuation.schema import Domain, Manifest, TaskSpec


def _manifest() -> Manifest:
    tasks = tuple(
        TaskSpec(
            study="test",
            domain=Domain.RL,
            variant="oracle",
            stage="confirmatory",
            control_name="pressure",
            control=1.0,
            size=8,
            seed=seed,
        )
        for seed in (11, 13, 17, 19, 23)
    )
    return Manifest(
        study="test",
        seeds=(11, 13, 17, 19, 23),
        tasks=tasks,
        config_hash="test",
    )


def test_aggregate_requires_all_five_runs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate_manifest(_manifest(), tmp_path, tmp_path / "aggregate")


def test_student_t_constant_is_df_four() -> None:
    assert T95_DF4 == pytest.approx(2.7764451051977987)


def test_aggregate_records_family(tmp_path: Path) -> None:
    from statphys.continuation.core.artifacts import RunStore

    manifest = _manifest()
    store = RunStore(tmp_path)
    common = {
        "order_parameter": 0.5,
        "susceptibility": 1.0,
        "binder_cumulant": 0.2,
        "generalization_error": 0.1,
        "ood_generalization_error": 0.2,
        "effective_multiplicity": 2.0,
        "interaction_range": 0.5,
        "macrostate_entropy": 0.6,
        "oracle_gap": 0.1,
        "intervention_response": 0.2,
    }
    for task in manifest.tasks:
        store.begin(task)
        store.complete(task, common, {"signed_order_samples": [0.5]}, elapsed_seconds=0.1, device="cpu")
    aggregate = aggregate_manifest(manifest, tmp_path, tmp_path / "aggregate")
    assert aggregate["records"][0]["family"] == "anchor"
