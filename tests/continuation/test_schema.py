from pathlib import Path

import pytest

from statphys.continuation.schema import Domain, Manifest, REQUIRED_SEED_COUNT, TaskSpec, expand_config


def _config(seeds: str) -> str:
    return f"""
[study]
name = "test"
seeds = [{seeds}]

[[experiments]]
domain = "diffusion"
variants = ["oracle", "topk"]
stage = "confirmatory"
control_name = "noise"
controls = [0.5, 1.0]
sizes = [4, 8]
parameters = {{ n_probe = 16 }}
"""


def test_expansion_has_exactly_five_seeds_per_condition(tmp_path: Path) -> None:
    path = tmp_path / "study.toml"
    path.write_text(_config("11, 13, 17, 19, 23"))
    manifest = expand_config(path)
    assert len(manifest.seeds) == REQUIRED_SEED_COUNT
    assert len(manifest.tasks) == 2 * 2 * 2 * REQUIRED_SEED_COUNT
    for condition in {task.condition_id for task in manifest.tasks}:
        assert {
            task.seed for task in manifest.tasks if task.condition_id == condition
        } == set(manifest.seeds)


def test_expansion_rejects_non_five_seed_study(tmp_path: Path) -> None:
    path = tmp_path / "study.toml"
    path.write_text(_config("11, 13, 17, 19"))
    with pytest.raises(ValueError, match="exactly 5"):
        expand_config(path)


def test_composition_preserves_component_task_ids() -> None:
    from statphys.continuation.schema import compose_manifests

    first = TaskSpec(
        study="component_a",
        domain=Domain.RL,
        variant="oracle",
        stage="confirmatory",
        control_name="pressure",
        control=1.0,
        size=8,
        seed=11,
    )
    manifest_a = Manifest(
        study="component_a",
        seeds=(11, 13, 17, 19, 23),
        tasks=tuple(
            TaskSpec.from_dict({**first.to_dict(), "seed": seed})
            for seed in (11, 13, 17, 19, 23)
        ),
        config_hash="a",
    )
    second = TaskSpec(
        study="component_b",
        domain=Domain.DIFFUSION,
        variant="oracle",
        stage="confirmatory",
        control_name="noise",
        control=0.5,
        size=8,
        seed=11,
    )
    manifest_b = Manifest(
        study="component_b",
        seeds=(11, 13, 17, 19, 23),
        tasks=tuple(
            TaskSpec.from_dict({**second.to_dict(), "seed": seed})
            for seed in (11, 13, 17, 19, 23)
        ),
        config_hash="b",
    )
    composed = compose_manifests((manifest_a, manifest_b), "complete")
    assert {task.task_id for task in composed.tasks} == {
        task.task_id for task in (*manifest_a.tasks, *manifest_b.tasks)
    }
    assert {task.study for task in composed.tasks} == {"component_a", "component_b"}
