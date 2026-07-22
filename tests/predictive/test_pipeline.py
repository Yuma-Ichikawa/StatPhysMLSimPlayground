from __future__ import annotations

import json

import pytest

from statphys.predictive.pipeline import (
    _compare_transition_models,
    _predict_boundaries,
    _split_holdout_rows,
    build_adaptive_manifest,
    build_manifest,
    render_slurm,
)
from statphys.predictive.style import COLORS, FIGSIZE, LINE_STYLES, MARKERS, apply_style
from statphys.predictive.simulators import run_task


def test_manifest_has_nested_design_and_holdout(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('''[study]\nname="test"\nseeds=[11,13]\ninner_replicates=3\n[domains.transformer]\nvariants=["anchor","holdout"]\nholdout_variants=["holdout"]\nsizes=[8]\ncontrols=[0.9,1.1]\nsecondary_controls=[0.0]\nparameters={chains=16,steps=4,dt=0.02}\n''')
    manifest = build_manifest(config)
    assert len(manifest.tasks) == 8
    assert {task.inner_replicates for task in manifest.tasks} == {3}
    assert sum(task.holdout for task in manifest.tasks) == 4
    result = run_task(manifest.tasks[0], device="cpu")
    assert len(result["replicates"]) == 3
    assert len({row["inner_seed"] for row in result["replicates"]}) == 3


def test_slurm_rejects_non_spark_and_renders_portably(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('''[study]\nname="test"\nseeds=[11]\ninner_replicates=2\n[domains.transformer]\nvariants=["anchor"]\nholdout_variants=[]\nsizes=[8]\ncontrols=[1.0]\nsecondary_controls=[0.0]\n''')
    manifest_path = build_manifest(config).write(tmp_path / "manifest.json")
    profile = tmp_path / "profile.toml"
    profile.write_text('''[slurm]\npartition="spark_3H"\ntime="03:00:00"\ngpus=1\ncpus=2\nmemory="8G"\ntasks_per_array=4\nmax_parallel=2\n''')
    script = render_slurm(manifest_path, profile, tmp_path / "job.sbatch").read_text()
    assert "#SBATCH --partition=spark_3H" in script
    assert "/mnt/" not in script
    assert 'cd "$STATPHYS_REPO"' in script
    assert '"$STATPHYS_PYTHON" -m statphys.predictive.cli' in script

    profile.write_text('''[slurm]\npartition="gpu_shared"\ntime="03:00:00"\ngpus=1\ntasks_per_array=4\nmax_parallel=2\n''')
    with pytest.raises(ValueError, match="DGX Spark partition"):
        render_slurm(manifest_path, profile, tmp_path / "non_spark.sbatch")


def test_journal_style_contract():
    import matplotlib as mpl

    apply_style()
    assert FIGSIZE == (6.4, 4.8)
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["font.size"] == 12
    assert mpl.rcParams["xtick.direction"] == "in"
    assert mpl.rcParams["ytick.direction"] == "in"
    assert LINE_STYLES == ["-", "--", "-.", ":"]
    assert MARKERS[:4] == ["o", "s", "^", "D"]
    assert COLORS[:4] == ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def test_adaptive_manifest_adds_only_boundary_window(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('''[study]\nname="adaptive"\nseeds=[11]\nadaptive_seeds=[17,19]\ninner_replicates=2\n[domains.transformer]\nvariants=["anchor"]\nholdout_variants=[]\nsizes=[8]\ncontrols=[0.8,1.0,1.2,1.4]\nsecondary_controls=[0.0]\n''')
    base_path = build_manifest(config).write(tmp_path / "base.json")
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({"boundaries": [{"domain": "transformer", "variant": "anchor", "size": 8, "secondary": 0.0, "observed": 1.2, "holdout": False}]}))
    adaptive = build_adaptive_manifest(base_path, aggregate, config)
    assert len(adaptive.tasks) == 10
    added = [task for task in adaptive.tasks if task.seed in {17, 19}]
    assert {task.control for task in added} == {1.0, 1.2, 1.4}
    assert {task.stage for task in added} == {"adaptive_confirmation"}


def test_predictive_benchmarks_include_nontrivial_baselines():
    def boundary(variant, size, secondary, observed, holdout=False):
        return {
            "domain": "transformer",
            "variant": variant,
            "size": size,
            "secondary": secondary,
            "observed": observed,
            "observed_ci95_low": observed - 0.1,
            "observed_ci95_high": observed + 0.1,
            "ci95_width": 0.2,
            "holdout": holdout,
        }

    boundaries = [
        boundary("anchor", 16, 0.0, 1.0),
        boundary("single", 16, 1.0, 2.0),
        boundary("augmented", 64, 0.5, 3.0),
        boundary("holdout", 16, 1.0, 2.2, holdout=True),
        boundary("holdout", 64, 0.0, 2.8, holdout=True),
    ]
    result = _predict_boundaries(boundaries)
    models = {row["model"] for row in result["records"]}
    assert models == {
        "constant",
        "source_anchor",
        "size_only",
        "nearest_calibration",
        "base",
        "augmented",
    }
    first_holdout = [
        row
        for row in result["records"]
        if row["size"] == 16 and row["secondary"] == 1.0
    ]
    predictions = {row["model"]: row["predicted"] for row in first_holdout}
    assert predictions["constant"] == pytest.approx(2.0)
    assert predictions["source_anchor"] == pytest.approx(1.0)
    assert predictions["nearest_calibration"] == pytest.approx(2.0)
    assert result["summary"]["augmented"]["holdout_b_mean_absolute_error"] is not None
    assert result["summary"]["augmented"]["mean_absolute_error_ci95_high"] > 0.0
    assert all("predicted_ci95_low" in row for row in first_holdout)
    assert {row["selection_status"] for row in first_holdout if row["model"] == "augmented"} == {"preregistered"}


def test_holdout_split_is_stable_disjoint_and_model_consistent():
    rows = [
        {
            "domain": domain,
            "variant": "holdout",
            "size": size,
            "secondary": secondary,
            "model": model,
        }
        for domain in ("diffusion", "transformer")
        for size, secondary in ((16, 0.0), (32, 0.5), (64, 1.0), (128, 1.5))
        for model in ("base", "augmented")
    ]

    def identities(split_rows):
        return {(row["domain"], row["size"], row["secondary"]) for row in split_rows}

    split = _split_holdout_rows(rows)
    reversed_split = _split_holdout_rows(list(reversed(rows)))
    assert identities(split["A"]) == identities(reversed_split["A"])
    assert identities(split["B"]) == identities(reversed_split["B"])
    assert identities(split["A"]).isdisjoint(identities(split["B"]))
    assert identities(split["A"]) | identities(split["B"]) == identities(rows)
    for identity in identities(rows):
        labels = {
            label
            for label, split_rows in split.items()
            if any((row["domain"], row["size"], row["secondary"]) == identity for row in split_rows)
        }
        assert len(labels) == 1


def test_transition_model_comparison_ignores_intervention_only_rows():
    conditions = [{
        "domain": "transformer",
        "variant": "holdout",
        "secondary": 0.0,
        "size": 16,
        "control": 1.0,
        "metrics": {"intervention_quality": {"mean": 0.5}},
    }]
    assert _compare_transition_models(conditions) == []
