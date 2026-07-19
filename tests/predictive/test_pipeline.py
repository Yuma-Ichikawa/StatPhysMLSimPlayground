from __future__ import annotations

import json

from statphys.predictive.pipeline import build_adaptive_manifest, build_manifest, render_slurm
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
