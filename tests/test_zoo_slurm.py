"""Tests for the architecture zoo and Slurm utilities."""

import pytest
import torch

from statphys.experiment.zoo import (
    ARCHITECTURES,
    architecture_experiment,
    build_architecture,
)
from statphys.utils.slurm import SlurmConfig, SlurmLauncher, render_sbatch, submit_array


class TestZoo:
    """Architecture zoo construction and shape contracts."""

    @pytest.mark.parametrize("name", sorted(ARCHITECTURES))
    def test_build_and_forward(self, name):
        d = 64
        net = build_architecture(name, d=d, seq_len=8)
        x = torch.randn(5, d)
        out = net(x)
        assert out.shape[0] == 5
        assert torch.isfinite(out).all()

    def test_unknown_architecture(self):
        with pytest.raises(ValueError):
            build_architecture("nope", d=16)

    def test_seq_len_divisibility(self):
        with pytest.raises(ValueError):
            build_architecture("tiny_gpt", d=65, seq_len=8)

    def test_architecture_experiment_linear(self):
        exp = architecture_experiment("linear", d=32, teacher_init="normal")
        res = exp.run_sample_complexity(
            alphas=[2.0, 8.0], n_seeds=1, max_epochs=200, n_test=128, verbose=False
        )
        errs = res.mean("test_error")
        assert errs[1] < errs[0]

    def test_architecture_experiment_gpt_smoke(self):
        exp = architecture_experiment(
            "tiny_gpt",
            d=32,
            arch_kwargs={"seq_len": 4, "d_model": 8, "n_heads": 1, "n_blocks": 1},
        )
        res = exp.run_sample_complexity(
            alphas=[4.0], n_seeds=1, max_epochs=30, n_test=64, verbose=False
        )
        assert len(res.mean("test_error")) == 1


class TestSlurm:
    """sbatch rendering and dry-run submission."""

    def test_render_basic(self):
        cfg = SlurmConfig(job_name="foo", partition="debug", gpus=1,
                          time_limit="00:10:00", setup_lines=["source .venv/bin/activate"])
        script = render_sbatch("python run.py", cfg, log_dir="logs")
        assert script.startswith("#!/bin/bash")
        assert "#SBATCH --job-name=foo" in script
        assert "#SBATCH --partition=debug" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --time=00:10:00" in script
        assert "source .venv/bin/activate" in script
        assert script.rstrip().endswith("python run.py")

    def test_render_no_optional(self):
        script = render_sbatch("echo hi", SlurmConfig())
        assert "--partition" not in script
        assert "--gres" not in script
        assert "--time=" not in script

    def test_render_array(self):
        script = render_sbatch("echo $SLURM_ARRAY_TASK_ID", SlurmConfig(), array="0-3%2")
        assert "#SBATCH --array=0-3%2" in script
        assert "%A_%a.out" in script

    def test_submit_dry_run(self, tmp_path):
        launcher = SlurmLauncher(script_dir=tmp_path, log_dir="logs")
        path = launcher.submit("echo hi", SlurmConfig(job_name="dry"), dry_run=True)
        assert path.endswith("dry.sbatch")
        content = (tmp_path / "dry.sbatch").read_text()
        assert "echo hi" in content

    def test_submit_array_dry_run(self, tmp_path):
        launcher = SlurmLauncher(script_dir=tmp_path, log_dir="logs")
        path = submit_array(
            ["echo a", "echo b"], SlurmConfig(job_name="arr"),
            launcher=launcher, max_parallel=1, dry_run=True,
        )
        content = (tmp_path / "arr.sbatch").read_text()
        assert "#SBATCH --array=0-1%1" in content
        assert "0) echo a ;;" in content
        assert "1) echo b ;;" in content
        assert path.endswith("arr.sbatch")

    def test_submit_array_empty(self):
        with pytest.raises(ValueError):
            submit_array([], SlurmConfig())
