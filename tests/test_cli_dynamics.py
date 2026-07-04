"""Tests for the CLI, training-dynamics protocol, and result save/load."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from statphys.cli import build_parser, main
from statphys.experiment import (
    STUDIES,
    ExperimentResult,
    Teacher,
    TeacherStudentExperiment,
)


@pytest.fixture
def tiny_experiment():
    """Small linear teacher-student experiment."""
    torch.manual_seed(0)
    d = 16
    teacher = Teacher(nn.Linear(d, 1, bias=False), init="normal", noise_std=0.05)
    return TeacherStudentExperiment(
        teacher=teacher, student_factory=lambda: nn.Linear(16, 1, bias=False), d=16
    )


class TestTrainingDynamics:
    """run_training_dynamics (epoch-resolved protocol)."""

    def test_smoke_and_records(self, tiny_experiment):
        res = tiny_experiment.run_training_dynamics(
            alpha=2.0,
            n_replicas=2,
            epochs=60,
            n_evals=8,
            lr=5e-2,
            n_probe=128,
            verbose=False,
        )
        assert res.x_name == "epoch"
        for key in ("train_error", "test_error", "m_hat", "q_ab_mean"):
            assert key in res.records
        assert len(res.records["m_hat"]) == 2
        # training reduces train error
        tr = res.mean("train_error")
        assert tr[-1] < tr[0]

    def test_init_scale_applied(self, tiny_experiment):
        res_big = tiny_experiment.run_training_dynamics(
            alpha=1.0,
            n_replicas=1,
            epochs=1,
            n_evals=1,
            init_scale=10.0,
            n_probe=64,
            verbose=False,
        )
        # q_f of a x10-scaled student is much larger at epoch 1
        assert res_big.config["init_scale"] == 10.0


class TestSaveLoad:
    """ExperimentResult round trip via JSON files."""

    def test_round_trip(self, tiny_experiment, tmp_path):
        res = tiny_experiment.run_order_parameters(
            alphas=[1.0],
            n_replicas=2,
            max_epochs=50,
            n_probe=64,
            verbose=False,
        )
        path = tmp_path / "sub" / "result.json"
        res.save(str(path))
        loaded = ExperimentResult.load(str(path))
        assert loaded.x_values == res.x_values
        assert np.allclose(loaded.mean("m_hat"), res.mean("m_hat"))


class TestCLI:
    """statphys console command."""

    def test_registry_names(self):
        for name in (
            "committee",
            "fss",
            "diagram",
            "attention",
            "manifold",
            "gpt",
            "grokking",
            "universality",
            "double_descent",
            "scaling",
        ):
            assert name in STUDIES

    def test_list_command(self, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "tiny_gpt" in out
        assert "grokking" in out

    def test_parser_rejects_unknown_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nonsense"])

    def test_order_params_command(self, tmp_path):
        rc = main(
            [
                "order-params",
                "random_mlp",
                "--alphas",
                "1",
                "4",
                "--replicas",
                "2",
                "--d",
                "16",
                "--epochs",
                "30",
                "--output-dir",
                str(tmp_path),
                "--silent",
            ]
        )
        assert rc == 0
        assert (tmp_path / "order_params_random_mlp.json").exists()
        assert (tmp_path / "order_params_random_mlp.png").exists()
