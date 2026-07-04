"""Tests for the general teacher-student experiment framework."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from statphys.experiment import (
    ExperimentResult,
    Teacher,
    TeacherStudentDataset,
    TeacherStudentExperiment,
    get_preset,
    init_weights_,
    linear_cka,
    weight_overlap,
)
from statphys.experiment import test_error as compute_test_error


class TestInitWeights:
    """Weight-initialization strategies."""

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("normal", {}),
            ("sparse", {"sparsity": 0.9}),
            ("low_rank", {"rank": 2}),
            ("orthogonal", {}),
            ("power_law", {"alpha": 3.0}),
            ("binary", {}),
            ("spiked", {"snr": 2.0}),
        ],
    )
    def test_methods_run(self, method, kwargs):
        net = nn.Linear(50, 10)
        init_weights_(net, method=method, **kwargs)
        assert torch.isfinite(net.weight).all()
        assert (net.bias == 0).all()

    def test_sparse_actually_sparse(self):
        net = nn.Linear(100, 20)
        init_weights_(net, method="sparse", sparsity=0.9)
        frac_zero = (net.weight == 0).float().mean().item()
        assert frac_zero > 0.8

    def test_low_rank_rank(self):
        net = nn.Linear(50, 30)
        init_weights_(net, method="low_rank", rank=3)
        rank = torch.linalg.matrix_rank(net.weight).item()
        assert rank <= 3

    def test_binary_values(self):
        net = nn.Linear(20, 5)
        init_weights_(net, method="binary")
        vals = net.weight.abs().unique()
        assert len(vals) == 1

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            init_weights_(nn.Linear(4, 2), method="nope")


class TestTeacher:
    """Teacher wrapper behavior."""

    def test_regression_labels(self):
        teacher = Teacher(nn.Linear(10, 1, bias=False), init="normal")
        x = torch.randn(32, 10)
        y = teacher(x)
        assert y.shape == (32,)

    def test_sign_readout(self):
        teacher = Teacher(nn.Linear(10, 1, bias=False), init="normal", readout="sign")
        y = teacher(torch.randn(64, 10))
        assert set(y.unique().tolist()) <= {-1.0, 1.0}

    def test_frozen(self):
        teacher = Teacher(nn.Linear(10, 1), init="normal")
        assert all(not p.requires_grad for p in teacher.model.parameters())

    def test_named_weights(self):
        teacher = Teacher(nn.Linear(10, 1), init="normal")
        assert "weight" in teacher.named_weights()


class TestDataset:
    """Input distributions."""

    @pytest.mark.parametrize("dist", ["gaussian", "rademacher", "sphere"])
    def test_distributions(self, dist):
        teacher = Teacher(nn.Linear(16, 1, bias=False), init="normal")
        ds = TeacherStudentDataset(teacher, d=16, input_dist=dist)
        X, y = ds.sample(20)
        assert X.shape == (20, 16)
        assert y.shape == (20,)

    def test_correlated_ar1(self):
        teacher = Teacher(nn.Linear(16, 1, bias=False), init="normal")
        ds = TeacherStudentDataset(
            teacher, d=16, input_dist="correlated", input_kwargs={"ar_coeff": 0.8}
        )
        X, _ = ds.sample(5000)
        corr = torch.corrcoef(X.T)[0, 1].item()
        assert corr == pytest.approx(0.8, abs=0.1)

    def test_sphere_norm(self):
        teacher = Teacher(nn.Linear(16, 1, bias=False), init="normal")
        ds = TeacherStudentDataset(teacher, d=16, input_dist="sphere")
        X, _ = ds.sample(10)
        np.testing.assert_allclose(X.norm(dim=1).numpy(), np.full(10, 4.0), atol=1e-4)


class TestMetrics:
    """Model-agnostic observables."""

    def test_test_error_zero_for_teacher(self):
        model = nn.Linear(20, 1, bias=False)
        teacher = Teacher(model, init="normal")
        ds = TeacherStudentDataset(teacher, d=20)
        err = compute_test_error(model, ds, n_test=256)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_weight_overlap_self(self):
        model = nn.Linear(20, 1, bias=False)
        teacher = Teacher(model, init="normal")
        ov = weight_overlap(model, teacher.named_weights())
        assert ov["weight"] == pytest.approx(1.0, abs=1e-6)

    def test_cka_identical(self):
        X = torch.randn(100, 8)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-5)


class TestExperiment:
    """End-to-end protocol runs (tiny sizes)."""

    def _make(self):
        teacher = Teacher(nn.Linear(16, 1, bias=False), init="normal", device="cpu")
        return TeacherStudentExperiment(
            teacher=teacher,
            student_factory=lambda: nn.Linear(16, 1, bias=False),
        )

    def test_sample_complexity(self):
        exp = self._make()
        res = exp.run_sample_complexity(
            alphas=[1.0, 4.0], n_seeds=2, max_epochs=200, n_test=128, verbose=False
        )
        assert res.x_name == "alpha"
        assert res.mean("test_error").shape == (2,)
        # Error should drop with more data
        assert res.mean("test_error")[1] < res.mean("test_error")[0]

    def test_online(self):
        exp = self._make()
        res = exp.run_online(t_max=2.0, t_steps=5, n_seeds=1, lr=0.1, n_test=128, verbose=False)
        assert res.x_name == "t"
        assert len(res.x_values) == len(res.mean("test_error"))

    def test_result_roundtrip(self):
        exp = self._make()
        res = exp.run_sample_complexity(
            alphas=[2.0], n_seeds=1, max_epochs=50, n_test=64, verbose=False
        )
        restored = ExperimentResult.from_dict(res.to_dict())
        assert restored.x_values == res.x_values

    def test_dim_inference(self):
        teacher = Teacher(nn.Sequential(nn.Linear(24, 4), nn.Tanh(), nn.Linear(4, 1)))
        exp = TeacherStudentExperiment(
            teacher=teacher, student_factory=lambda: nn.Linear(24, 1)
        )
        assert exp.d == 24


class TestPresets:
    """Preset factory."""

    def test_get_preset(self):
        exp = get_preset("random_mlp", d=32, hidden=2)
        assert exp.d == 32

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent")
