"""Tests for physics-style order parameters and phase-diagram protocols."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from statphys.experiment import (
    Teacher,
    TeacherStudentExperiment,
    binder_cumulant,
    function_order_params,
    participation_ratio,
    replica_overlaps,
    run_phase_diagram,
    specialization_index,
    susceptibility,
)


@pytest.fixture
def linear_teacher():
    """Fixed random linear teacher."""
    torch.manual_seed(0)
    return Teacher(nn.Linear(20, 1, bias=False), init="normal")


class TestFunctionOrderParams:
    """function_order_params behaviour."""

    def test_perfect_student_has_unit_overlap(self, linear_teacher):
        X = torch.randn(512, 20)
        params = function_order_params(linear_teacher.model, linear_teacher, X)
        assert params["m_hat"] == pytest.approx(1.0, abs=1e-5)
        assert params["q_f"] == pytest.approx(params["rho_f"], rel=1e-5)

    def test_orthogonal_student_has_zero_overlap(self, linear_teacher):
        torch.manual_seed(123)
        stranger = nn.Linear(20, 1, bias=False)
        X = torch.randn(4096, 20)
        params = function_order_params(stranger, linear_teacher, X)
        assert abs(params["m_hat"]) < 0.5  # random directions are near-orthogonal

    def test_noise_does_not_affect_m_hat(self):
        torch.manual_seed(0)
        noisy = Teacher(nn.Linear(20, 1, bias=False), init="normal", noise_std=5.0)
        X = torch.randn(512, 20)
        params = function_order_params(noisy.model, noisy, X)
        assert params["m_hat"] == pytest.approx(1.0, abs=1e-5)


class TestReplicaOverlaps:
    """replica_overlaps behaviour."""

    def test_identical_replicas_give_q_one(self, linear_teacher):
        X = torch.randn(256, 20)
        m = linear_teacher.model
        out = replica_overlaps([m, m, m], X)
        assert out["q_ab_mean"] == pytest.approx(1.0, abs=1e-6)
        assert len(out["q_ab_pairs"]) == 3

    def test_independent_replicas_give_small_q(self):
        torch.manual_seed(0)
        models = [nn.Linear(50, 1, bias=False) for _ in range(4)]
        X = torch.randn(4096, 50)
        out = replica_overlaps(models, X)
        assert abs(out["q_ab_mean"]) < 0.5


class TestScalarObservables:
    """susceptibility / Binder / participation ratio."""

    def test_susceptibility_zero_for_constant(self):
        assert susceptibility([0.7, 0.7, 0.7], scale=100) == pytest.approx(0.0)

    def test_susceptibility_scales(self):
        base = susceptibility([0.1, 0.5, 0.9], scale=1.0)
        assert susceptibility([0.1, 0.5, 0.9], scale=10.0) == pytest.approx(10 * base)

    def test_binder_gaussian_near_zero(self):
        rng = np.random.default_rng(0)
        u4 = binder_cumulant(rng.standard_normal(200_000))
        assert abs(u4) < 0.02  # U4 = 0 for a centered Gaussian

    def test_binder_ordered_phase(self):
        # delta-distributed order parameter -> U4 = 2/3
        assert binder_cumulant([0.8] * 10) == pytest.approx(2.0 / 3.0)

    def test_participation_ratio_bounds(self):
        torch.manual_seed(0)
        iso = torch.randn(2000, 16)
        pr_iso = participation_ratio(iso)
        assert 10 < pr_iso <= 16.5
        collapsed = torch.randn(2000, 1).expand(2000, 16) + 0.001 * torch.randn(2000, 16)
        assert participation_ratio(collapsed) < 2.0


class TestSpecializationIndex:
    """specialization_index behaviour."""

    def test_teacher_is_fully_specialized(self):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(30, 4), nn.Tanh(), nn.Linear(4, 1))
        teacher = Teacher(net, init="orthogonal")
        idx = specialization_index(teacher.model, teacher)
        assert idx > 0.8

    def test_shape_mismatch_gives_nan(self, linear_teacher):
        student = nn.Sequential(nn.Linear(7, 3), nn.Tanh(), nn.Linear(3, 1))
        assert np.isnan(specialization_index(student, linear_teacher))


class TestOrderParameterProtocol:
    """run_order_parameters and run_phase_diagram smoke tests."""

    def test_run_order_parameters_smoke(self):
        torch.manual_seed(0)
        d = 24
        teacher = Teacher(nn.Linear(d, 1, bias=False), init="normal", noise_std=0.05)
        exp = TeacherStudentExperiment(
            teacher=teacher, student_factory=lambda: nn.Linear(d, 1, bias=False), d=d
        )
        res = exp.run_order_parameters(
            alphas=[0.5, 4.0],
            n_replicas=3,
            lr=5e-2,
            max_epochs=200,
            n_probe=512,
            verbose=False,
        )
        for key in ("m_hat", "q_ab_mean", "chi_m", "binder_m", "test_error"):
            assert key in res.records
        assert len(res.records["m_hat"]) == 3  # per-replica rows
        assert len(res.records["q_ab_mean"]) == 1  # aggregate row
        # learning improves the overlap between alpha=0.5 and alpha=4
        m = res.mean("m_hat")
        assert m[1] > m[0]

    def test_phase_diagram_smoke(self):
        d = 16

        def factory(noise: float) -> TeacherStudentExperiment:
            teacher = Teacher(nn.Linear(d, 1, bias=False), init="normal", noise_std=noise)
            return TeacherStudentExperiment(
                teacher=teacher,
                student_factory=lambda: nn.Linear(d, 1, bias=False),
                d=d,
            )

        res = run_phase_diagram(
            factory,
            param_name="noise_std",
            param_values=[0.0, 0.5],
            alphas=[1.0, 4.0],
            n_replicas=2,
            lr=5e-2,
            max_epochs=150,
            n_probe=256,
            verbose=False,
        )
        assert res.grid("m_hat").shape == (2, 2)
        assert "chi_m" in res.grids
        d_out = res.to_dict()
        assert d_out["param_name"] == "noise_std"
