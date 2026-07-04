"""Regression tests for the 2026-07 code audit fixes."""

import numpy as np
import pytest
import torch

from statphys.loss import HingeLoss, MSELoss, RidgeLoss
from statphys.model import LinearRegression
from statphys.theory.base import TheoryResult, TheoryType


class TestTheoryTypeUnification:
    """simulation.config.TheoryType must be the canonical enum."""

    def test_single_enum(self):
        from statphys.simulation.config import TheoryType as SimTheoryType

        assert SimTheoryType is TheoryType


class TestTheoryResultDictAccess:
    """TheoryResult supports `in` / [] access used by vis.default_plots."""

    @pytest.fixture()
    def result(self):
        return TheoryResult(
            theory_type=TheoryType.REPLICA,
            order_params={"m": [0.5, 0.8], "q": [0.6, 0.9]},
            param_values=[1.0, 2.0],
            converged=[True, True],
            iterations=[10, 10],
        )

    def test_contains(self, result):
        assert "m" in result
        assert "eg" not in result

    def test_getitem(self, result):
        np.testing.assert_allclose(result["m"], [0.5, 0.8])

    def test_keys(self, result):
        assert set(result.keys()) == {"m", "q"}


class TestOnlineScale:
    """for_online applies 0.5 only to squared-error losses."""

    def test_mse_online_scale(self):
        assert MSELoss.online_scale == 0.5
        assert RidgeLoss.online_scale == 0.5

    def test_hinge_online_scale(self):
        assert HingeLoss.online_scale == 1.0

    def test_mse_for_online_value(self):
        d = 8
        model = LinearRegression(d=d)
        loss_fn = MSELoss(reg_param=0.0)
        y_pred = torch.tensor([2.0])
        y_true = torch.tensor([0.0])
        loss = loss_fn.for_online(y_pred, y_true, model, d=d)
        # (1/2)(2 - 0)^2 = 2
        assert loss.item() == pytest.approx(2.0)

    def test_ridge_regularization_uses_d(self):
        d = 8
        model = LinearRegression(d=d)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        loss_fn = RidgeLoss(reg_param=2.0)
        y_pred = torch.tensor([1.0])
        y_true = torch.tensor([1.0])
        loss = loss_fn.for_online(y_pred, y_true, model, d=d)
        # data term 0; reg = (lam/2)*||w||^2/d = (2/2)*8/8 = 1
        assert loss.item() == pytest.approx(1.0)


class TestMseSteadyState:
    """ODE integration converges to the corrected steady state."""

    def test_ode_matches_steady_state(self):
        from statphys.theory.online.scenario.gaussian_linear_mse import (
            GaussianLinearMseEquations,
        )

        eqs = GaussianLinearMseEquations(rho=1.0, eta_noise=0.1, lr=0.3, reg_param=0.05)
        m_star, q_star = eqs.steady_state()

        # Integrate ODE to long time with simple Euler
        y = np.array([0.01, 0.01])
        dt = 0.01
        for _ in range(50000):
            y = y + dt * eqs(0.0, y, {})

        assert y[0] == pytest.approx(m_star, abs=1e-4)
        assert y[1] == pytest.approx(q_star, abs=1e-4)

    def test_unstable_lr_raises(self):
        from statphys.theory.online.scenario.gaussian_linear_mse import (
            GaussianLinearMseEquations,
        )

        eqs = GaussianLinearMseEquations(rho=1.0, eta_noise=0.0, lr=3.0, reg_param=0.0)
        with pytest.raises(ValueError):
            eqs.steady_state()


class TestCommitteeOde:
    """Saad-Solla committee ODE has the correct fixed point and drift."""

    def test_fixed_point_at_perfect_learning(self):
        from statphys.theory.online.scenario.gaussian_committee_mse import (
            GaussianCommitteeMseEquations,
        )

        eqs = GaussianCommitteeMseEquations(k_student=1, k_teacher=1, rho=1.0, lr=0.1)
        dy = eqs(0.0, np.array([1.0, 1.0]), {})
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_initial_drift_positive(self):
        from statphys.theory.online.scenario.gaussian_committee_mse import (
            GaussianCommitteeMseEquations,
        )

        eqs = GaussianCommitteeMseEquations(k_student=1, k_teacher=1, rho=1.0, lr=0.1)
        dy = eqs(0.0, np.array([0.01, 0.25]), {})
        assert dy[0] > 0  # overlap must grow from small positive r

    def test_generalization_error_zero_at_perfect(self):
        from statphys.theory.online.scenario.gaussian_committee_mse import (
            GaussianCommitteeMseEquations,
        )

        eqs = GaussianCommitteeMseEquations(k_student=1, k_teacher=1, rho=1.0)
        assert eqs.generalization_error(np.array([1.0, 1.0])) == pytest.approx(0.0, abs=1e-10)


class TestSpecialFunctionsI3I4:
    """I3/I4 quadrature implementations match Monte Carlo estimates."""

    def test_i3_erf_matches_monte_carlo(self):
        from statphys.utils.special_functions import I3

        Q_ab, Q_ac, Q_bc = 0.4, 0.3, 0.5
        val = I3(Q_ab, Q_ac, Q_bc, activation="erf", n_points=30)

        rng = np.random.default_rng(0)
        cov = np.array([[1, Q_ab, Q_ac], [Q_ab, 1, Q_bc], [Q_ac, Q_bc, 1]])
        L = np.linalg.cholesky(cov)
        z = L @ rng.standard_normal((3, 500_000))
        from scipy.special import erf

        g = erf(z / np.sqrt(2))
        gp = np.sqrt(2 / np.pi) * np.exp(-(z[0] ** 2) / 2)
        mc = np.mean(gp * g[1] * g[2])

        assert val == pytest.approx(mc, abs=5e-3)


class TestOnlineSimEvalAlignment:
    """Online simulation trajectory length matches t_values length."""

    def test_trajectory_length(self):
        from statphys.dataset import GaussianDataset
        from statphys.simulation import OnlineSimulation, SimulationConfig

        d = 20
        config = SimulationConfig.for_online(
            t_max=1.0, t_steps=7, n_seeds=1, lr=0.05, verbose=False, use_theory=False
        )
        dataset = GaussianDataset(d=d, rho=1.0, eta=0.0)
        sim = OnlineSimulation(config)
        result = sim.run(dataset, LinearRegression, MSELoss(reg_param=0.0))

        traj = np.array(result.experiment_results["trajectories_mean"])
        assert traj.shape[0] == 7
