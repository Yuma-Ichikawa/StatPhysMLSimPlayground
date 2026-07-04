"""Tests for the exact online committee-machine dynamics module."""

import numpy as np
import pytest

from statphys.experiment.online_committee import (
    committee_generalization_error,
    escape_time,
    simulate_online_committee,
    specialization_gap,
)


class TestGeneralizationError:
    """Exactness properties of the arcsin eps_g formula."""

    def test_perfect_recovery_gives_zero(self):
        k = 3
        eye = np.eye(k)
        assert committee_generalization_error(eye, eye, eye) == pytest.approx(0.0, abs=1e-12)

    def test_zero_overlap_gives_independent_sum(self):
        # R = 0: eps_g = 1/2 (E[f_s^2] + E[f_t^2]); for Q = T = I both
        # terms equal (1/pi) arcsin(1/2) per the arcsin formula.
        k = 2
        eye = np.eye(k)
        eg = committee_generalization_error(eye, np.zeros((k, k)), eye)
        expected = (2.0 / np.pi) * np.arcsin(0.5)
        assert eg == pytest.approx(expected, rel=1e-12)

    def test_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        d, k = 4000, 2
        a = rng.standard_normal((d, k))
        qm, _ = np.linalg.qr(a)
        w_t = qm.T * np.sqrt(d)
        w_s = 0.7 * w_t + 0.5 * rng.standard_normal((k, d))

        Q = w_s @ w_s.T / d
        R = w_s @ w_t.T / d
        T = w_t @ w_t.T / d
        eg_formula = committee_generalization_error(Q, R, T)

        from scipy.special import erf

        x = rng.standard_normal((200_000, d))
        h_s = x @ w_s.T / np.sqrt(d)
        h_t = x @ w_t.T / np.sqrt(d)
        f_s = erf(h_s / np.sqrt(2)).sum(axis=1) / np.sqrt(k)
        f_t = erf(h_t / np.sqrt(2)).sum(axis=1) / np.sqrt(k)
        eg_mc = 0.5 * np.mean((f_s - f_t) ** 2)

        assert eg_formula == pytest.approx(eg_mc, rel=0.03)


class TestSpecializationGap:
    """Symmetry-breaking order parameter."""

    def test_symmetric_state_is_zero(self):
        R = np.full((3, 3), 0.4)
        assert specialization_gap(R) == pytest.approx(0.0, abs=1e-12)

    def test_specialized_state_is_large(self):
        R = np.eye(3) * 0.9 + 0.05
        assert specialization_gap(R) > 0.8


class TestEscapeTime:
    """Plateau escape-time detection from the specialization gap."""

    def test_detects_crossing(self):
        t = np.linspace(0, 100, 200)
        gap = np.where(t < 60, 0.01, 0.9)
        t_esc = escape_time(t, gap)
        assert 55 < t_esc < 65

    def test_no_escape_is_nan(self):
        t = np.linspace(0, 100, 200)
        gap = np.full_like(t, 0.01)
        assert np.isnan(escape_time(t, gap))


class TestSimulation:
    """End-to-end online committee simulation."""

    def test_learns_and_specializes(self):
        traj = simulate_online_committee(
            d=128, k=2, lr=1.0, t_max=400.0, n_snapshots=150, init_scale=1e-3, seed=0
        )
        assert traj["eps_g"][0] > traj["eps_g"][-1]
        assert traj["eps_g"][-1] < 0.01
        assert traj["spec_gap"][-1] > 0.5
        assert np.isfinite(traj["escape_time"])
        assert traj["R"].shape == (len(traj["t"]), 2, 2)

    def test_shapes_and_params(self):
        traj = simulate_online_committee(d=64, k=2, m=3, lr=0.3, t_max=5.0, n_snapshots=10)
        assert traj["R"].shape[1:] == (2, 3)
        assert traj["params"]["m"] == 3
