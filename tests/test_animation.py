"""Smoke tests for statphys.vis.animation (new decision-boundary / log-axis support)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from statphys.vis.animation import animate_decision_boundary, animate_learning_curve


class TestAnimateLearningCurveLogAxes:
    """animate_learning_curve with logx/logy for epoch-scale plots."""

    def test_returns_funcanimation_with_log_axes(self):
        t = np.logspace(0, 3, 20)
        traj = {"train_error": np.linspace(10, 1e-3, 20), "test_error": np.linspace(15, 0.5, 20)}
        anim = animate_learning_curve(t, traj, xlabel="epoch", ylabel="error", logx=True, logy=True)
        assert isinstance(anim, FuncAnimation)
        assert anim._save_count == 20

    def test_linear_axes_still_work(self):
        t = np.linspace(0, 10, 15)
        traj = {"m": np.linspace(0, 1, 15)}
        anim = animate_learning_curve(t, traj)
        assert isinstance(anim, FuncAnimation)


class TestAnimateDecisionBoundary:
    """animate_decision_boundary geometry and API."""

    def test_returns_funcanimation(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        y = np.sign(X[:, 0])
        weights = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        anim = animate_decision_boundary(X, y, weights, metric_values=np.array([0.5, 0.3, 0.1]))
        assert isinstance(anim, FuncAnimation)
        assert anim._save_count == 3

    def test_rejects_non_2d_inputs(self):
        X = np.random.randn(10, 3)
        y = np.sign(X[:, 0])
        with pytest.raises(ValueError):
            animate_decision_boundary(X, y, [np.array([1.0, 0.0, 0.0])])

    def test_boundary_is_perpendicular_to_weight(self):
        # For w = (1, 0), the boundary {x : w.x = 0} is the vertical line x1=0,
        # i.e. direction (0, 1) -- check the update() logic reproduces this.
        X = np.array([[1.0, 0.0], [-1.0, 0.0]])
        y = np.array([1.0, -1.0])
        anim = animate_decision_boundary(X, y, [np.array([1.0, 0.0])])
        artists = anim._func(0)
        boundary_line = artists[0]
        xs, ys = boundary_line.get_data()
        # the boundary line should run purely along x2 (x1 ~ 0 everywhere)
        assert np.allclose(xs, 0.0, atol=1e-8)
        assert ys[0] != ys[1]
