"""Tests for theory module."""

import numpy as np
import pytest

from statphys.theory.base import TheoryResult, TheoryType
from statphys.theory.online import GaussianLinearMseEquations, ODESolver
from statphys.theory.replica import GaussianLinearRidgeEquations, SaddlePointSolver


class TestSaddlePointSolver:
    """Tests for SaddlePointSolver."""

    def test_simple_equations(self):
        """Test with simple fixed-point equations."""

        # Simple equations that converge to m=1, q=1
        def simple_equations(m, q, alpha, **kwargs):
            new_m = 0.9 * m + 0.1
            new_q = 0.9 * q + 0.1
            return new_m, new_q

        solver = SaddlePointSolver(
            equations=simple_equations,
            order_params=["m", "q"],
            damping=0.5,
            tol=1e-6,
            max_iter=1000,
        )

        result = solver.solve(alpha_values=[1.0], init_values=(0.5, 0.5))

        assert result.converged[0]
        assert result.order_params["m"][0] == pytest.approx(1.0, abs=1e-4)
        assert result.order_params["q"][0] == pytest.approx(1.0, abs=1e-4)

    def test_multiple_alphas(self):
        """Test solving for multiple alpha values."""

        def equations(m, q, alpha, **kwargs):
            return 0.9 * m + 0.05 * alpha, 0.9 * q + 0.05 * alpha

        solver = SaddlePointSolver(
            equations=equations,
            order_params=["m", "q"],
            damping=0.5,
            tol=1e-6,
        )

        alpha_values = [0.5, 1.0, 2.0]
        result = solver.solve(alpha_values=alpha_values)

        assert len(result.param_values) == 3
        assert len(result.order_params["m"]) == 3

    def test_continuation(self):
        """Test that continuation improves convergence."""

        def equations(m, q, alpha, **kwargs):
            return alpha / (1 + q), m**2 + 0.1

        solver = SaddlePointSolver(
            equations=equations,
            order_params=["m", "q"],
            damping=0.3,
            tol=1e-6,
        )

        alpha_values = np.linspace(0.1, 2.0, 10)

        # With continuation
        result = solver.solve(alpha_values=alpha_values, use_continuation=True)

        # Should converge for most points
        assert sum(result.converged) > 5


class TestGaussianLinearRidgeEquations:
    """Tests for GaussianLinearRidgeEquations."""

    def test_init(self):
        """Test initialization."""
        equations = GaussianLinearRidgeEquations(rho=1.0, eta=0.1, reg_param=0.01)

        assert equations.rho == 1.0
        assert equations.eta == 0.1
        assert equations.reg_param == 0.01

    def test_generalization_error(self):
        """Test generalization error formula."""
        equations = GaussianLinearRidgeEquations(rho=1.0)

        # At perfect learning: m=1, q=1, E_g should be 0
        eg = equations.generalization_error(m=1.0, q=1.0, rho=1.0)
        assert eg == pytest.approx(0.0, abs=1e-6)


class TestODESolver:
    """Tests for ODESolver."""

    def test_exponential_decay(self):
        """Test with simple exponential decay ODE."""

        def decay_ode(t, y, params):
            rate = params.get("rate", 1.0)
            return -rate * y

        solver = ODESolver(
            equations=decay_ode,
            order_params=["x"],
            tol=1e-8,
        )

        result = solver.solve(
            t_span=(0, 5),
            init_values=(1.0,),
            n_points=50,
            rate=1.0,
        )

        # Should decay exponentially
        x_final = result.order_params["x"][-1]
        expected = np.exp(-5)
        assert x_final == pytest.approx(expected, abs=1e-4)

    def test_theory_type(self):
        """Test theory type."""

        def ode(t, y, params):
            return np.zeros_like(y)

        solver = ODESolver(equations=ode, order_params=["x"])
        assert solver.get_theory_type() == TheoryType.ONLINE


class TestGaussianLinearMseEquations:
    """Tests for GaussianLinearMseEquations."""

    def test_init(self):
        """Test initialization."""
        equations = GaussianLinearMseEquations(rho=1.0, eta_noise=0.1, lr=0.1)

        assert equations.rho == 1.0
        assert equations.eta_noise == 0.1
        assert equations.lr == 0.1

    def test_equilibrium(self):
        """Test that equations have sensible equilibrium."""
        equations = GaussianLinearMseEquations(rho=1.0, eta_noise=0.0, lr=0.1, reg_param=0.0)

        # At m=rho, q=rho, derivatives should be small
        dy = equations(t=10.0, y=np.array([1.0, 1.0]), params={})

        # dm/dt should be close to 0 when m = rho
        assert abs(dy[0]) < 0.1


class TestTheoryResult:
    """Tests for TheoryResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TheoryResult(
            theory_type=TheoryType.REPLICA,
            order_params={"m": [0.5, 0.8], "q": [0.6, 0.9]},
            param_values=[1.0, 2.0],
            converged=[True, True],
            iterations=[100, 150],
        )

        d = result.to_dict()

        assert d["theory_type"] == "replica"
        assert d["order_params"]["m"] == [0.5, 0.8]

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "theory_type": "online",
            "order_params": {"m": [0.5]},
            "param_values": [1.0],
            "converged": [True],
            "iterations": [0],
        }

        result = TheoryResult.from_dict(d)

        assert result.theory_type == TheoryType.ONLINE
        assert result.order_params["m"] == [0.5]

    def test_get_order_param(self):
        """Test order parameter retrieval."""
        result = TheoryResult(
            theory_type=TheoryType.REPLICA,
            order_params={"m": [0.5, 0.8, 1.0]},
            param_values=[1.0, 2.0, 3.0],
            converged=[True, True, True],
            iterations=[100, 100, 100],
        )

        m = result.get_order_param("m")
        assert isinstance(m, np.ndarray)
        assert len(m) == 3
