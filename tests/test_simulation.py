"""Tests for simulation module."""

import numpy as np
import pytest

from statphys.dataset import GaussianDataset
from statphys.loss import RidgeLoss
from statphys.model import LinearRegression
from statphys.simulation import (
    OnlineSimulation,
    ReplicaSimulation,
    SimulationConfig,
    SimulationRunner,
    TheoryType,
)
from statphys.utils import fix_seed


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_init(self):
        """Test default initialization."""
        config = SimulationConfig()

        assert config.theory_type == TheoryType.REPLICA
        assert config.n_seeds == 5
        assert len(config.seed_list) == 5

    def test_alpha_values_generation(self):
        """Test alpha values are generated correctly."""
        config = SimulationConfig(
            alpha_range=(0.5, 2.0),
            alpha_steps=4,
        )

        alphas = config.get_alpha_values()

        assert len(alphas) == 4
        assert alphas[0] == pytest.approx(0.5, abs=1e-6)
        assert alphas[-1] == pytest.approx(2.0, abs=1e-6)

    def test_for_replica(self):
        """Test for_replica factory method."""
        config = SimulationConfig.for_replica(
            alpha_range=(0.1, 3.0),
            n_seeds=3,
        )

        assert config.theory_type == TheoryType.REPLICA
        assert config.n_seeds == 3

    def test_for_online(self):
        """Test for_online factory method."""
        config = SimulationConfig.for_online(
            t_max=5.0,
            t_steps=50,
        )

        assert config.theory_type == TheoryType.ONLINE
        assert config.t_max == 5.0
        assert config.t_steps == 50

    def test_to_from_dict(self):
        """Test dictionary conversion."""
        config = SimulationConfig(
            alpha_range=(0.5, 2.0),
            n_seeds=3,
            lr=0.05,
        )

        d = config.to_dict()
        config2 = SimulationConfig.from_dict(d)

        assert config2.alpha_range == config.alpha_range
        assert config2.n_seeds == config.n_seeds
        assert config2.lr == config.lr


class TestReplicaSimulation:
    """Tests for ReplicaSimulation."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        fix_seed(42)
        d = 50  # Small dimension for fast tests
        dataset = GaussianDataset(d=d, rho=1.0, eta=0.0)

        config = SimulationConfig.for_replica(
            alpha_range=(1.0, 2.0),
            alpha_steps=2,
            n_seeds=2,
            max_iter=1000,
            lr=0.1,
            verbose=False,
        )

        return dataset, config

    def test_run_basic(self, simple_setup):
        """Test basic simulation run."""
        dataset, config = simple_setup

        sim = ReplicaSimulation(config)
        result = sim.run(
            dataset=dataset,
            model_class=LinearRegression,
            loss_fn=RidgeLoss(0.01),
        )

        assert result.theory_type == TheoryType.REPLICA
        assert "alpha_values" in result.experiment_results
        assert "order_params_mean" in result.experiment_results

    def test_result_shape(self, simple_setup):
        """Test result shapes."""
        dataset, config = simple_setup

        sim = ReplicaSimulation(config)
        result = sim.run(
            dataset=dataset,
            model_class=LinearRegression,
            loss_fn=RidgeLoss(0.01),
        )

        op_mean = np.array(result.experiment_results["order_params_mean"])
        op_std = np.array(result.experiment_results["order_params_std"])

        # Should have shape (n_alphas, n_params)
        assert op_mean.shape[0] == config.alpha_steps
        assert op_std.shape[0] == config.alpha_steps


class TestOnlineSimulation:
    """Tests for OnlineSimulation."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        fix_seed(42)
        d = 30  # Small dimension
        dataset = GaussianDataset(d=d, rho=1.0, eta=0.0)

        config = SimulationConfig.for_online(
            t_max=1.0,
            t_steps=5,
            n_seeds=2,
            lr=0.5 / d,
            verbose=False,
        )

        return dataset, config

    def test_run_basic(self, simple_setup):
        """Test basic online simulation."""
        dataset, config = simple_setup

        sim = OnlineSimulation(config)
        result = sim.run(
            dataset=dataset,
            model_class=LinearRegression,
            loss_fn=RidgeLoss(0.01),
        )

        assert result.theory_type == TheoryType.ONLINE
        assert "t_values" in result.experiment_results
        assert "trajectories_mean" in result.experiment_results


class TestSimulationRunner:
    """Tests for SimulationRunner."""

    def test_run_replica(self):
        """Test running replica simulation through runner."""
        fix_seed(42)
        d = 30
        dataset = GaussianDataset(d=d)

        config = SimulationConfig.for_replica(
            alpha_range=(1.0, 2.0),
            alpha_steps=2,
            n_seeds=1,
            max_iter=500,
            verbose=False,
        )

        runner = SimulationRunner()
        result = runner.run(
            config=config,
            dataset=dataset,
            model_class=LinearRegression,
            loss_fn=RidgeLoss(0.01),
        )

        assert result is not None
        assert result.theory_type == TheoryType.REPLICA

    def test_run_online(self):
        """Test running online simulation through runner."""
        fix_seed(42)
        d = 30
        dataset = GaussianDataset(d=d)

        config = SimulationConfig.for_online(
            t_max=0.5,
            t_steps=3,
            n_seeds=1,
            lr=0.5 / d,
            verbose=False,
        )

        runner = SimulationRunner()
        result = runner.run(
            config=config,
            dataset=dataset,
            model_class=LinearRegression,
            loss_fn=RidgeLoss(0.01),
        )

        assert result is not None
        assert result.theory_type == TheoryType.ONLINE
