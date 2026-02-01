"""
Pytest configuration and shared fixtures for StatPhys-ML tests.
"""

import pytest
import torch
import numpy as np

from statphys.utils import fix_seed


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before each test for reproducibility."""
    fix_seed(42)
    yield


@pytest.fixture
def device():
    """Get available device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def small_d():
    """Small dimension for fast tests."""
    return 50


@pytest.fixture
def medium_d():
    """Medium dimension for moderate tests."""
    return 200


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 32


@pytest.fixture
def gaussian_dataset(small_d):
    """Create a basic Gaussian dataset for testing."""
    from statphys.dataset import GaussianDataset

    return GaussianDataset(d=small_d, rho=1.0, eta=0.1)


@pytest.fixture
def linear_model(small_d):
    """Create a linear regression model for testing."""
    from statphys.model import LinearRegression

    return LinearRegression(d=small_d)


@pytest.fixture
def ridge_loss():
    """Create a ridge loss function for testing."""
    from statphys.loss import RidgeLoss

    return RidgeLoss(reg_param=0.01)


@pytest.fixture
def replica_config():
    """Create a minimal replica simulation config for fast tests."""
    from statphys.simulation import SimulationConfig

    return SimulationConfig.for_replica(
        alpha_range=(1.0, 2.0),
        alpha_steps=2,
        n_seeds=2,
        max_iter=500,
        lr=0.1,
        verbose=False,
    )


@pytest.fixture
def online_config(small_d):
    """Create a minimal online simulation config for fast tests."""
    from statphys.simulation import SimulationConfig

    return SimulationConfig.for_online(
        t_max=0.5,
        t_steps=5,
        n_seeds=2,
        lr=0.5 / small_d,
        verbose=False,
    )
