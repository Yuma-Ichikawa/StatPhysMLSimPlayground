"""
Simulation module for numerical experiments.

This module provides:
- SimulationConfig: Configuration for experiments
- ReplicaSimulation: Static analysis experiments
- OnlineSimulation: Online learning experiments
- SimulationRunner: Unified experiment runner

Example:
    >>> from statphys.simulation import ReplicaSimulation, SimulationConfig
    >>> from statphys.dataset import GaussianDataset
    >>> from statphys.model import LinearRegression
    >>>
    >>> config = SimulationConfig(n_seeds=5, alpha_range=(0.1, 5.0))
    >>> sim = ReplicaSimulation(config)
    >>> results = sim.run(dataset, LinearRegression, loss_fn)
"""

from statphys.simulation.config import SimulationConfig, TheoryType
from statphys.simulation.base import BaseSimulation, SimulationResult
from statphys.simulation.replica_sim import ReplicaSimulation
from statphys.simulation.online_sim import OnlineSimulation
from statphys.simulation.runner import SimulationRunner

__all__ = [
    # Configuration
    "SimulationConfig",
    "TheoryType",
    # Base classes
    "BaseSimulation",
    "SimulationResult",
    # Specific simulations
    "ReplicaSimulation",
    "OnlineSimulation",
    # Runner
    "SimulationRunner",
]
