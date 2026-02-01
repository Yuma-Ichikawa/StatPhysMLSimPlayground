"""
Base classes for simulations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch

from statphys.simulation.config import SimulationConfig, TheoryType
from statphys.theory.base import TheoryResult


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Attributes:
        theory_type: Type of simulation (REPLICA, ONLINE).
        experiment_results: Results from numerical experiments.
        theory_results: Results from theoretical calculations (if computed).
        config: Simulation configuration used.
        metadata: Additional metadata.
    """

    theory_type: TheoryType
    experiment_results: Dict[str, Any]
    theory_results: Optional[TheoryResult] = None
    config: Optional[SimulationConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_experiment_mean(self, key: str) -> np.ndarray:
        """Get mean of experiment results across seeds."""
        data = np.array(self.experiment_results[key])
        return np.mean(data, axis=0)

    def get_experiment_std(self, key: str) -> np.ndarray:
        """Get std of experiment results across seeds."""
        data = np.array(self.experiment_results[key])
        return np.std(data, axis=0)

    def get_theory_values(self, key: str) -> np.ndarray:
        """Get theory values for a parameter."""
        if self.theory_results is None:
            raise ValueError("No theory results available")
        return np.array(self.theory_results.order_params[key])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theory_type": self.theory_type.value,
            "experiment_results": self.experiment_results,
            "theory_results": self.theory_results.to_dict() if self.theory_results else None,
            "config": self.config.to_dict() if self.config else None,
            "metadata": self.metadata,
        }


class BaseSimulation(ABC):
    """
    Abstract base class for simulations.

    Provides common functionality for running experiments
    with multiple seeds and comparing to theory.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def run(
        self,
        dataset: Any,
        model_class: Type,
        loss_fn: Callable,
        calc_order_params: Optional[Callable] = None,
        theory_solver: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Run the simulation.

        Args:
            dataset: Dataset instance or class.
            model_class: Model class to instantiate.
            loss_fn: Loss function.
            calc_order_params: Function to compute order parameters.
            theory_solver: Theory solver for comparison.
            **kwargs: Additional arguments.

        Returns:
            SimulationResult containing experiment and theory results.
        """
        pass

    def _get_optimizer(
        self,
        model: torch.nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        """Get optimizer based on config."""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _print_progress(self, message: str) -> None:
        """Print progress if verbose."""
        if self.config.verbose:
            print(message)
