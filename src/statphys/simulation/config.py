"""
Configuration classes for simulations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TheoryType(Enum):
    """Enum for theory types."""

    REPLICA = "replica"
    ONLINE = "online"
    DMFT = "dmft"


@dataclass
class SimulationConfig:
    """
    Configuration for simulation experiments.

    Attributes:
        theory_type: Type of theory (REPLICA, ONLINE, DMFT).
        use_theory: Whether to compute theoretical predictions.
        n_seeds: Number of random seeds for averaging.
        seed_list: Specific seeds to use (overrides n_seeds).
        device: Computation device ('cpu' or 'cuda').

        # Replica-specific
        alpha_range: Range of alpha values (min, max).
        alpha_steps: Number of alpha values.
        alpha_values: Explicit alpha values (overrides range).

        # Online-specific
        t_max: Maximum time (in units of n/d).
        t_steps: Number of time steps.

        # Training parameters
        lr: Learning rate.
        max_iter: Maximum training iterations.
        tol: Convergence tolerance.
        patience: Early stopping patience.
        optimizer: Optimizer name ('sgd', 'adam', 'adamw').

        # Regularization
        reg_param: Regularization parameter λ.

        # Verbosity
        verbose: Whether to print progress.
        verbose_interval: Print interval.
    """

    # Theory settings
    theory_type: TheoryType = TheoryType.REPLICA
    use_theory: bool = True

    # Seed settings
    n_seeds: int = 5
    seed_list: Optional[List[int]] = None
    base_seed: int = 100

    # Device
    device: str = "cpu"

    # Replica-specific settings
    alpha_range: Tuple[float, float] = (0.1, 5.0)
    alpha_steps: int = 20
    alpha_values: Optional[List[float]] = None

    # Online-specific settings
    t_max: float = 10.0
    t_steps: int = 100

    # Training parameters
    lr: float = 0.01
    max_iter: int = 50000
    tol: float = 1e-6
    patience: int = 100
    optimizer: str = "adamw"

    # Regularization
    reg_param: float = 0.01

    # Verbosity
    verbose: bool = True
    verbose_interval: int = 5000

    def __post_init__(self):
        """Post-initialization processing."""
        # Generate alpha values if not provided
        if self.alpha_values is None:
            self.alpha_values = np.linspace(
                self.alpha_range[0], self.alpha_range[1], self.alpha_steps
            ).tolist()

        # Generate seed list if not provided
        if self.seed_list is None:
            self.seed_list = [self.base_seed + i for i in range(self.n_seeds)]
        else:
            self.n_seeds = len(self.seed_list)

    def get_alpha_values(self) -> np.ndarray:
        """Get alpha values as numpy array."""
        return np.array(self.alpha_values)

    def get_time_values(self) -> np.ndarray:
        """Get time values for online learning."""
        return np.linspace(0, self.t_max, self.t_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theory_type": self.theory_type.value,
            "use_theory": self.use_theory,
            "n_seeds": self.n_seeds,
            "seed_list": self.seed_list,
            "device": self.device,
            "alpha_range": self.alpha_range,
            "alpha_steps": self.alpha_steps,
            "alpha_values": self.alpha_values,
            "t_max": self.t_max,
            "t_steps": self.t_steps,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "reg_param": self.reg_param,
            "verbose": self.verbose,
            "verbose_interval": self.verbose_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary."""
        data = data.copy()
        if "theory_type" in data:
            data["theory_type"] = TheoryType(data["theory_type"])
        return cls(**data)

    @classmethod
    def for_replica(
        cls,
        alpha_range: Tuple[float, float] = (0.1, 5.0),
        alpha_steps: int = 20,
        n_seeds: int = 5,
        **kwargs: Any,
    ) -> "SimulationConfig":
        """Create configuration for replica experiments."""
        return cls(
            theory_type=TheoryType.REPLICA,
            alpha_range=alpha_range,
            alpha_steps=alpha_steps,
            n_seeds=n_seeds,
            **kwargs,
        )

    @classmethod
    def for_online(
        cls,
        t_max: float = 10.0,
        t_steps: int = 100,
        n_seeds: int = 5,
        **kwargs: Any,
    ) -> "SimulationConfig":
        """Create configuration for online learning experiments."""
        return cls(
            theory_type=TheoryType.ONLINE,
            t_max=t_max,
            t_steps=t_steps,
            n_seeds=n_seeds,
            **kwargs,
        )
