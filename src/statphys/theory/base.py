"""Base classes for theoretical calculations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TheoryType(Enum):
    """Enum for different theory types."""

    REPLICA = "replica"
    ONLINE = "online"
    DMFT = "dmft"
    CAVITY = "cavity"


@dataclass
class TheoryResult:
    """
    Container for theory calculation results.

    Attributes:
        theory_type: Type of theory used.
        order_params: Dictionary of order parameters at each point.
        param_values: Parameter values (alpha for replica, time for online).
        converged: Whether the solver converged at each point.
        iterations: Number of iterations at each point.
        metadata: Additional metadata about the calculation.

    """

    theory_type: TheoryType
    order_params: dict[str, list[float]]
    param_values: list[float]
    converged: list[bool]
    iterations: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theory_type": self.theory_type.value,
            "order_params": self.order_params,
            "param_values": self.param_values,
            "converged": self.converged,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TheoryResult":
        """Create from dictionary."""
        return cls(
            theory_type=TheoryType(data["theory_type"]),
            order_params=data["order_params"],
            param_values=data["param_values"],
            converged=data["converged"],
            iterations=data["iterations"],
            metadata=data.get("metadata", {}),
        )

    def get_order_param(self, name: str) -> np.ndarray:
        """Get a specific order parameter as numpy array."""
        return np.array(self.order_params[name])

    def __repr__(self) -> str:
        n_points = len(self.param_values)
        n_converged = sum(self.converged)
        return (
            f"TheoryResult(type={self.theory_type.value}, "
            f"points={n_points}, converged={n_converged}/{n_points})"
        )


class BaseTheory(ABC):
    """
    Abstract base class for theoretical calculations.

    All theory solvers (Replica, Online, DMFT) inherit from this class.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize BaseTheory.

        Args:
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            verbose: Whether to print progress.

        """
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    @abstractmethod
    def solve(self, **kwargs: Any) -> TheoryResult:
        """
        Solve the theoretical equations.

        Returns:
            TheoryResult containing the solution.

        """
        pass

    @abstractmethod
    def get_theory_type(self) -> TheoryType:
        """Return the theory type."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Get solver configuration."""
        return {
            "class": self.__class__.__name__,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
        }
