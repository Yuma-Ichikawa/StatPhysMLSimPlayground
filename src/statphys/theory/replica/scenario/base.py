"""Base class for replica saddle-point equations.

Each scenario is determined by: Data × Model × Loss
- Data: Input distribution (gaussian, sparse, structured, etc.)
- Model: Student model architecture (linear, committee, two-layer, etc.)
- Loss: Loss function (mse, ridge, lasso, logistic, hinge, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from statphys.utils.special_functions import (
    classification_error_linear,
    regression_error_linear,
)


class ReplicaEquations(ABC):
    """
    Abstract base class for replica saddle-point equations.

    Replica method derives self-consistent equations for order parameters
    in the high-dimensional limit (n, d → ∞ with α = n/d fixed).

    The equations can be written in two forms:

    1. **Fixed-point form** (currently used by solver):
       (m_new, q_new, ...) = F(m, q, ...; α, params)

       The solver iterates: x_{n+1} = (1-γ)x_n + γF(x_n)

    2. **Residual form** (0 = right-hand side):
       0 = G(m, q, ...; α, params)

       This is equivalent to F(x) - x = 0 in fixed-point form.

    Subclasses implement specific equations for different learning scenarios.

    Mathematical Background:
        The replica method computes:
        1. Average free energy via replica trick
        2. Saddle-point approximation (large n, d limit)
        3. Order parameters that extremize the free energy

    Usage:
        class GaussianLinearRidgeEquations(ReplicaEquations):
            def __call__(self, m, q, alpha, **kwargs):
                # Compute updated order parameters
                new_m = ...
                new_q = ...
                return new_m, new_q

            def generalization_error(self, m, q, **kwargs):
                return 0.5 * (rho - 2*m + q)
    """

    def __init__(self, **params: Any):
        """
        Initialize with problem-specific parameters.

        Args:
            **params: Parameters like rho, eta, lambda, etc.
        """
        self.params = params

    @abstractmethod
    def __call__(
        self,
        *order_params: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, ...]:
        """
        Compute updated order parameters (fixed-point iteration form).

        The solver calls this function iteratively:
            x_{n+1} = F(x_n)

        until convergence: ||x_{n+1} - x_n|| < tol.

        Args:
            *order_params: Current order parameter values (m, q, ...).
            alpha: Sample ratio n/d.
            **kwargs: Additional parameters (can override init params).

        Returns:
            Tuple of updated order parameter values.
        """
        pass

    def residual(
        self,
        *order_params: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, ...]:
        """
        Compute residual (0 = right-hand side form).

        This returns G(x) = F(x) - x, where F is the fixed-point map.
        The saddle-point equations are: 0 = G(x).

        Args:
            *order_params: Current order parameter values.
            alpha: Sample ratio n/d.
            **kwargs: Additional parameters.

        Returns:
            Tuple of residuals. Solution satisfies all residuals ≈ 0.
        """
        new_params = self(*order_params, alpha=alpha, **kwargs)
        return tuple(new - old for new, old in zip(new_params, order_params, strict=False))

    @abstractmethod
    def generalization_error(
        self,
        *order_params: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error from order parameters.

        For regression (MSE): E_g = (1/2)(ρ - 2m + q)
        For classification: P(error) = (1/π) arccos(m/√(qρ))

        Args:
            *order_params: Order parameter values.
            **kwargs: Additional parameters (e.g., rho).

        Returns:
            Generalization error value.
        """
        pass

    def get_order_param_names(self) -> list[str]:
        """
        Return names of order parameters.

        Default is ["m", "q"] for linear models.
        """
        return ["m", "q"]

    def get_default_init(self) -> tuple[float, ...]:
        """
        Return default initial values for order parameters.

        Good initialization is important for convergence.
        """
        n_params = len(self.get_order_param_names())
        return tuple([0.5] * n_params)  # Moderate starting point

    def get_scenario_info(self) -> dict[str, Any]:
        """Return scenario information for documentation."""
        return {
            "class": self.__class__.__name__,
            "order_params": self.get_order_param_names(),
            "params": self.params,
        }

    def is_physical(self, *order_params: float, **kwargs: Any) -> bool:
        """
        Check if order parameters satisfy physical constraints.

        Physical constraints:
        - q >= 0 (self-overlap is non-negative)
        - m² <= q * rho (Cauchy-Schwarz inequality)

        Args:
            *order_params: Order parameter values
            **kwargs: Additional parameters

        Returns:
            True if physical, False otherwise
        """
        m, q = order_params[:2]
        rho = kwargs.get("rho", self.params.get("rho", 1.0))

        if q < 0:
            return False
        if m**2 > q * rho * (1 + 1e-6):  # Small tolerance
            return False
        return True
