"""Base class for online learning ODE equations."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class OnlineEquations(ABC):
    """
    Abstract base class for online learning ODE equations.

    Online learning dynamics are described by ODEs in the high-dimensional limit:
        dm/dt = F_m(m, q, t; params)
        dq/dt = F_q(m, q, t; params)

    where t = τ/d is the normalized time (τ = number of online samples seen).

    Subclasses implement specific dynamics for different learning algorithms
    and model architectures.

    Mathematical formulation:
        - Time scale: t = τ/d where τ is the number of samples processed
        - For each sample τ, we do one SGD step
        - In the d → ∞ limit, the order parameters evolve deterministically

    Usage:
        class MyEquations(OnlineEquations):
            def __call__(self, t, y, params):
                m, q = y
                # Compute dm/dt and dq/dt
                return np.array([dm_dt, dq_dt])

            def generalization_error(self, y, **kwargs):
                m, q = y
                return 0.5 * (rho - 2*m + q)
    """

    def __init__(self, **params: Any):
        """
        Initialize with problem-specific parameters.

        Args:
            **params: Parameters like rho, eta, lr, reg_param, etc.
        """
        self.params = params

    @abstractmethod
    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute ODE right-hand side dy/dt.

        This defines the ODE system:
            dy/dt = F(t, y; params)

        where y = [m, q, ...] are the order parameters.

        Args:
            t: Current time (= number of samples / d). This is the normalized
               time where t = τ/d and τ is the sample index.
            y: Current order parameter values as numpy array.
            params: Additional parameters (can override __init__ params).

        Returns:
            Array of dy/dt values with same shape as y.
        """
        pass

    @abstractmethod
    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error from order parameters.

        For regression (MSE): E_g = (1/2)(ρ - 2m + q)
        For classification: P(error) = (1/π) arccos(m / √(qρ))

        Args:
            y: Order parameter values.
            **kwargs: Additional parameters.

        Returns:
            Generalization error value.
        """
        pass

    def get_order_param_names(self) -> list[str]:
        """
        Return names of order parameters.

        Default is ["m", "q"] for linear models.
        Override for more complex models (e.g., committee machines).
        """
        return ["m", "q"]

    def get_default_init(self) -> tuple[float, ...]:
        """
        Return default initial values for order parameters.

        Default is (0.0, 0.01) meaning zero teacher overlap and small self-overlap.
        """
        n_params = len(self.get_order_param_names())
        if n_params == 2:
            return (0.0, 0.01)  # m=0, q=0.01
        return tuple([0.01] * n_params)

    def get_model_info(self) -> dict[str, Any]:
        """Return model information for documentation."""
        return {
            "class": self.__class__.__name__,
            "order_params": self.get_order_param_names(),
            "params": self.params,
        }
