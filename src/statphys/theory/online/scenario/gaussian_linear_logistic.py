"""
Scenario: Gaussian Data × Linear Model × Logistic Loss (Online)

ODE equations for online logistic regression.

Data: x ~ N(0, I_d), y = sign((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Logistic loss ℓ(y, z) = log(1 + exp(-yz))

References:
    - Dietrich, Opper, Sompolinsky (1999). Phys. Rev. Lett.
    - Engel, Van den Broeck (2001). Statistical Mechanics of Learning.
"""

from typing import Any

import numpy as np
from scipy.integrate import dblquad

from statphys.theory.online.scenario.base import OnlineEquations
from statphys.utils.special_functions import classification_error_linear, sigmoid


class GaussianLinearLogisticEquations(OnlineEquations):
    """
    ODE equations for online logistic regression.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)  (deterministic teacher)

    Logistic loss: ℓ(y, z) = log(1 + exp(-yz))
    Logistic gradient: g(y, z) = y · σ(-yz) where σ is sigmoid

    SGD update:
        w^{τ+1} = w^τ - η ∇ℓ = w^τ + η g(y, z) x

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η √ρ · E[g(y,z) · u/√ρ] - ηλm
        dq/dt = 2η √q · E[g(y,z) · z/√q] + η² E[g²] - 2ηλq

    where expectations are over the joint Gaussian (u, z) with:
        - u = W₀ᵀx/√d ~ N(0, ρ)  (teacher field)
        - z = wᵀx/√d ~ N(0, q)   (student field)
        - Cov(u, z) = m

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 0.1,
        reg_param: float = 0.0,
        use_quadrature: bool = True,
        **params: Any,
    ):
        """
        Initialize GaussianLinearLogisticEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            lr: Learning rate η. Default 0.1.
            reg_param: L2 regularization λ. Default 0.0.
            use_quadrature: If True, use numerical quadrature (deterministic).
                           If False, use Monte Carlo (for debugging).
        """
        super().__init__(
            rho=rho, lr=lr, reg_param=reg_param, use_quadrature=use_quadrature, **params
        )
        self.rho = rho
        self.lr = lr
        self.reg_param = reg_param
        self.use_quadrature = use_quadrature

    def _compute_expectations_quadrature(
        self,
        m: float,
        q: float,
        rho: float,
    ) -> tuple[float, float, float]:
        """
        Compute expectations using numerical quadrature (deterministic).

        Returns:
            (E[g·u/√ρ], E[g·z/√q], E[g²])
        """
        # Ensure numerical stability
        q = max(q, 1e-6)
        rho = max(rho, 1e-6)

        # Clip m to valid range
        max_m = np.sqrt(q * rho) * 0.999
        m = np.clip(m, -max_m, max_m)

        # Correlation coefficient
        corr = m / np.sqrt(q * rho)
        std_perp = np.sqrt(max(1 - corr**2, 1e-10))

        def make_integrand(func_type):
            """Create integrand function for different expectation types."""

            def integrand(xi2, xi1):
                u = np.sqrt(rho) * xi1
                z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)
                y = np.sign(u) if u != 0 else 1.0
                g = y * sigmoid(-y * z)
                gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)

                if func_type == "gu":
                    return g * xi1 * gauss  # g · u/√ρ
                elif func_type == "gz":
                    z_normalized = corr * xi1 + std_perp * xi2  # z/√q
                    return g * z_normalized * gauss  # g · z/√q
                else:  # g2
                    return g**2 * gauss

            return integrand

        # Compute expectations via 2D integration
        E_gu, _ = dblquad(make_integrand("gu"), -5, 5, -5, 5)
        E_gz, _ = dblquad(make_integrand("gz"), -5, 5, -5, 5)
        E_g2, _ = dblquad(make_integrand("g2"), -5, 5, -5, 5)

        return E_gu, E_gz, E_g2

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online logistic regression.

        Uses numerical quadrature for deterministic ODE evaluation.

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, lr, reg_param

        Returns:
            [dm/dt, dq/dt]
        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        lam = params.get("reg_param", self.reg_param)

        # Ensure numerical stability
        q = max(q, 1e-6)

        # Compute expectations using quadrature
        E_gu, E_gz, E_g2 = self._compute_expectations_quadrature(m, q, rho)

        # ODE equations:
        # dm/dt = η √ρ · E[g·u/√ρ] - ηλm
        # dq/dt = 2η √q · E[g·z/√q] + η² E[g²] - 2ηλq
        dm_dt = lr * np.sqrt(rho) * E_gu - lr * lam * m
        dq_dt = 2 * lr * np.sqrt(q) * E_gz + lr**2 * E_g2 - 2 * lr * lam * q

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) arccos(m/√(qρ))

        Args:
            y: [m, q] order parameters
            **kwargs: Can override rho

        Returns:
            Classification error probability
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)
