"""
Scenario: Gaussian Data × Linear Model × Logistic Loss

Saddle-point equations for logistic regression.

Data: x ~ N(0, I_d), y = sign((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Logistic loss ℓ(y, z) = log(1 + exp(-yz))

References:
    - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics
      of support vector networks." Phys. Rev. Lett.
    - Salehi et al. (2019). "The impact of regularization on
      high-dimensional logistic regression." NeurIPS
"""

from typing import Any

import numpy as np
from scipy.integrate import dblquad

from statphys.theory.replica.scenario.base import ReplicaEquations
from statphys.utils.special_functions import classification_error_linear, sigmoid


class GaussianLinearLogisticEquations(ReplicaEquations):
    """
    Saddle-point equations for logistic regression.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)

    Logistic loss:
        ℓ(y, z) = log(1 + exp(-yz))

    Student minimizes:
        L = (1/n) Σᵢ ℓ(yᵢ, wᵀxᵢ/√d) + (λ/2)||w||²

    The saddle-point equations involve expectations over the
    joint Gaussian distribution of teacher and student fields:
        (u, z) ~ N(0, [[ρ, m], [m, q]])

    where:
        - u = W₀ᵀx/√d is the teacher field
        - z = wᵀx/√d is the student field

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        n_quad: int = 50,
        **params: Any,
    ):
        """
        Initialize GaussianLinearLogisticEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            reg_param: L2 regularization parameter λ. Default 0.01.
            n_quad: Number of quadrature points for numerical integration.
        """
        super().__init__(rho=rho, reg_param=reg_param, n_quad=n_quad, **params)
        self.rho = rho
        self.reg_param = reg_param
        self.n_quad = n_quad

    def _compute_joint_expectations(
        self,
        m: float,
        q: float,
        rho: float,
    ) -> tuple[float, float]:
        """
        Compute expectations E[g·u] and E[g²] over joint Gaussian (u, z).

        The joint distribution is:
            (u, z) ~ N(0, [[ρ, m], [m, q]])

        The logistic gradient is:
            g(y, z) = y · σ(-y·z) where y = sign(u)

        Args:
            m: Teacher-student overlap
            q: Student self-overlap
            rho: Teacher norm

        Returns:
            (E[g·u/√ρ], E[g²]) expectations needed for saddle-point equations
        """
        # Ensure numerical stability
        q = max(q, 1e-6)
        rho = max(rho, 1e-6)

        # Clip m to valid range (Cauchy-Schwarz: m² ≤ q·ρ)
        max_m = np.sqrt(q * rho) * 0.999
        m = np.clip(m, -max_m, max_m)

        # Correlation coefficient
        corr = m / np.sqrt(q * rho)

        # For numerical integration, we use:
        # u = √ρ · ξ₁
        # z = √q · (corr·ξ₁ + √(1-corr²)·ξ₂)
        # where ξ₁, ξ₂ ~ N(0,1) are independent

        std_perp = np.sqrt(max(1 - corr**2, 1e-10))

        def integrand_E_gu(xi2, xi1):
            """Integrand for E[g·u/√ρ]."""
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)
            y = np.sign(u) if u != 0 else 1.0
            g = y * sigmoid(-y * z)
            # Gaussian weight
            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return g * xi1 * gauss  # xi1 = u/√ρ

        def integrand_E_g2(xi2, xi1):
            """Integrand for E[g²]."""
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)
            y = np.sign(u) if u != 0 else 1.0
            g = y * sigmoid(-y * z)
            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return g**2 * gauss

        # Numerical integration over [-5, 5]² (captures > 99.99% of Gaussian mass)
        E_gu, _ = dblquad(integrand_E_gu, -5, 5, -5, 5)
        E_g2, _ = dblquad(integrand_E_g2, -5, 5, -5, 5)

        return E_gu, E_g2

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q for logistic regression.

        Uses 2D Gaussian integration over joint (u, z) distribution.

        Args:
            m: Current teacher-student overlap
            q: Current self-overlap
            alpha: Sample ratio n/d
            **kwargs: Can override rho, reg_param

        Returns:
            (m_new, q_new) updated order parameters
        """
        rho = kwargs.get("rho", self.rho)
        lam = kwargs.get("reg_param", self.reg_param)

        # Ensure numerical stability
        q = max(q, 1e-6)

        # Compute expectations
        E_gu, E_g2 = self._compute_joint_expectations(m, q, rho)

        # Effective scale
        scale = alpha / (1 + lam)

        # Damped fixed-point iteration
        lr = 0.1

        # Gradient-based update for m and q
        # dm/dt ∝ α·√ρ·E[g·u/√ρ] - λ·m
        # dq/dt ∝ α·E[g²] - λ·q
        new_m = m + lr * (scale * np.sqrt(rho) * E_gu - lam * m)
        new_q = q + lr * (scale * E_g2 - lam * q)

        # Physical constraints
        new_m = max(new_m, 1e-6)
        new_q = max(new_q, 1e-6)

        return new_m, new_q

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) arccos(m/√(qρ))

        Args:
            m: Teacher-student overlap
            q: Self-overlap
            **kwargs: Can override rho

        Returns:
            Classification error probability
        """
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)
