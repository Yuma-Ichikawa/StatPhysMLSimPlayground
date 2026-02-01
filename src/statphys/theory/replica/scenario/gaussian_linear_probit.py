"""
Scenario: Gaussian Data × Linear Model × Probit Loss

Saddle-point equations for probit regression.

Data: x ~ N(0, I_d), P(y=1|x) = Φ((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Probit loss ℓ(y, z) = -log Φ(yz)

References:
    - Engel, Van den Broeck (2001). Statistical Mechanics of Learning.
    - Opper, Kinzel (1996). "Statistical mechanics of generalization."
      Physics of Neural Networks III
"""

from typing import Any

import numpy as np
from scipy.integrate import dblquad
from scipy.special import erf

from statphys.theory.replica.scenario.base import ReplicaEquations


class GaussianLinearProbitEquations(ReplicaEquations):
    """
    Saddle-point equations for probit regression.

    Teacher generates labels via probit model:
        P(y=1|x) = Φ((1/√d) W₀ᵀ x)

    where Φ is the Gaussian CDF.

    Probit loss:
        ℓ(y, z) = -log Φ(yz)

    Probit gradient:
        ∂ℓ/∂z = -y · φ(yz) / Φ(yz)

    where φ is the Gaussian PDF.

    The saddle-point equations involve expectations over the joint
    Gaussian distribution of teacher and student fields, weighted
    by the label probabilities.

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize GaussianLinearProbitEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            reg_param: L2 regularization λ. Default 0.01.
        """
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param

    def _Phi(self, x: float | np.ndarray) -> float | np.ndarray:
        """Gaussian CDF: Φ(x) = P(Z ≤ x) for Z ~ N(0,1)."""
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def _phi(self, x: float | np.ndarray) -> float | np.ndarray:
        """Gaussian PDF: φ(x) = (1/√(2π)) exp(-x²/2)."""
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def _probit_gradient(self, y: float, z: float) -> float:
        """
        Compute probit loss gradient.

        g(y, z) = -y · φ(yz) / Φ(yz)

        This is the derivative of -log Φ(yz) w.r.t. z.
        """
        yz = y * z
        phi_val = self._phi(yz)
        Phi_val = self._Phi(yz)
        # Clip Φ to avoid division by zero
        Phi_val = np.clip(Phi_val, 1e-10, 1 - 1e-10)
        return -y * phi_val / Phi_val

    def _compute_expectations(
        self,
        m: float,
        q: float,
        rho: float,
    ) -> tuple[float, float]:
        """
        Compute expectations E[g·u/√ρ] and E[g²] for probit loss.

        The label y is sampled according to P(y=1|u) = Φ(u).

        Args:
            m: Teacher-student overlap
            q: Student self-overlap
            rho: Teacher norm

        Returns:
            (E[g·u/√ρ], E[g²])
        """
        q = max(q, 1e-6)
        rho = max(rho, 1e-6)

        max_m = np.sqrt(q * rho) * 0.999
        m = np.clip(m, -max_m, max_m)

        corr = m / np.sqrt(q * rho)
        std_perp = np.sqrt(max(1 - corr**2, 1e-10))

        def integrand_E_gu(xi2, xi1):
            """E[g·u/√ρ] integrand."""
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)

            # P(y=1|u) = Φ(u), P(y=-1|u) = 1 - Φ(u)
            p_plus = self._Phi(u)
            p_minus = 1 - p_plus

            # Expected gradient
            g_plus = self._probit_gradient(1.0, z)
            g_minus = self._probit_gradient(-1.0, z)
            E_g = p_plus * g_plus + p_minus * g_minus

            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return E_g * xi1 * gauss  # xi1 = u/√ρ

        def integrand_E_g2(xi2, xi1):
            """E[g²] integrand."""
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)

            p_plus = self._Phi(u)
            p_minus = 1 - p_plus

            g_plus = self._probit_gradient(1.0, z)
            g_minus = self._probit_gradient(-1.0, z)
            E_g2 = p_plus * g_plus**2 + p_minus * g_minus**2

            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return E_g2 * gauss

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
        Compute updated m and q for probit model.

        Uses 2D Gaussian integration over joint (u, z) distribution,
        with label probabilities weighted by the probit model.

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

        q = max(q, 1e-6)

        E_gu, E_g2 = self._compute_expectations(m, q, rho)

        scale = alpha / (1 + lam)
        lr = 0.1

        # Gradient-based update
        new_m = m + lr * (scale * np.sqrt(rho) * E_gu - lam * m)
        new_q = q + lr * (scale * E_g2 - lam * q)

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
        Compute classification error for probit.

        P(error) = (1/π) arccos(m/√(qρ))

        Args:
            m: Teacher-student overlap
            q: Self-overlap
            **kwargs: Can override rho

        Returns:
            Classification error probability
        """
        rho = kwargs.get("rho", self.rho)
        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5
