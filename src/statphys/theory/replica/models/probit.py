"""Saddle-point equations for probit regression."""

from typing import Any

import numpy as np
from scipy.special import erf

from statphys.theory.replica.models.base import ReplicaEquations


class ProbitEquations(ReplicaEquations):
    """
    Saddle-point equations for probit regression.

    Teacher generates labels via probit model:
        P(y=1|x) = Φ((1/√d) W₀ᵀ x)

    where Φ is the Gaussian CDF.

    Probit loss:
        ℓ(y, z) = -log Φ(yz)

    Student minimizes:
        L = (1/n) Σᵢ ℓ(yᵢ, wᵀxᵢ/√d) + (λ/2)||w||²

    The probit model is analytically convenient because
    Gaussian integrals have closed-form solutions due to
    the Gaussian structure of the loss.

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    References:
        - Engel, Van den Broeck (2001). Statistical Mechanics of Learning.
        - Opper, Kinzel (1996). "Statistical mechanics of generalization."
          Physics of Neural Networks III
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize ProbitEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            reg_param: L2 regularization λ. Default 0.01.
        """
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param

    def _Phi(self, x: float) -> float:
        """Gaussian CDF: Φ(x) = P(Z ≤ x) for Z ~ N(0,1)."""
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def _G(self, x: float) -> float:
        """Gaussian PDF: G(x) = (1/√(2π)) exp(-x²/2)."""
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q for probit model.

        For probit, the analysis simplifies due to Gaussian structure.

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
        q = max(q, 1e-10)

        # Variance components
        V_teacher = rho  # Teacher signal variance
        V_student = q  # Student self-overlap

        # Correlation coefficient
        corr = m / np.sqrt(V_teacher * V_student) if V_teacher > 0 and V_student > 0 else 0.0
        corr = np.clip(corr, -0.999, 0.999)

        # Scale factor from sample complexity and regularization
        scale = alpha / (1 + lam)

        # Approximate updates (simplified from full integral equations)
        # These capture the essential behavior but may not be exact
        new_m = m + 0.1 * scale * (np.sqrt(rho) * corr * (1 - corr**2) - lam * m)
        new_q = q + 0.1 * scale * ((1 - corr**2) - lam * q)

        # Physical constraints
        new_m = max(new_m, 1e-10)
        new_q = max(new_q, 1e-10)

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
