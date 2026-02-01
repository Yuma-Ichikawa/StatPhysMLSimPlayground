"""
Scenario: Gaussian Data × Linear Model × Hinge Loss (Online SVM)

ODE equations for online SVM/hinge loss learning.

Data: x ~ N(0, I_d), y = sign((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Hinge loss ℓ(y, z) = max(0, κ - yz)

References:
    - Dietrich, Opper, Sompolinsky (1999). Phys. Rev. Lett.
    - Gardner (1988). J. Phys. A
"""

from typing import Any

import numpy as np
from scipy.special import erf

from statphys.theory.online.scenario.base import OnlineEquations


class GaussianLinearHingeEquations(OnlineEquations):
    """
    ODE equations for online SVM/hinge loss learning.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)

    Hinge loss: ℓ(y, z) = max(0, κ - yz)
    where κ is the margin parameter.

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η √ρ · φ(θ)/Δ - ηλm
        dq/dt = 2η² H(-θ) - 2ηλq

    where:
        - κ is the margin
        - Δ = √(q(1 - ρ_corr²)) : conditional std of student field
        - θ = (κ - effective_mean) / Δ : threshold parameter
        - H(x) = (1/2)(1 - erf(x/√2)) : Gaussian tail
        - φ(x) = (1/√(2π)) exp(-x²/2) : Gaussian PDF

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 0.1,
        margin: float = 1.0,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize GaussianLinearHingeEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            lr: Learning rate η. Default 0.1.
            margin: Hinge loss margin κ. Default 1.0 (standard SVM).
            reg_param: L2 regularization λ. Default 0.0.
        """
        super().__init__(rho=rho, lr=lr, margin=margin, reg_param=reg_param, **params)
        self.rho = rho
        self.lr = lr
        self.margin = margin
        self.reg_param = reg_param

    def _H(self, x: float) -> float:
        """Complementary Gaussian CDF: H(x) = P(Z > x)."""
        return 0.5 * (1 - erf(x / np.sqrt(2)))

    def _phi(self, x: float) -> float:
        """Gaussian PDF: φ(x) = (1/√(2π)) exp(-x²/2)."""
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online hinge loss.

        ODE system:
            dm/dt = η √ρ · φ(θ)/Δ - ηλm
            dq/dt = 2η² H(-θ) - 2ηλq

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, lr, margin, reg_param

        Returns:
            [dm/dt, dq/dt]
        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        kappa = params.get("margin", self.margin)
        lam = params.get("reg_param", self.reg_param)

        # Stability parameter (cosine of angle)
        stability = m / np.sqrt(q * rho + 1e-10)
        stability = np.clip(stability, -10, 10)

        # Conditional standard deviation of student field
        Delta = np.sqrt(q * (1 - stability**2) + 1e-10)

        # Effective threshold for hinge loss margin condition
        threshold = (kappa - m * stability / np.sqrt(q + 1e-10)) / Delta

        # Probability of margin violation
        prob_violation = self._H(-threshold)

        # Gradient contributions (from violated samples)
        phi_threshold = self._phi(threshold)

        # ODE equations
        dm_dt = lr * np.sqrt(rho) * phi_threshold / Delta - lr * lam * m
        dq_dt = lr**2 * 2 * prob_violation - 2 * lr * lam * q

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

        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5
