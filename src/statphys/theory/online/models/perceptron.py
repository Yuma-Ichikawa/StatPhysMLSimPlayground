"""ODE equations for online perceptron learning."""

from typing import Any

import numpy as np
from scipy.special import erf

from statphys.theory.online.models.base import OnlineEquations


class OnlinePerceptronEquations(OnlineEquations):
    """
    ODE equations for online perceptron learning.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)

    Perceptron update rule:
        w^{τ+1} = w^τ + η y x  (if y ≠ sign(wᵀx))

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η √ρ · φ(κ) / √q
        dq/dt = 2η² ε(κ)

    where:
        - κ = m / √(qρ) : stability parameter (cosine of angle)
        - φ(κ) = (1/√(2π)) exp(-κ²/2) : Gaussian PDF
        - ε(κ) = H(κ) : error rate (complementary Gaussian CDF)
        - H(x) = (1/2)(1 - erf(x/√2)) : Gaussian tail function

    Classification error:
        P(error) = (1/π) arccos(κ) = (1/π) arccos(m/√(qρ))

    References:
        - Opper (1996). "Online versus offline learning from random examples."
          Europhys. Lett.
        - Kinzel, Opper (1991). "Dynamics of learning." Physics of Neural Networks
        - Saad, Solla (1995). Phys. Rev. E

    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 1.0,
        **params: Any,
    ):
        """
        Initialize OnlinePerceptronEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            lr: Learning rate η. Default 1.0 (standard perceptron).

        """
        super().__init__(rho=rho, lr=lr, **params)
        self.rho = rho
        self.lr = lr

    def _H(self, x: float) -> float:
        """
        Complementary Gaussian CDF.

        H(x) = P(Z > x) = (1/2)(1 - erf(x/√2))

        where Z ~ N(0,1).
        """
        return 0.5 * (1 - erf(x / np.sqrt(2)))

    def _phi(self, x: float) -> float:
        """
        Gaussian PDF.

        φ(x) = (1/√(2π)) exp(-x²/2)
        """
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online perceptron.

        ODE system:
            dm/dt = η √ρ · φ(κ) / √q
            dq/dt = 2η² H(κ)

        where κ = m / √(qρ) is the stability parameter.

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, lr

        Returns:
            [dm/dt, dq/dt]

        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)

        # Stability parameter (avoid division by zero)
        kappa = m / np.sqrt(q * rho + 1e-10)
        kappa = np.clip(kappa, -10, 10)

        # Error rate (probability of misclassification)
        epsilon = self._H(kappa)

        # Gaussian density at stability
        phi_kappa = self._phi(kappa)

        # ODE equations (Saad & Solla style)
        dm_dt = lr * np.sqrt(rho) * phi_kappa / np.sqrt(q + 1e-10)
        dq_dt = lr**2 * 2 * epsilon

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) arccos(m / √(qρ))

        This is the fraction of test samples misclassified.

        Args:
            y: [m, q] order parameters
            **kwargs: Can override rho

        Returns:
            Classification error probability

        """
        m, q = y
        rho = kwargs.get("rho", self.rho)

        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5  # Random guess if degenerate

    def stability_parameter(self, y: np.ndarray, rho: float = None) -> float:
        """
        Compute stability parameter κ = m / √(qρ).

        This measures the alignment between student and teacher.
        κ → 1 means perfect alignment, κ = 0 means orthogonal.

        Args:
            y: [m, q] order parameters
            rho: Teacher norm (uses self.rho if not provided)

        Returns:
            Stability parameter κ

        """
        m, q = y
        rho = rho if rho is not None else self.rho
        if q > 0 and rho > 0:
            return m / np.sqrt(q * rho)
        return 0.0
