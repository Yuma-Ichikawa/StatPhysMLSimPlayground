"""Saddle-point equations for logistic regression."""

from typing import Any

import numpy as np
from scipy.integrate import quad

from statphys.theory.replica.models.base import ReplicaEquations


class LogisticRegressionEquations(ReplicaEquations):
    """
    Saddle-point equations for logistic regression.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)

    Logistic loss:
        ℓ(y, z) = log(1 + exp(-yz))

    Student minimizes:
        L = (1/n) Σᵢ ℓ(yᵢ, wᵀxᵢ/√d) + (λ/2)||w||²

    The saddle-point equations involve expectations over the
    joint Gaussian distribution of teacher and student fields.

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    References:
        - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics
          of support vector networks." Phys. Rev. Lett.
        - Salehi et al. (2019). "The impact of regularization on
          high-dimensional logistic regression." NeurIPS
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize LogisticRegressionEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            reg_param: L2 regularization parameter λ. Default 0.01.
        """
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q for logistic regression.

        Uses numerical integration over effective field distribution.

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
        q = max(q, 0.001)
        m = np.clip(m, -np.sqrt(q * rho) * 0.999, np.sqrt(q * rho) * 0.999)

        # Effective field variance
        scale = alpha / (1 + lam)

        def integrand_m(z):
            """Integrand for m update."""
            # Teacher determines label
            y_teacher = np.sign(np.sqrt(rho) + 0.1 * z)
            # Student field
            field = m / np.sqrt(q + 0.001) * np.sqrt(rho) + np.sqrt(q) * z
            # Logistic gradient
            grad = y_teacher * self._sigmoid(-y_teacher * field)
            gaussian = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return grad * np.sqrt(rho) * gaussian

        def integrand_q(z):
            """Integrand for q update."""
            y_teacher = np.sign(np.sqrt(rho) + 0.1 * z)
            field = m / np.sqrt(q + 0.001) * np.sqrt(rho) + np.sqrt(q) * z
            grad = y_teacher * self._sigmoid(-y_teacher * field)
            gaussian = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return grad**2 * gaussian

        # Numerical integration
        dm, _ = quad(integrand_m, -5, 5)
        dq, _ = quad(integrand_q, -5, 5)

        # Damped update
        lr = 0.1
        new_m = m + lr * (scale * dm - lam * m)
        new_q = q + lr * (scale * dq - lam * q)

        # Physical constraints
        new_m = max(new_m, 0.001)
        new_q = max(new_q, 0.001)

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

        This is the probability of misclassifying a new sample.

        Args:
            m: Teacher-student overlap
            q: Self-overlap
            **kwargs: Can override rho

        Returns:
            Classification error probability
        """
        rho = kwargs.get("rho", self.rho)
        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5
