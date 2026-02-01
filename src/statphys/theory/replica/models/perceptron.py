"""Saddle-point equations for perceptron/SVM learning."""

from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from statphys.theory.replica.models.base import ReplicaEquations


class PerceptronEquations(ReplicaEquations):
    """
    Saddle-point equations for perceptron learning.

    Teacher-student setup with sign teacher:
        y = sign((1/√d) W₀ᵀ x)

    Can analyze various loss functions:
    - Perceptron loss: max(0, -yz)
    - Hinge loss (SVM): max(0, κ - yz)
    - Zero-one loss (for Gardner volume)

    Order parameters:
        m = wᵀW₀/d : teacher-student overlap
        q = ||w||²/d : self-overlap

    Gardner volume condition (for pattern storage):
        yᵢ · (wᵀxᵢ)/(√d ||w||) ≥ κ  for all i

    Critical capacity αc depends on the margin κ:
        αc(κ=0) = 2      (Perceptron)
        αc(κ>0) < 2      (SVM with margin)

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    References:
        - Gardner (1988). "The space of interactions in neural
          network models." J. Phys. A 21, 257
        - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics
          of support vector networks." Phys. Rev. Lett. 82, 2975

    """

    def __init__(
        self,
        rho: float = 1.0,
        margin: float = 0.0,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize PerceptronEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            margin: Margin parameter κ. Default 0.0 (perceptron).
                   κ > 0 for SVM with margin.
            reg_param: L2 regularization λ. Default 0.0.

        """
        super().__init__(rho=rho, margin=margin, reg_param=reg_param, **params)
        self.rho = rho
        self.margin = margin
        self.reg_param = reg_param

    def _H(self, x: float) -> float:
        """Gaussian tail function: H(x) = P(Z > x) for Z ~ N(0,1)."""
        return 0.5 * (1 - erf(x / np.sqrt(2)))

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
        Compute updated m and q for perceptron/SVM.

        Args:
            m: Current teacher-student overlap
            q: Current self-overlap
            alpha: Sample ratio n/d
            **kwargs: Can override rho, margin, reg_param

        Returns:
            (m_new, q_new) updated order parameters

        """
        rho = kwargs.get("rho", self.rho)
        kappa = kwargs.get("margin", self.margin)
        lam = kwargs.get("reg_param", self.reg_param)

        # Ensure numerical stability
        q = max(q, 1e-10)

        # Correlation between teacher and student fields
        rho_corr = m / np.sqrt(rho * q) if rho > 0 else 0.0
        rho_corr = np.clip(rho_corr, -0.999, 0.999)

        # Conditional std of student field given teacher label
        Delta = np.sqrt(q * (1 - rho_corr**2) + 1e-10)

        # Integrands for update equations
        def integrand_m(z):
            """Integrand for m update."""
            h = m / np.sqrt(q + 1e-10) + Delta * z
            # Contribution from margin-violating samples
            if kappa > 0:
                indicator = 1.0 if h < kappa else 0.0
            else:
                indicator = 1.0 if h < 0 else 0.0
            return indicator * (kappa - h) * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        def integrand_q(z):
            """Integrand for q update."""
            h = m / np.sqrt(q + 1e-10) + Delta * z
            indicator = (1.0 if h < kappa else 0.0) if kappa > 0 else (1.0 if h < 0 else 0.0)
            return indicator * (kappa - h) ** 2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        # Numerical integration
        dm_contrib, _ = quad(integrand_m, -10, 10)
        dq_contrib, _ = quad(integrand_q, -10, 10)

        # Damped update
        lr = 0.1
        new_m = m + lr * (alpha * np.sqrt(rho) * dm_contrib - lam * m)
        new_q = q + lr * (alpha * dq_contrib - lam * q)

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
        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5

    def critical_capacity(self, margin: float = None) -> float:
        """
        Compute critical storage capacity α_c.

        For the spherical perceptron with margin κ:
            α_c(κ) depends on the margin

        At κ = 0: α_c = 2 (Gardner bound)

        Args:
            margin: Margin parameter (uses self.margin if None)

        Returns:
            Critical capacity α_c

        """
        kappa = margin if margin is not None else self.margin

        if kappa <= 0:
            return 2.0  # Gardner bound

        # Approximate formula for α_c(κ) > 0
        # This is a rough estimate; exact value requires solving integral equations
        return 2.0 / (1 + kappa**2)
