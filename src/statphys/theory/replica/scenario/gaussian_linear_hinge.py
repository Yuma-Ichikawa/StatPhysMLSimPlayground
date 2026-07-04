"""
Scenario: Gaussian Data × Linear Model × Hinge Loss (SVM/Perceptron)

Saddle-point equations for perceptron/SVM learning.

Data: x ~ N(0, I_d), y = sign((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Hinge loss max(0, κ - yz) or Perceptron loss max(0, -yz)

References:
    - Gardner (1988). "The space of interactions in neural
      network models." J. Phys. A 21, 257
    - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics
      of support vector networks." Phys. Rev. Lett. 82, 2975

Warning:
    This implementation uses a heuristic gradient-flow relaxation of the
    regularized ERM stationarity conditions, not the exact Gardner/RS
    saddle-point equations. Results are qualitatively correct but not
    quantitatively exact.

"""

from typing import Any

import numpy as np
from scipy.integrate import quad

from statphys.theory.replica.scenario.gradient_flow import GradientFlowEquations
from statphys.utils.constants import (
    CORR_CLIP,
    EPS_DIV,
    GAUSS_INT_BOUND_WIDE,
)
from statphys.utils.special_functions import classification_error_linear, gaussian_pdf


class GaussianLinearHingeEquations(GradientFlowEquations):
    """
    Saddle-point equations for perceptron/SVM learning.

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

    Args:
        rho: Teacher norm (||W₀||²/d). Default 1.0.
        margin: Margin parameter κ. Default 0.0 (perceptron);
            κ > 0 for SVM with margin.
        reg_param: L2 regularization λ. Default 0.0.
        damping: Step size of the internal gradient-flow relaxation.

    """

    #: Gardner bound for the spherical perceptron at zero margin
    GARDNER_CAPACITY = 2.0

    def __init__(
        self,
        rho: float = 1.0,
        margin: float = 0.0,
        reg_param: float = 0.0,
        **params: Any,
    ):
        super().__init__(rho=rho, reg_param=reg_param, margin=margin, **params)
        self.margin = margin

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q for perceptron/SVM.

        Uses a 1D conditional-Gaussian integral over margin-violating
        samples followed by the shared damped relaxation step.

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

        q = max(q, EPS_DIV)

        # Correlation between teacher and student fields
        rho_corr = m / np.sqrt(rho * q) if rho > 0 else 0.0
        rho_corr = np.clip(rho_corr, -CORR_CLIP, CORR_CLIP)

        # Conditional std of student field given teacher label
        Delta = np.sqrt(q * (1 - rho_corr**2) + EPS_DIV)
        h_mean = m / np.sqrt(q + EPS_DIV)
        b = GAUSS_INT_BOUND_WIDE

        def hinge_active(h: float) -> bool:
            return h < kappa if kappa > 0 else h < 0

        def integrand_m(z: float) -> float:
            h = h_mean + Delta * z
            if not hinge_active(h):
                return 0.0
            return (kappa - h) * gaussian_pdf(z)

        def integrand_q(z: float) -> float:
            h = h_mean + Delta * z
            if not hinge_active(h):
                return 0.0
            return (kappa - h) ** 2 * gaussian_pdf(z)

        dm_contrib, _ = quad(integrand_m, -b, b)
        dq_contrib, _ = quad(integrand_q, -b, b)

        dm = alpha * np.sqrt(rho) * dm_contrib - lam * m
        dq = alpha * dq_contrib - lam * q
        return self._relax(m, q, dm, dq)

    def generalization_error(self, m: float, q: float, **kwargs: Any) -> float:
        """Classification error P(error) = (1/π) arccos(m/√(qρ))."""
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)

    def critical_capacity(self, margin: float | None = None) -> float:
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
            return self.GARDNER_CAPACITY

        # Approximate formula for α_c(κ) > 0
        # This is a rough estimate; exact value requires solving integral equations
        return self.GARDNER_CAPACITY / (1 + kappa**2)
