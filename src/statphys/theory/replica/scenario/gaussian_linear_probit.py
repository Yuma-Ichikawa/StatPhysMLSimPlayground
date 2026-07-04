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

Warning:
    This implementation uses a heuristic gradient-flow relaxation of the
    regularized ERM stationarity conditions, not the exact RS saddle-point
    equations. Results are qualitatively correct but not quantitatively exact.

"""

from typing import Any

import numpy as np

from statphys.theory.replica.scenario.gradient_flow import GradientFlowEquations
from statphys.utils.constants import EPS_DIV
from statphys.utils.special_functions import (
    classification_error_linear,
    gaussian_cdf,
    gaussian_pdf,
)


class GaussianLinearProbitEquations(GradientFlowEquations):
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

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    Args:
        rho: Teacher norm (||W₀||²/d). Default 1.0.
        reg_param: L2 regularization λ. Default 0.01.
        damping: Step size of the internal gradient-flow relaxation.
        int_bound: Half-width of the Gaussian integration domain.

    """

    @staticmethod
    def _probit_gradient(y: float, z: float) -> float:
        """g(y, z) = -y · φ(yz) / Φ(yz), the derivative of -log Φ(yz)."""
        yz = y * z
        Phi_val = np.clip(gaussian_cdf(yz), EPS_DIV, 1 - EPS_DIV)
        return -y * gaussian_pdf(yz) / Phi_val

    def _gradient_moments(self, u: float, z: float) -> tuple[float, float]:
        """
        Gradient moments averaged over the probit label distribution.

        P(y=1|u) = Φ(u), P(y=-1|u) = 1 - Φ(u).
        """
        p_plus = gaussian_cdf(u)
        p_minus = 1 - p_plus
        g_plus = self._probit_gradient(1.0, z)
        g_minus = self._probit_gradient(-1.0, z)
        e_g = p_plus * g_plus + p_minus * g_minus
        e_g2 = p_plus * g_plus**2 + p_minus * g_minus**2
        return e_g, e_g2

    def generalization_error(self, m: float, q: float, **kwargs: Any) -> float:
        """Classification error P(error) = (1/π) arccos(m/√(qρ))."""
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)
