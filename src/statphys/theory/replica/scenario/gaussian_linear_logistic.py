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

Warning:
    This implementation uses a heuristic gradient-flow relaxation of the
    regularized ERM stationarity conditions, not the exact RS saddle-point
    equations (which require proximal operators, cf. Salehi et al. 2019).
    Results are qualitatively correct but not quantitatively exact.

"""

from typing import Any

from statphys.theory.replica.scenario.gradient_flow import GradientFlowEquations
from statphys.utils.special_functions import classification_error_linear, sigmoid


class GaussianLinearLogisticEquations(GradientFlowEquations):
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

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    Args:
        rho: Teacher norm (||W₀||²/d). Default 1.0.
        reg_param: L2 regularization parameter λ. Default 0.01.
        damping: Step size of the internal gradient-flow relaxation.
        int_bound: Half-width of the Gaussian integration domain.

    """

    def _gradient_moments(self, u: float, z: float) -> tuple[float, float]:
        """
        Logistic gradient moments for a deterministic sign teacher.

        g(y, z) = y · σ(-y·z) with y = sign(u).
        """
        y = 1.0 if u >= 0 else -1.0
        g = y * sigmoid(-y * z)
        return g, g * g

    def generalization_error(self, m: float, q: float, **kwargs: Any) -> float:
        """Classification error P(error) = (1/π) arccos(m/√(qρ))."""
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)
