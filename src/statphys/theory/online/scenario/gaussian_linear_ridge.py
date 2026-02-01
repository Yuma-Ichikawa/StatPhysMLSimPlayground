"""
Scenario: Gaussian Data × Linear Model × Ridge Loss (Online)

ODE equations for online ridge regression.
This is an alias for gaussian_linear_mse with explicit ridge naming.

Data: x ~ N(0, I_d), y = (1/√d) W₀ᵀ x + ε
Model: Linear regression f(x) = wᵀx/√d
Loss: Ridge (MSE + L2 regularization)

References:
    - Engel, Van den Broeck (2001). Statistical Mechanics of Learning.
"""

from typing import Any

from statphys.theory.online.scenario.gaussian_linear_mse import GaussianLinearMseEquations


class GaussianLinearRidgeEquations(GaussianLinearMseEquations):
    """
    ODE equations for online ridge regression.

    This is an alias for GaussianLinearMseEquations with explicit ridge naming.
    The dynamics are identical - ridge regularization appears as λ in
    the ODE equations.

    Usage:
        equations = GaussianLinearRidgeEquations(rho=1.0, eta_noise=0.1, lr=0.5, reg_param=0.1)
    """

    pass
