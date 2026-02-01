"""
ODE equations for online learning dynamics.

Scenarios are organized by: Data × Model × Loss

Naming convention: {data}_{model}_{loss}.py
- data: Input distribution (gaussian, sparse, structured)
- model: Student architecture (linear, committee, twolayer)
- loss: Loss function (mse, ridge, logistic, hinge, perceptron)

Available scenarios:
- gaussian_linear_mse: Gaussian data, linear model, MSE/SGD
- gaussian_linear_ridge: Gaussian data, linear model, ridge regression
- gaussian_linear_perceptron: Gaussian data, linear model, perceptron loss
- gaussian_linear_logistic: Gaussian data, linear model, logistic loss
- gaussian_linear_hinge: Gaussian data, linear model, hinge/SVM loss
- gaussian_committee_mse: Gaussian data, committee machine, MSE loss
"""

from statphys.theory.online.scenario.base import OnlineEquations
from statphys.theory.online.scenario.gaussian_committee_mse import (
    GaussianCommitteeMseEquations,
)
from statphys.theory.online.scenario.gaussian_linear_hinge import (
    GaussianLinearHingeEquations,
)
from statphys.theory.online.scenario.gaussian_linear_logistic import (
    GaussianLinearLogisticEquations,
)
from statphys.theory.online.scenario.gaussian_linear_mse import (
    GaussianLinearMseEquations,
)
from statphys.theory.online.scenario.gaussian_linear_perceptron import (
    GaussianLinearPerceptronEquations,
)
from statphys.theory.online.scenario.gaussian_linear_ridge import (
    GaussianLinearRidgeEquations,
)

__all__ = [
    # Base class
    "OnlineEquations",
    # Gaussian + Linear scenarios
    "GaussianLinearMseEquations",
    "GaussianLinearRidgeEquations",
    "GaussianLinearPerceptronEquations",
    "GaussianLinearLogisticEquations",
    "GaussianLinearHingeEquations",
    # Gaussian + Committee scenarios
    "GaussianCommitteeMseEquations",
]

# Registry for easy access by name
ONLINE_SCENARIOS = {
    "gaussian_linear_mse": GaussianLinearMseEquations,
    "gaussian_linear_ridge": GaussianLinearRidgeEquations,
    "gaussian_linear_perceptron": GaussianLinearPerceptronEquations,
    "gaussian_linear_logistic": GaussianLinearLogisticEquations,
    "gaussian_linear_hinge": GaussianLinearHingeEquations,
    "gaussian_committee_mse": GaussianCommitteeMseEquations,
}


def get_online_equations(name: str, **kwargs) -> OnlineEquations:
    """
    Get online equations by scenario name.

    Args:
        name: Scenario name (e.g., 'gaussian_linear_mse')
        **kwargs: Parameters for the equations

    Returns:
        OnlineEquations instance

    Example:
        >>> equations = get_online_equations("gaussian_linear_mse", rho=1.0, lr=0.1)
    """
    if name not in ONLINE_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(ONLINE_SCENARIOS.keys())}")
    return ONLINE_SCENARIOS[name](**kwargs)
