"""
Saddle-point equations for replica calculations.

Scenarios are organized by: Data × Model × Loss

Naming convention: {data}_{model}_{loss}.py
- data: Input distribution (gaussian, sparse, structured)
- model: Student architecture (linear, committee, twolayer)
- loss: Loss function (ridge, lasso, logistic, hinge, probit, mse)

Available scenarios:
- gaussian_linear_ridge: Gaussian data, linear model, ridge regression
- gaussian_linear_lasso: Gaussian data, linear model, LASSO regression
- gaussian_linear_logistic: Gaussian data, linear model, logistic loss
- gaussian_linear_hinge: Gaussian data, linear model, hinge/perceptron loss
- gaussian_linear_probit: Gaussian data, linear model, probit loss
- gaussian_committee_mse: Gaussian data, committee machine, MSE loss
"""

from statphys.theory.replica.scenario.base import ReplicaEquations
from statphys.theory.replica.scenario.gaussian_committee_mse import (
    GaussianCommitteeMseEquations,
)
from statphys.theory.replica.scenario.gaussian_linear_hinge import (
    GaussianLinearHingeEquations,
)
from statphys.theory.replica.scenario.gaussian_linear_lasso import (
    GaussianLinearLassoEquations,
)
from statphys.theory.replica.scenario.gaussian_linear_logistic import (
    GaussianLinearLogisticEquations,
)
from statphys.theory.replica.scenario.gaussian_linear_probit import (
    GaussianLinearProbitEquations,
)
from statphys.theory.replica.scenario.gaussian_linear_ridge import (
    GaussianLinearRidgeEquations,
)

__all__ = [
    # Base class
    "ReplicaEquations",
    # Gaussian + Linear scenarios
    "GaussianLinearRidgeEquations",
    "GaussianLinearLassoEquations",
    "GaussianLinearLogisticEquations",
    "GaussianLinearHingeEquations",
    "GaussianLinearProbitEquations",
    # Gaussian + Committee scenarios
    "GaussianCommitteeMseEquations",
]

# Registry for easy access by name
REPLICA_SCENARIOS = {
    "gaussian_linear_ridge": GaussianLinearRidgeEquations,
    "gaussian_linear_lasso": GaussianLinearLassoEquations,
    "gaussian_linear_logistic": GaussianLinearLogisticEquations,
    "gaussian_linear_hinge": GaussianLinearHingeEquations,
    "gaussian_linear_probit": GaussianLinearProbitEquations,
    "gaussian_committee_mse": GaussianCommitteeMseEquations,
}


def get_replica_equations(name: str, **kwargs) -> ReplicaEquations:
    """
    Get replica equations by scenario name.

    Args:
        name: Scenario name (e.g., 'gaussian_linear_ridge')
        **kwargs: Parameters for the equations

    Returns:
        ReplicaEquations instance

    Example:
        >>> equations = get_replica_equations("gaussian_linear_ridge", rho=1.0, reg_param=0.01)
    """
    if name not in REPLICA_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(REPLICA_SCENARIOS.keys())}")
    return REPLICA_SCENARIOS[name](**kwargs)
