"""
Replica method module.

Provides solvers for saddle-point equations arising from the replica trick
in statistical mechanics of learning.

Scenarios (organized by data × model × loss):
---------------------------------------------
- GaussianLinearRidgeEquations (alias: RidgeRegressionEquations): Ridge regression
- GaussianLinearLassoEquations (alias: LassoEquations): LASSO regression
- GaussianLinearLogisticEquations (alias: LogisticRegressionEquations): Logistic regression
- GaussianLinearHingeEquations (alias: PerceptronEquations): Perceptron/SVM
- GaussianLinearProbitEquations (alias: ProbitEquations): Probit regression
- GaussianCommitteeMseEquations (alias: CommitteeMachineEquations): Committee machine

Example:
--------
>>> from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations
>>> equations = RidgeRegressionEquations(rho=1.0, eta=0.1, reg_param=0.01)
>>> solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])
>>> result = solver.solve(alpha_values=[0.5, 1.0, 2.0], rho=1.0, eta=0.1)
"""

# Scenario module
from statphys.theory.replica.scenario import (
    REPLICA_SCENARIOS,
    ReplicaEquations,
    get_replica_equations,
    GaussianLinearRidgeEquations,
    GaussianLinearLassoEquations,
    GaussianLinearLogisticEquations,
    GaussianLinearHingeEquations,
    GaussianLinearProbitEquations,
    GaussianCommitteeMseEquations,
)

# Integration utilities (re-exported from utils for convenience)
from statphys.utils.integration import (
    double_gaussian_integral,
    gaussian_integral,
    hard_threshold,
    moreau_envelope,
    proximal_operator,
    soft_threshold,
)

# Solver
from statphys.theory.replica.solver import SaddlePointSolver

# Convenience aliases (for shorter, more intuitive names)
RidgeRegressionEquations = GaussianLinearRidgeEquations
LassoEquations = GaussianLinearLassoEquations
LogisticRegressionEquations = GaussianLinearLogisticEquations
PerceptronEquations = GaussianLinearHingeEquations
ProbitEquations = GaussianLinearProbitEquations
CommitteeMachineEquations = GaussianCommitteeMseEquations

__all__ = [
    # Solver
    "SaddlePointSolver",
    # Base class
    "ReplicaEquations",
    # Scenarios (full names)
    "GaussianLinearRidgeEquations",
    "GaussianLinearLassoEquations",
    "GaussianLinearLogisticEquations",
    "GaussianLinearHingeEquations",
    "GaussianLinearProbitEquations",
    "GaussianCommitteeMseEquations",
    # Scenarios (short aliases for convenience)
    "RidgeRegressionEquations",
    "LassoEquations",
    "LogisticRegressionEquations",
    "PerceptronEquations",
    "ProbitEquations",
    "CommitteeMachineEquations",
    # Integration utilities
    "gaussian_integral",
    "double_gaussian_integral",
    "moreau_envelope",
    "proximal_operator",
    "soft_threshold",
    "hard_threshold",
    # Scenario utilities
    "get_replica_equations",
    "REPLICA_SCENARIOS",
]
