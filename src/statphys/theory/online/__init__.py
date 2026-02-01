"""
Online learning dynamics module.

Provides ODE solvers for learning dynamics in the high-dimensional limit.
Time is normalized as t = τ/d where τ is the number of samples seen.

Scenarios (organized by data × model × loss):
---------------------------------------------
- GaussianLinearMseEquations (alias: OnlineSGDEquations): Online SGD for linear regression
- GaussianLinearRidgeEquations (alias: OnlineRidgeEquations): Online ridge regression
- GaussianLinearPerceptronEquations (alias: OnlinePerceptronEquations): Online perceptron learning
- GaussianLinearLogisticEquations (alias: OnlineLogisticEquations): Online logistic regression
- GaussianLinearHingeEquations (alias: OnlineHingeEquations): Online SVM/hinge loss learning
- GaussianCommitteeMseEquations (alias: OnlineCommitteeEquations): Online committee machine

Example:
-------
>>> from statphys.theory.online import ODESolver, OnlineSGDEquations
>>> equations = OnlineSGDEquations(rho=1.0, lr=0.5, eta_noise=0.1)
>>> solver = ODESolver(equations=equations, order_params=['m', 'q'])
>>> result = solver.solve(t_span=(0, 10), init_values=(0.0, 0.01))

"""

# Scenario module
from statphys.theory.online.scenario import (
    ONLINE_SCENARIOS,
    GaussianCommitteeMseEquations,
    GaussianLinearHingeEquations,
    GaussianLinearLogisticEquations,
    GaussianLinearMseEquations,
    GaussianLinearPerceptronEquations,
    GaussianLinearRidgeEquations,
    OnlineEquations,
    get_online_equations,
)

# Solver
from statphys.theory.online.solver import AdaptiveODESolver, ODESolver

# Convenience aliases (for shorter, more intuitive names)
OnlineSGDEquations = GaussianLinearMseEquations
OnlineRidgeEquations = GaussianLinearRidgeEquations
OnlinePerceptronEquations = GaussianLinearPerceptronEquations
OnlineLogisticEquations = GaussianLinearLogisticEquations
OnlineHingeEquations = GaussianLinearHingeEquations
OnlineCommitteeEquations = GaussianCommitteeMseEquations

__all__ = [
    # Solvers
    "ODESolver",
    "AdaptiveODESolver",
    # Base class
    "OnlineEquations",
    # Scenarios (full names)
    "GaussianLinearMseEquations",
    "GaussianLinearRidgeEquations",
    "GaussianLinearPerceptronEquations",
    "GaussianLinearLogisticEquations",
    "GaussianLinearHingeEquations",
    "GaussianCommitteeMseEquations",
    # Scenarios (short aliases for convenience)
    "OnlineSGDEquations",
    "OnlineRidgeEquations",
    "OnlinePerceptronEquations",
    "OnlineLogisticEquations",
    "OnlineHingeEquations",
    "OnlineCommitteeEquations",
    # Scenario utilities
    "get_online_equations",
    "ONLINE_SCENARIOS",
]
