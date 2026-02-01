"""
Replica method module.

Provides solvers for saddle-point equations arising from
the replica trick in statistical mechanics.
"""

from statphys.theory.replica.equations import (
    CommitteeMachineEquations,
    LassoEquations,
    LogisticRegressionEquations,
    PerceptronEquations,
    ProbitEquations,
    ReplicaEquations,
    RidgeRegressionEquations,
)
from statphys.theory.replica.integration import (
    double_gaussian_integral,
    gaussian_integral,
    moreau_envelope,
    proximal_operator,
)
from statphys.theory.replica.solver import SaddlePointSolver

__all__ = [
    "SaddlePointSolver",
    # Base class
    "ReplicaEquations",
    # Regression equations
    "RidgeRegressionEquations",
    "LassoEquations",
    # Classification equations
    "LogisticRegressionEquations",
    "PerceptronEquations",
    "ProbitEquations",
    # Committee machine
    "CommitteeMachineEquations",
    # Integration utilities
    "gaussian_integral",
    "double_gaussian_integral",
    "moreau_envelope",
    "proximal_operator",
]
