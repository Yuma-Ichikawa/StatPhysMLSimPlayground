"""
Replica method module.

Provides solvers for saddle-point equations arising from
the replica trick in statistical mechanics.
"""

from statphys.theory.replica.solver import SaddlePointSolver
from statphys.theory.replica.equations import (
    ReplicaEquations,
    RidgeRegressionEquations,
    LassoEquations,
    LogisticRegressionEquations,
    PerceptronEquations,
    ProbitEquations,
    CommitteeMachineEquations,
)
from statphys.theory.replica.integration import (
    gaussian_integral,
    double_gaussian_integral,
    moreau_envelope,
    proximal_operator,
)

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
