"""
Online learning dynamics module.

Provides ODE solvers for learning dynamics in the
high-dimensional limit.
"""

from statphys.theory.online.equations import (
    OnlineCommitteeEquations,
    OnlineEquations,
    OnlineHingeEquations,
    OnlineLogisticEquations,
    OnlinePerceptronEquations,
    OnlineRidgeEquations,
    OnlineSGDEquations,
)
from statphys.theory.online.solver import ODESolver

__all__ = [
    "ODESolver",
    # Base class
    "OnlineEquations",
    # Regression
    "OnlineSGDEquations",
    "OnlineRidgeEquations",
    # Classification
    "OnlinePerceptronEquations",
    "OnlineLogisticEquations",
    "OnlineHingeEquations",
    # Neural networks
    "OnlineCommitteeEquations",
]
