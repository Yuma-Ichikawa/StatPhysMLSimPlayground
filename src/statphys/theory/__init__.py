"""
Theory module for statistical mechanics calculations.

This module provides:
- Replica method: Saddle-point equation solvers with damping
- Online learning: ODE solvers for learning dynamics
- DMFT: Coming soon

Example:
    >>> from statphys.theory.replica import SaddlePointSolver
    >>> solver = SaddlePointSolver(equations=my_equations)
    >>> result = solver.solve(alpha_values=[0.1, 0.5, 1.0, 2.0])
"""

from statphys.theory.base import BaseTheory, TheoryResult, TheoryType
from statphys.theory.replica import SaddlePointSolver, ReplicaEquations
from statphys.theory.online import ODESolver, OnlineEquations

__all__ = [
    # Base classes
    "BaseTheory",
    "TheoryResult",
    "TheoryType",
    # Replica
    "SaddlePointSolver",
    "ReplicaEquations",
    # Online
    "ODESolver",
    "OnlineEquations",
]
