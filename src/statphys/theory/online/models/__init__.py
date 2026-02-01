"""Pre-defined ODE equations for online learning dynamics.

Available models:
- linear: OnlineSGDEquations, OnlineRidgeEquations
- perceptron: OnlinePerceptronEquations
- logistic: OnlineLogisticEquations
- hinge: OnlineHingeEquations
- committee: OnlineCommitteeEquations
"""

from statphys.theory.online.models.base import OnlineEquations
from statphys.theory.online.models.committee import OnlineCommitteeEquations
from statphys.theory.online.models.hinge import OnlineHingeEquations
from statphys.theory.online.models.linear import (
    OnlineRidgeEquations,
    OnlineSGDEquations,
)
from statphys.theory.online.models.logistic import OnlineLogisticEquations
from statphys.theory.online.models.perceptron import OnlinePerceptronEquations

__all__ = [
    "OnlineEquations",
    "OnlineSGDEquations",
    "OnlineRidgeEquations",
    "OnlinePerceptronEquations",
    "OnlineLogisticEquations",
    "OnlineHingeEquations",
    "OnlineCommitteeEquations",
]

# Registry for easy access by name
ONLINE_MODELS = {
    "sgd": OnlineSGDEquations,
    "ridge": OnlineRidgeEquations,
    "perceptron": OnlinePerceptronEquations,
    "logistic": OnlineLogisticEquations,
    "hinge": OnlineHingeEquations,
    "committee": OnlineCommitteeEquations,
}


def get_online_equations(name: str, **kwargs):
    """
    Get online equations by name.

    Args:
        name: Model name ('sgd', 'ridge', 'perceptron', 'logistic', 'hinge', 'committee')
        **kwargs: Parameters for the model

    Returns:
        OnlineEquations instance

    Example:
        >>> equations = get_online_equations("sgd", rho=1.0, lr=0.1)

    """
    if name not in ONLINE_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(ONLINE_MODELS.keys())}")
    return ONLINE_MODELS[name](**kwargs)
