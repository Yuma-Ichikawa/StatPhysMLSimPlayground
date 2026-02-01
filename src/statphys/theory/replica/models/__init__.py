"""Pre-defined saddle-point equations for replica calculations.

Available models:
- ridge: RidgeRegressionEquations
- lasso: LassoEquations
- logistic: LogisticRegressionEquations
- perceptron: PerceptronEquations
- probit: ProbitEquations
- committee: CommitteeMachineEquations
"""

from statphys.theory.replica.models.base import ReplicaEquations
from statphys.theory.replica.models.committee import CommitteeMachineEquations
from statphys.theory.replica.models.lasso import LassoEquations
from statphys.theory.replica.models.linear import RidgeRegressionEquations
from statphys.theory.replica.models.logistic import LogisticRegressionEquations
from statphys.theory.replica.models.perceptron import PerceptronEquations
from statphys.theory.replica.models.probit import ProbitEquations

__all__ = [
    "ReplicaEquations",
    "RidgeRegressionEquations",
    "LassoEquations",
    "LogisticRegressionEquations",
    "PerceptronEquations",
    "ProbitEquations",
    "CommitteeMachineEquations",
]

# Registry for easy access by name
REPLICA_MODELS = {
    "ridge": RidgeRegressionEquations,
    "lasso": LassoEquations,
    "logistic": LogisticRegressionEquations,
    "perceptron": PerceptronEquations,
    "probit": ProbitEquations,
    "committee": CommitteeMachineEquations,
}


def get_replica_equations(name: str, **kwargs):
    """
    Get replica equations by name.

    Args:
        name: Model name ('ridge', 'lasso', 'logistic', 'perceptron', 'probit', 'committee')
        **kwargs: Parameters for the model

    Returns:
        ReplicaEquations instance

    Example:
        >>> equations = get_replica_equations("ridge", rho=1.0, eta=0.1, reg_param=0.01)
    """
    if name not in REPLICA_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(REPLICA_MODELS.keys())}")
    return REPLICA_MODELS[name](**kwargs)
