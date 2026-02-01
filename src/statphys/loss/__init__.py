"""
Loss functions module for statistical mechanics simulations.

This module provides:
- BaseLoss: Abstract base class for loss functions
- Regression losses: MSE, Ridge, LASSO, ElasticNet, Huber
- Classification losses: CrossEntropy, Hinge, Logistic, Probit, Softmax

Example:
    >>> from statphys.loss import RidgeLoss
    >>> loss_fn = RidgeLoss(reg_param=0.01)
    >>> loss = loss_fn(y_pred, y_true, model)

"""

from statphys.loss.base import BaseLoss
from statphys.loss.classification import (
    CrossEntropyLoss,
    ExponentialLoss,
    HingeLoss,
    LogisticLoss,
    MultiMarginLoss,
    PerceptronLoss,
    ProbitLoss,
    RampLoss,
    SoftmaxCrossEntropyLoss,
    SquaredHingeLoss,
)
from statphys.loss.registry import LossRegistry, get_loss, register_loss
from statphys.loss.regression import (
    ElasticNetLoss,
    HuberLoss,
    LassoLoss,
    MSELoss,
    PseudoHuberLoss,
    RidgeLoss,
)

__all__ = [
    # Base class
    "BaseLoss",
    # Regression losses
    "MSELoss",
    "RidgeLoss",
    "LassoLoss",
    "ElasticNetLoss",
    "HuberLoss",
    "PseudoHuberLoss",
    # Binary classification losses
    "CrossEntropyLoss",
    "HingeLoss",
    "SquaredHingeLoss",
    "LogisticLoss",
    "PerceptronLoss",
    "ExponentialLoss",
    "RampLoss",
    "ProbitLoss",
    # Multi-class classification losses
    "SoftmaxCrossEntropyLoss",
    "MultiMarginLoss",
    # Registry
    "LossRegistry",
    "register_loss",
    "get_loss",
]
