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
from statphys.loss.regression import (
    MSELoss,
    RidgeLoss,
    LassoLoss,
    ElasticNetLoss,
    HuberLoss,
    PseudoHuberLoss,
)
from statphys.loss.classification import (
    CrossEntropyLoss,
    HingeLoss,
    SquaredHingeLoss,
    LogisticLoss,
    PerceptronLoss,
    ExponentialLoss,
    RampLoss,
    ProbitLoss,
    SoftmaxCrossEntropyLoss,
    MultiMarginLoss,
)
from statphys.loss.registry import LossRegistry, register_loss, get_loss

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
