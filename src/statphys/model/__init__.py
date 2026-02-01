"""
Model module for statistical mechanics simulations.

This module provides:
- BaseModel: Abstract base class for all models
- LinearRegression: Linear regression model
- LinearClassifier: Linear classification model
- CommitteeMachine: Committee machine with hidden units
- TwoLayerNetwork: Two-layer neural network
- DeepLinearNetwork: Deep linear network (identity activation)
- RandomFeaturesModel: Random features / kernel approximation
- SoftmaxRegression: Multi-class softmax regression
- SingleLayerTransformer: Single-layer transformer model

Example:
    >>> from statphys.model import LinearRegression
    >>> model = LinearRegression(d=500)
    >>> y_pred = model(X)
"""

from statphys.model.base import BaseModel, OrderParamsMixin
from statphys.model.linear import LinearRegression, LinearClassifier, RidgeRegression
from statphys.model.committee import CommitteeMachine, SoftCommitteeMachine
from statphys.model.mlp import TwoLayerNetwork, TwoLayerNetworkReLU, DeepNetwork
from statphys.model.transformer import SingleLayerTransformer, SingleLayerAttention
from statphys.model.random_features import (
    RandomFeaturesModel,
    KernelRidgeModel,
    DeepLinearNetwork,
)
from statphys.model.softmax import SoftmaxRegression, SoftmaxRegressionWithBias
from statphys.model.registry import ModelRegistry, register_model, get_model

__all__ = [
    # Base classes
    "BaseModel",
    "OrderParamsMixin",
    # Linear models
    "LinearRegression",
    "LinearClassifier",
    "RidgeRegression",
    # Committee machines
    "CommitteeMachine",
    "SoftCommitteeMachine",
    # MLP models
    "TwoLayerNetwork",
    "TwoLayerNetworkReLU",
    "DeepNetwork",
    # Deep linear network
    "DeepLinearNetwork",
    # Random features / kernel
    "RandomFeaturesModel",
    "KernelRidgeModel",
    # Multi-class models
    "SoftmaxRegression",
    "SoftmaxRegressionWithBias",
    # Transformer models
    "SingleLayerTransformer",
    "SingleLayerAttention",
    # Registry
    "ModelRegistry",
    "register_model",
    "get_model",
]
