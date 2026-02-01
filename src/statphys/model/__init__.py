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
from statphys.model.committee import CommitteeMachine, SoftCommitteeMachine
from statphys.model.linear import LinearClassifier, LinearRegression, RidgeRegression
from statphys.model.mlp import DeepNetwork, TwoLayerNetwork, TwoLayerNetworkReLU
from statphys.model.random_features import (
    DeepLinearNetwork,
    KernelRidgeModel,
    RandomFeaturesModel,
)
from statphys.model.registry import ModelRegistry, get_model, register_model
from statphys.model.softmax import SoftmaxRegression, SoftmaxRegressionWithBias
from statphys.model.transformer import SingleLayerAttention, SingleLayerTransformer
from statphys.model.sequence import (
    LinearSelfAttention,
    StateSpaceModel,
    LinearRNN,
    ModernHopfieldNetwork,
)

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
    # Sequence models (LSA, SSM, RNN, Hopfield)
    "LinearSelfAttention",
    "StateSpaceModel",
    "LinearRNN",
    "ModernHopfieldNetwork",
    # Registry
    "ModelRegistry",
    "register_model",
    "get_model",
]
