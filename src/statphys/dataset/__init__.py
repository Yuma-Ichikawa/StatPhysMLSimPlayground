"""
Dataset generation module for statistical mechanics simulations.

This module provides datasets organized by data type:

Basic Input Distributions:
- GaussianDataset: Standard iid Gaussian input data
- SparseDataset: Sparse input data
- StructuredDataset: Correlated/structured input data

GLM and Classification:
- LogisticTeacherDataset, ProbitTeacherDataset: GLM teachers
- GaussianMixtureDataset: For DMFT analysis

In-Context Learning:
- ICLLinearRegressionDataset: Linear regression ICL tasks
- ICLNonlinearRegressionDataset: Nonlinear regression ICL tasks

Sequence/Token Data:
- MarkovChainDataset: Markov chain sequences
- CopyTaskDataset: Copy/induction tasks
- GeneralizedPottsDataset: Potts model sequences
- TiedLowRankAttentionDataset: Attention teacher sequences
- MixedGaussianSequenceDataset: Cluster-structured sequences

Attention-Indexed:
- AttentionIndexedModelDataset: AIM model

Fairness/Bias:
- TeacherMixtureFairnessDataset: Two-group fairness analysis

Noisy Labels:
- NoisyGMMSelfDistillationDataset: Label noise for distillation

Example:
    >>> from statphys.dataset import GaussianDataset
    >>> dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)
    >>> X, y = dataset.generate_dataset(n_samples=1000)
"""

from statphys.dataset.base import BaseDataset, TeacherType

# Basic input distributions
from statphys.dataset.gaussian import (
    GaussianClassificationDataset,
    GaussianDataset,
    GaussianMultiOutputDataset,
)
from statphys.dataset.sparse import BernoulliGaussianDataset, SparseDataset
from statphys.dataset.structured import (
    CorrelatedGaussianDataset,
    SpikedCovarianceDataset,
    StructuredDataset,
)

# GLM and classification
from statphys.dataset.glm import (
    GaussianMixtureDataset,
    LogisticTeacherDataset,
    MulticlassGaussianMixtureDataset,
    ProbitTeacherDataset,
)

# In-Context Learning
from statphys.dataset.icl import (
    ICLLinearRegressionDataset,
    ICLNonlinearRegressionDataset,
)

# Sequence/Token data
from statphys.dataset.sequence import (
    MarkovChainDataset,
    CopyTaskDataset,
    GeneralizedPottsDataset,
    TiedLowRankAttentionDataset,
    MixedGaussianSequenceDataset,
)

# Attention-indexed
from statphys.dataset.attention import (
    AttentionIndexedModelDataset,
)

# Fairness/Bias
from statphys.dataset.fairness import (
    TeacherMixtureFairnessDataset,
)

# Noisy labels
from statphys.dataset.noisy import (
    NoisyGMMSelfDistillationDataset,
)

# Registry
from statphys.dataset.registry import DatasetRegistry, get_dataset, register_dataset

__all__ = [
    # Base classes
    "BaseDataset",
    "TeacherType",
    # Basic Gaussian datasets
    "GaussianDataset",
    "GaussianClassificationDataset",
    "GaussianMultiOutputDataset",
    # Sparse datasets
    "SparseDataset",
    "BernoulliGaussianDataset",
    # Structured datasets
    "StructuredDataset",
    "CorrelatedGaussianDataset",
    "SpikedCovarianceDataset",
    # GLM datasets
    "LogisticTeacherDataset",
    "ProbitTeacherDataset",
    # Gaussian Mixture datasets
    "GaussianMixtureDataset",
    "MulticlassGaussianMixtureDataset",
    # ICL datasets
    "ICLLinearRegressionDataset",
    "ICLNonlinearRegressionDataset",
    # Sequence/Token datasets
    "MarkovChainDataset",
    "CopyTaskDataset",
    "GeneralizedPottsDataset",
    "TiedLowRankAttentionDataset",
    "MixedGaussianSequenceDataset",
    # Attention-indexed datasets
    "AttentionIndexedModelDataset",
    # Fairness/Bias datasets
    "TeacherMixtureFairnessDataset",
    # Noisy label datasets
    "NoisyGMMSelfDistillationDataset",
    # Registry
    "DatasetRegistry",
    "register_dataset",
    "get_dataset",
]
