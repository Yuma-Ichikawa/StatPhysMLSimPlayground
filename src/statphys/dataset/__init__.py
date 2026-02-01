"""
Dataset generation module for statistical mechanics simulations.

This module provides:
- BaseDataset: Abstract base class for all datasets
- GaussianDataset: Standard iid Gaussian input data
- SparseDataset: Sparse input data
- StructuredDataset: Correlated/structured input data
- GLM datasets: Logistic/Probit teachers
- Gaussian Mixture datasets: For DMFT analysis

Example:
    >>> from statphys.dataset import GaussianDataset
    >>> dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)
    >>> X, y = dataset.generate_dataset(n_samples=1000)
"""

from statphys.dataset.base import BaseDataset, TeacherType
from statphys.dataset.gaussian import (
    GaussianDataset,
    GaussianClassificationDataset,
    GaussianMultiOutputDataset,
)
from statphys.dataset.sparse import SparseDataset, BernoulliGaussianDataset
from statphys.dataset.structured import (
    StructuredDataset,
    CorrelatedGaussianDataset,
    SpikedCovarianceDataset,
)
from statphys.dataset.glm import (
    LogisticTeacherDataset,
    ProbitTeacherDataset,
    GaussianMixtureDataset,
    MulticlassGaussianMixtureDataset,
)
from statphys.dataset.registry import DatasetRegistry, register_dataset, get_dataset

__all__ = [
    # Base classes
    "BaseDataset",
    "TeacherType",
    # Gaussian datasets
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
    # Registry
    "DatasetRegistry",
    "register_dataset",
    "get_dataset",
]
