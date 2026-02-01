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
    GaussianClassificationDataset,
    GaussianDataset,
    GaussianMultiOutputDataset,
)
from statphys.dataset.glm import (
    GaussianMixtureDataset,
    LogisticTeacherDataset,
    MulticlassGaussianMixtureDataset,
    ProbitTeacherDataset,
)
from statphys.dataset.registry import DatasetRegistry, get_dataset, register_dataset
from statphys.dataset.sparse import BernoulliGaussianDataset, SparseDataset
from statphys.dataset.structured import (
    CorrelatedGaussianDataset,
    SpikedCovarianceDataset,
    StructuredDataset,
)

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
