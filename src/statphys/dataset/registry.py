"""Dataset registry for dynamic dataset creation."""

from typing import Any, Optional

from statphys.dataset.base import BaseDataset


class DatasetRegistry:
    """
    Registry for dataset classes.

    Allows registering and retrieving dataset classes by name,
    enabling configuration-driven dataset creation.

    Example:
        >>> registry = DatasetRegistry()
        >>> registry.register("gaussian", GaussianDataset)
        >>> dataset = registry.create("gaussian", d=500, rho=1.0)

    """

    _instance: Optional["DatasetRegistry"] = None
    _registry: dict[str, type[BaseDataset]] = {}

    def __new__(cls) -> "DatasetRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, dataset_class: type[BaseDataset]) -> None:
        """
        Register a dataset class.

        Args:
            name: Name to register the dataset under.
            dataset_class: Dataset class to register.

        """
        self._registry[name.lower()] = dataset_class

    def get(self, name: str) -> type[BaseDataset]:
        """
        Get a registered dataset class.

        Args:
            name: Name of the registered dataset.

        Returns:
            The registered dataset class.

        Raises:
            KeyError: If the dataset is not registered.

        """
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return self._registry[name_lower]

    def create(self, name: str, **kwargs: Any) -> BaseDataset:
        """
        Create a dataset instance by name.

        Args:
            name: Name of the registered dataset.
            **kwargs: Arguments to pass to the dataset constructor.

        Returns:
            Dataset instance.

        """
        dataset_class = self.get(name)
        return dataset_class(**kwargs)

    def list(self) -> list[str]:
        """List all registered dataset names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a dataset is registered."""
        return name.lower() in self._registry


# Global registry instance
_global_registry = DatasetRegistry()


def register_dataset(name: str) -> callable:
    """
    Decorator to register a dataset class.

    Example:
        >>> @register_dataset("my_dataset")
        ... class MyDataset(BaseDataset):
        ...     pass

    """

    def decorator(cls: type[BaseDataset]) -> type[BaseDataset]:
        _global_registry.register(name, cls)
        return cls

    return decorator


def get_dataset(name: str, **kwargs: Any) -> BaseDataset:
    """
    Get a dataset instance by name from the global registry.

    Args:
        name: Name of the registered dataset.
        **kwargs: Arguments to pass to the dataset constructor.

    Returns:
        Dataset instance.

    """
    return _global_registry.create(name, **kwargs)


# Register default datasets
def _register_defaults() -> None:
    """Register default dataset classes."""
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

    _global_registry.register("gaussian", GaussianDataset)
    _global_registry.register("gaussian_regression", GaussianDataset)
    _global_registry.register("gaussian_classification", GaussianClassificationDataset)
    _global_registry.register("gaussian_multioutput", GaussianMultiOutputDataset)
    _global_registry.register("sparse", SparseDataset)
    _global_registry.register("bernoulli_gaussian", BernoulliGaussianDataset)
    _global_registry.register("structured", StructuredDataset)
    _global_registry.register("correlated_gaussian", CorrelatedGaussianDataset)
    _global_registry.register("spiked_covariance", SpikedCovarianceDataset)


# Auto-register on import
_register_defaults()
