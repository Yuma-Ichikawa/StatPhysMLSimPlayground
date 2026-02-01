"""
Loss function registry.
"""

from typing import Any, Dict, Optional, Type

from statphys.loss.base import BaseLoss


class LossRegistry:
    """
    Registry for loss function classes.

    Example:
        >>> registry = LossRegistry()
        >>> registry.register("ridge", RidgeLoss)
        >>> loss_fn = registry.create("ridge", reg_param=0.01)
    """

    _instance: Optional["LossRegistry"] = None
    _registry: Dict[str, Type[BaseLoss]] = {}

    def __new__(cls) -> "LossRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, loss_class: Type[BaseLoss]) -> None:
        """Register a loss class."""
        self._registry[name.lower()] = loss_class

    def get(self, name: str) -> Type[BaseLoss]:
        """Get a registered loss class."""
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Loss '{name}' not found. Available: {available}")
        return self._registry[name_lower]

    def create(self, name: str, **kwargs: Any) -> BaseLoss:
        """Create a loss instance by name."""
        loss_class = self.get(name)
        return loss_class(**kwargs)

    def list(self) -> list[str]:
        """List all registered loss names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a loss is registered."""
        return name.lower() in self._registry


# Global registry instance
_global_registry = LossRegistry()


def register_loss(name: str) -> callable:
    """
    Decorator to register a loss class.

    Example:
        >>> @register_loss("my_loss")
        ... class MyLoss(BaseLoss):
        ...     pass
    """

    def decorator(cls: Type[BaseLoss]) -> Type[BaseLoss]:
        _global_registry.register(name, cls)
        return cls

    return decorator


def get_loss(name: str, **kwargs: Any) -> BaseLoss:
    """
    Get a loss instance by name from the global registry.

    Args:
        name: Name of the registered loss.
        **kwargs: Arguments to pass to the loss constructor.

    Returns:
        Loss instance.
    """
    return _global_registry.create(name, **kwargs)


# Register default losses
def _register_defaults() -> None:
    """Register default loss classes."""
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
        LogisticLoss,
        HingeLoss,
        SquaredHingeLoss,
        PerceptronLoss,
        ExponentialLoss,
        RampLoss,
    )

    # Regression losses
    _global_registry.register("mse", MSELoss)
    _global_registry.register("l2", MSELoss)
    _global_registry.register("ridge", RidgeLoss)
    _global_registry.register("lasso", LassoLoss)
    _global_registry.register("l1", LassoLoss)
    _global_registry.register("elastic_net", ElasticNetLoss)
    _global_registry.register("huber", HuberLoss)
    _global_registry.register("pseudo_huber", PseudoHuberLoss)

    # Classification losses
    _global_registry.register("cross_entropy", CrossEntropyLoss)
    _global_registry.register("bce", CrossEntropyLoss)
    _global_registry.register("logistic", LogisticLoss)
    _global_registry.register("hinge", HingeLoss)
    _global_registry.register("svm", HingeLoss)
    _global_registry.register("squared_hinge", SquaredHingeLoss)
    _global_registry.register("perceptron", PerceptronLoss)
    _global_registry.register("exponential", ExponentialLoss)
    _global_registry.register("ramp", RampLoss)


# Auto-register on import
_register_defaults()
