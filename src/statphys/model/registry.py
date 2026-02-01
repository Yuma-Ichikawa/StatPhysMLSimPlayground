"""Model registry for dynamic model creation."""

from typing import Any, Optional

from statphys.model.base import BaseModel


class ModelRegistry:
    """
    Registry for model classes.

    Allows registering and retrieving model classes by name.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("linear", LinearRegression)
        >>> model = registry.create("linear", d=500)

    """

    _instance: Optional["ModelRegistry"] = None
    _registry: dict[str, type[BaseModel]] = {}

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a model class."""
        self._registry[name.lower()] = model_class

    def get(self, name: str) -> type[BaseModel]:
        """Get a registered model class."""
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._registry[name_lower]

    def create(self, name: str, **kwargs: Any) -> BaseModel:
        """Create a model instance by name."""
        model_class = self.get(name)
        return model_class(**kwargs)

    def list(self) -> list[str]:
        """List all registered model names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a model is registered."""
        return name.lower() in self._registry


# Global registry instance
_global_registry = ModelRegistry()


def register_model(name: str) -> callable:
    """
    Decorator to register a model class.

    Example:
        >>> @register_model("my_model")
        ... class MyModel(BaseModel):
        ...     pass

    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        _global_registry.register(name, cls)
        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """
    Get a model instance by name from the global registry.

    Args:
        name: Name of the registered model.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        Model instance.

    """
    return _global_registry.create(name, **kwargs)


# Register default models
def _register_defaults() -> None:
    """Register default model classes."""
    from statphys.model.committee import CommitteeMachine, SoftCommitteeMachine
    from statphys.model.linear import LinearClassifier, LinearRegression, RidgeRegression
    from statphys.model.mlp import DeepNetwork, TwoLayerNetwork, TwoLayerNetworkReLU
    from statphys.model.transformer import SingleLayerAttention, SingleLayerTransformer

    _global_registry.register("linear", LinearRegression)
    _global_registry.register("linear_regression", LinearRegression)
    _global_registry.register("ridge", RidgeRegression)
    _global_registry.register("linear_classifier", LinearClassifier)
    _global_registry.register("perceptron", LinearClassifier)
    _global_registry.register("committee", CommitteeMachine)
    _global_registry.register("soft_committee", SoftCommitteeMachine)
    _global_registry.register("two_layer", TwoLayerNetwork)
    _global_registry.register("two_layer_relu", TwoLayerNetworkReLU)
    _global_registry.register("mlp", TwoLayerNetwork)
    _global_registry.register("deep", DeepNetwork)
    _global_registry.register("attention", SingleLayerAttention)
    _global_registry.register("transformer", SingleLayerTransformer)


# Auto-register on import
_register_defaults()
