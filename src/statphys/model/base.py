"""
Base classes for models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import numpy as np


class OrderParamsMixin:
    """
    Mixin class for computing order parameters.

    This mixin provides methods for computing overlaps and other
    order parameters that are common in statistical mechanics analyses.
    """

    def compute_self_overlap(self, normalize: bool = True) -> float:
        """
        Compute self-overlap q = (1/d) * ||w||^2.

        Args:
            normalize: If True, normalize by dimension.

        Returns:
            Self-overlap value.
        """
        if not hasattr(self, "get_weight_vector"):
            raise NotImplementedError("Model must implement get_weight_vector()")

        w = self.get_weight_vector()
        overlap = torch.sum(w**2).item()

        if normalize:
            return overlap / w.numel()
        return overlap

    def compute_teacher_overlap(
        self,
        teacher_weights: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """
        Compute overlap with teacher m = (1/d) * w^T @ w0.

        Args:
            teacher_weights: Teacher weight vector.
            normalize: If True, normalize by dimension.

        Returns:
            Teacher overlap value.
        """
        if not hasattr(self, "get_weight_vector"):
            raise NotImplementedError("Model must implement get_weight_vector()")

        w = self.get_weight_vector()
        w0 = teacher_weights.flatten()
        overlap = torch.dot(w.flatten(), w0).item()

        if normalize:
            return overlap / w.numel()
        return overlap


class BaseModel(nn.Module, OrderParamsMixin, ABC):
    """
    Abstract base class for all models in statistical mechanics simulations.

    This class extends PyTorch's nn.Module and provides additional
    functionality for order parameter computation and analysis.

    Attributes:
        d: Input dimension.

    Subclasses must implement:
        - forward(): Forward pass
        - get_weight_vector(): Return learnable weights as a flat vector
        - compute_order_params(): Compute model-specific order parameters
    """

    def __init__(self, d: int, **kwargs: Any):
        """
        Initialize the model.

        Args:
            d: Input dimension.
            **kwargs: Additional arguments for subclasses.
        """
        super().__init__()
        self.d = d

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).

        Returns:
            Output tensor.
        """
        pass

    @abstractmethod
    def get_weight_vector(self) -> torch.Tensor:
        """
        Get the main weight vector of the model.

        Returns:
            Weight tensor (flattened if necessary).
        """
        pass

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """
        Compute order parameters for the model.

        Args:
            teacher_params: Dictionary containing teacher parameters (W0, rho, eta, etc.)
            include_generalization_error: Whether to compute generalization error.

        Returns:
            Dictionary of order parameters.
        """
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)
        eta = teacher_params.get("eta", 0.0)

        # Compute basic overlaps
        q = self.compute_self_overlap()
        m = self.compute_teacher_overlap(W0) if W0 is not None else 0.0

        result = {
            "m": m,
            "q": q,
        }

        # Compute generalization error if requested
        if include_generalization_error and W0 is not None:
            # For linear teacher-student: E_g = 0.5 * (rho - 2*m + q) + 0.5 * eta * q
            # This is for MSE loss
            eg = 0.5 * (rho - 2 * m + q)
            if eta > 0:
                eg += 0.5 * eta  # Noise contribution
            result["eg"] = eg

        return result

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.

        Returns:
            Dictionary containing model configuration.
        """
        return {
            "class": self.__class__.__name__,
            "d": self.d,
        }

    def reset_parameters(self) -> None:
        """
        Reset model parameters to initial values.

        Override in subclasses for custom initialization.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, params={self.num_parameters})"
