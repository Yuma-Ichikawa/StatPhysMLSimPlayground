"""
Base class for loss functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    This class provides a common interface for all loss functions
    used in statistical mechanics simulations.

    Loss functions can include regularization terms and support
    both online (single-sample) and batch training modes.

    Attributes:
        reg_param: Regularization parameter (λ).
        reduction: How to reduce the loss ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize BaseLoss.

        Args:
            reg_param: Regularization parameter. Defaults to 0.0.
            reduction: Reduction method. Defaults to 'mean'.
        """
        self.reg_param = reg_param
        self.reduction = reduction

    @abstractmethod
    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the main loss term (without regularization).

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            Loss tensor (before reduction).
        """
        pass

    def _compute_regularization(
        self,
        model: nn.Module,
        online: bool = False,
    ) -> torch.Tensor:
        """
        Compute regularization term.

        Args:
            model: The model (for accessing parameters).
            online: If True, normalize by d (for online learning).

        Returns:
            Regularization term.
        """
        if self.reg_param == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            reg = reg + torch.sum(param**2)

        if online:
            # For online learning: λ * ||w||^2 / d
            d = sum(p.numel() for p in model.parameters())
            reg = self.reg_param * reg / d
        else:
            # For batch learning: λ * ||w||^2
            reg = self.reg_param * reg

        return reg

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: Optional[nn.Module] = None,
        online: bool = False,
    ) -> torch.Tensor:
        """
        Compute loss with optional regularization.

        Args:
            y_pred: Predicted values.
            y_true: True values.
            model: Model for computing regularization. Optional.
            online: If True, use online learning normalization.

        Returns:
            Total loss value.
        """
        # Compute main loss
        loss = self._compute_loss(y_pred, y_true)
        loss = self._reduce(loss)

        # Add regularization if model is provided
        if model is not None and self.reg_param > 0:
            reg = self._compute_regularization(model, online=online)
            loss = loss + reg

        return loss

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        return {
            "class": self.__class__.__name__,
            "reg_param": self.reg_param,
            "reduction": self.reduction,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reg_param={self.reg_param})"
