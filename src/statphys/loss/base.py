"""Base class for loss functions."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    This class provides a common interface for all loss functions
    used in statistical mechanics simulations.

    Loss functions can include regularization terms and support
    both replica (batch) and online (single-sample) training modes.

    Scaling conventions:
    -------------------
    **Replica (batch learning)** with n = O(d):
        L = Σ ℓ(y, ŷ) + λ * ||w||²
        - Data term: Σℓ = n × O(1) → O(d) since n = O(d)
        - Regularization: λ * ||w||² = λ * d * q → O(d)
        - Total loss: O(d)

    **Online learning**:
        L = (1/d) * ℓ(y, ŷ) + (λ/d) * ||w||²
        - Data term: O(1/d) per sample
        - Regularization: (λ/d) * ||w||² = λ * q → O(1)
        - Total loss: O(1/d)

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

    def _compute_regularization_replica(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute regularization term for replica/batch learning.

        Replica scaling: λ * ||w||²
        This is O(d) since ||w||² ~ d * q for normalized weights.

        Args:
            model: The model (for accessing parameters).

        Returns:
            Regularization term.

        """
        if self.reg_param == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            reg = reg + torch.sum(param**2)

        # Replica: λ * ||w||² (no 1/d normalization)
        return self.reg_param * reg

    def _compute_regularization_online(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute regularization term for online learning.

        Online scaling: (λ/d) * ||w||² = λ * q
        This is O(1) since q = ||w||²/d.

        Args:
            model: The model (for accessing parameters).

        Returns:
            Regularization term.

        """
        if self.reg_param == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        d = 0
        for param in model.parameters():
            reg = reg + torch.sum(param**2)
            d += param.numel()

        # Online: (λ/d) * ||w||² = λ * (||w||²/d) = λ * q
        return self.reg_param * reg / d

    def _compute_regularization(
        self,
        model: nn.Module,
        online: bool = False,
    ) -> torch.Tensor:
        """
        Compute regularization term (backward compatible).

        Args:
            model: The model (for accessing parameters).
            online: If True, use online scaling (1/d).

        Returns:
            Regularization term.

        """
        if online:
            return self._compute_regularization_online(model)
        else:
            return self._compute_regularization_replica(model)

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

    def for_replica(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for replica/batch learning.

        Scaling: L = Σ ℓ(y, ŷ) + λ * ||w||²
        - Data term: sum over samples → O(d) since n = O(d)
        - Regularization: λ * ||w||² → O(d)
        - Total: O(d)

        Note: We do NOT divide by n because n = O(d), so the data term
        should remain O(d) to match the regularization term scaling.

        Args:
            y_pred: Predicted values.
            y_true: True values.
            model: Model for computing regularization. Optional.

        Returns:
            Total loss value for replica simulation.

        """
        # Compute main loss (SUM over samples, NOT mean)
        # This gives O(d) scaling since n = O(d)
        loss = self._compute_loss(y_pred, y_true)
        loss = loss.sum()  # Sum, not mean!

        # Add regularization with replica scaling: λ * ||w||² → O(d)
        if model is not None and self.reg_param > 0:
            reg = self._compute_regularization_replica(model)
            loss = loss + reg

        return loss

    def for_online(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module,
        d: int | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for online learning (single sample).

        Scaling: L = ℓ(y, ŷ) + (λ/2) * ||w||²/d
        - Data term: Single sample loss (no 1/d scaling)
        - Regularization: (λ/2) * ||w||²/d = (λ/2) * q → O(1)

        This scaling, combined with learning rate η/d, matches the
        standard online learning ODE theory:
            dm/dt = η(ρ - m) - ηλm
            dq/dt = η²V + 2η(m - q) - 2ηλq

        Usage:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr/d)
            loss = loss_fn.for_online(y_pred, y_true, model, d=d)

        Args:
            y_pred: Predicted value (single sample).
            y_true: True value (single sample).
            model: Model for computing regularization and dimension.
            d: Input dimension. If None, inferred from model.

        Returns:
            Total loss value for online learning.

        """
        # Infer dimension from model if not provided
        if d is None:
            d = sum(p.numel() for p in model.parameters())

        # Compute main loss for single sample (NO 1/d scaling on data term)
        loss = self._compute_loss(y_pred, y_true)

        # Handle both scalar and batched predictions
        if loss.dim() > 0:
            loss = loss.mean()  # If batched, take mean first

        # Apply (1/2) factor to match standard loss form L = (1/2)(y-ŷ)² + (λ/2)||w||²/d
        # This ensures gradients match ODE theory:
        #   ∇L = (ŷ - y)x/√d + (λ/d)w
        loss = 0.5 * loss

        # Add regularization: (λ/2) * ||w||²/d = (λ/2) * q
        # _compute_regularization_online returns λ * ||w||²/d
        if self.reg_param > 0:
            reg = self._compute_regularization_online(model)
            loss = loss + 0.5 * reg

        return loss

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
        online: bool = False,
    ) -> torch.Tensor:
        """
        Compute loss with optional regularization (backward compatible).

        For new code, prefer using `for_replica()` or `for_online()` methods
        which have clearer scaling semantics.

        Note: This method uses mean reduction for backward compatibility.
        Use `for_replica()` (sum) or `for_online()` (1/d scaling) for
        proper statistical mechanics scaling.

        Args:
            y_pred: Predicted values.
            y_true: True values.
            model: Model for computing regularization. Optional.
            online: If True, use online learning normalization.

        Returns:
            Total loss value.

        """
        # Backward compatible: use mean reduction
        loss = self._compute_loss(y_pred, y_true)
        loss = self._reduce(loss)  # Uses self.reduction (default: 'mean')

        # Add regularization if model is provided
        if model is not None and self.reg_param > 0:
            if online:
                reg = self._compute_regularization_online(model)
            else:
                reg = self._compute_regularization_replica(model)
            loss = loss + reg

        return loss

    def get_config(self) -> dict[str, Any]:
        """Get loss configuration."""
        return {
            "class": self.__class__.__name__,
            "reg_param": self.reg_param,
            "reduction": self.reduction,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reg_param={self.reg_param})"
