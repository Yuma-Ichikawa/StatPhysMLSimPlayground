"""
Regression loss functions.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from statphys.loss.base import BaseLoss


class MSELoss(BaseLoss):
    """
    Mean Squared Error loss.

    L(y, ŷ) = (1/n) * Σ(y - ŷ)²

    The canonical loss for studying ridge regression and
    gradient descent dynamics.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize MSELoss.

        Args:
            reg_param: Regularization parameter (for L2/Ridge).
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return (y_pred - y_true) ** 2


class RidgeLoss(BaseLoss):
    """
    Ridge regression loss (MSE + L2 regularization).

    L(y, ŷ, w) = (1/n) * Σ(y - ŷ)² + λ||w||²

    Equivalent to MSELoss with reg_param > 0, but explicitly named.
    """

    def __init__(
        self,
        reg_param: float = 0.01,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize RidgeLoss.

        Args:
            reg_param: Ridge parameter λ. Defaults to 0.01.
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return (y_pred - y_true) ** 2


class LassoLoss(BaseLoss):
    """
    LASSO regression loss (MSE + L1 regularization).

    L(y, ŷ, w) = (1/n) * Σ(y - ŷ)² + λ||w||₁

    Important for studying sparse solutions and compressed sensing.
    """

    def __init__(
        self,
        reg_param: float = 0.01,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize LassoLoss.

        Args:
            reg_param: LASSO parameter λ. Defaults to 0.01.
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return (y_pred - y_true) ** 2

    def _compute_regularization(
        self,
        model: nn.Module,
        online: bool = False,
    ) -> torch.Tensor:
        """Compute L1 regularization."""
        if self.reg_param == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            reg = reg + torch.sum(torch.abs(param))

        if online:
            d = sum(p.numel() for p in model.parameters())
            reg = self.reg_param * reg / d
        else:
            reg = self.reg_param * reg

        return reg


class ElasticNetLoss(BaseLoss):
    """
    Elastic Net loss (MSE + L1 + L2 regularization).

    L(y, ŷ, w) = (1/n) * Σ(y - ŷ)² + λ₁||w||₁ + λ₂||w||²

    Combines benefits of LASSO and Ridge.
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        reg_param: float = 0.01,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize ElasticNetLoss.

        Args:
            l1_ratio: Mixing parameter (0=Ridge, 1=LASSO). Defaults to 0.5.
            reg_param: Total regularization strength.
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.l1_ratio = l1_ratio

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return (y_pred - y_true) ** 2

    def _compute_regularization(
        self,
        model: nn.Module,
        online: bool = False,
    ) -> torch.Tensor:
        """Compute Elastic Net regularization (L1 + L2)."""
        if self.reg_param == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)

        for param in model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(param))
            l2_reg = l2_reg + torch.sum(param**2)

        reg = self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg

        if online:
            d = sum(p.numel() for p in model.parameters())
            reg = self.reg_param * reg / d
        else:
            reg = self.reg_param * reg

        return reg

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["l1_ratio"] = self.l1_ratio
        return config


class HuberLoss(BaseLoss):
    """
    Huber loss (robust regression).

    L(y, ŷ) = 0.5*(y - ŷ)²           if |y - ŷ| ≤ δ
             δ*(|y - ŷ| - 0.5*δ)    otherwise

    Robust to outliers while being differentiable.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize HuberLoss.

        Args:
            delta: Threshold for switching between quadratic and linear.
            reg_param: Regularization parameter.
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.delta = delta

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Huber loss."""
        diff = torch.abs(y_pred - y_true)
        quadratic = 0.5 * diff**2
        linear = self.delta * (diff - 0.5 * self.delta)
        return torch.where(diff <= self.delta, quadratic, linear)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["delta"] = self.delta
        return config


class PseudoHuberLoss(BaseLoss):
    """
    Pseudo-Huber loss (smooth approximation to Huber).

    L(y, ŷ) = δ² * (sqrt(1 + ((y - ŷ)/δ)²) - 1)

    Smooth everywhere, approximates Huber loss.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize PseudoHuberLoss.

        Args:
            delta: Scale parameter.
            reg_param: Regularization parameter.
            reduction: Reduction method.
        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.delta = delta

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Pseudo-Huber loss."""
        diff = (y_pred - y_true) / self.delta
        return self.delta**2 * (torch.sqrt(1 + diff**2) - 1)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["delta"] = self.delta
        return config
