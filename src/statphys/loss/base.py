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

    **Online learning** (single sample; with the f(x) = w·x/√d model
    normalization the optimizer lr maps 1:1 onto the ODE learning rate η):
        L = c * ℓ(y, ŷ) + (λ/2) * ||w||² / d
        - Data term: single-sample loss, scaled by ``online_scale`` (c).
          For squared-error losses c = 1/2 so that the data term matches
          the theory convention ℓ = (1/2)(y - ŷ)².
          For margin/log losses c = 1 (the loss is already in standard form).
        - Regularization: (λ/2) * ||w||²/d = (λ/2) * q → O(1)

    Attributes:
        reg_param: Regularization parameter (λ).
        reduction: How to reduce the loss ('mean', 'sum', 'none').
        online_scale: Multiplier applied to the data term in `for_online()`.
            Subclasses override this to match the theory convention
            (0.5 for squared-error losses, 1.0 otherwise).

    """

    #: Multiplier applied to the data term in `for_online()`.
    #: Squared-error losses override this with 0.5 so that the effective
    #: single-sample loss is (1/2)(y - ŷ)², matching online SGD theory.
    online_scale: float = 1.0

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
            return torch.tensor(0.0, device=self._model_device(model))

        reg = torch.tensor(0.0, device=self._model_device(model))
        for param in model.parameters():
            reg = reg + torch.sum(param**2)

        # Replica: λ * ||w||² (no 1/d normalization)
        return self.reg_param * reg

    def _compute_regularization_online(
        self,
        model: nn.Module,
        d: int | None = None,
    ) -> torch.Tensor:
        """
        Compute regularization term for online learning.

        Online scaling: (λ/d) * ||w||² = λ * q
        This is O(1) since q = ||w||²/d.

        Args:
            model: The model (for accessing parameters).
            d: Normalization dimension. If None, inferred from the model
               (`model.d` if available, otherwise total parameter count).

        Returns:
            Regularization term.

        """
        if self.reg_param == 0:
            return torch.tensor(0.0, device=self._model_device(model))

        reg = torch.tensor(0.0, device=self._model_device(model))
        numel = 0
        for param in model.parameters():
            reg = reg + torch.sum(param**2)
            numel += param.numel()

        d = self._resolve_dim(model, d, numel)

        # Online: (λ/d) * ||w||² = λ * (||w||²/d) = λ * q
        return self.reg_param * reg / d

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        """Return the device of the model's first parameter (CPU if none)."""
        param = next(model.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    @staticmethod
    def _resolve_dim(model: nn.Module, d: int | None, numel: int) -> int:
        """Resolve normalization dimension: explicit d > model.d > numel."""
        if d is not None:
            return d
        model_d = getattr(model, "d", None)
        if isinstance(model_d, int) and model_d > 0:
            return model_d
        return max(numel, 1)

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

        Scaling: L = c * ℓ(y, ŷ) + (λ/2) * ||w||²/d
        where c = ``self.online_scale`` (0.5 for squared-error losses,
        1.0 for margin/log losses whose standard form needs no factor).

        With this scaling, SGD with optimizer learning rate η matches the
        standard online learning ODE theory in time t = τ/d. For MSE:
            dm/dt = η(ρ - m) - ηλm
            dq/dt = η²V + 2η(m - q) - 2ηλq

        Usage:
            optimizer = torch.optim.SGD(model.parameters(), lr=eta)
            loss = loss_fn.for_online(y_pred, y_true, model, d=d)

        Args:
            y_pred: Predicted value (single sample).
            y_true: True value (single sample).
            model: Model for computing regularization and dimension.
            d: Input dimension used for the regularization normalization.
               If None, inferred from `model.d` (fallback: parameter count).

        Returns:
            Total loss value for online learning.

        """
        # Compute main loss for single sample (no 1/d scaling on data term)
        loss = self._compute_loss(y_pred, y_true)

        # Handle both scalar and batched predictions
        if loss.dim() > 0:
            loss = loss.mean()  # If batched, take mean first

        # Scale data term to the theory-standard form.
        # For squared-error losses online_scale = 0.5, giving (1/2)(y-ŷ)²
        # so that ∇L = (ŷ - y)x/√d + (λ/d)w matches the ODE theory.
        if self.online_scale != 1.0:
            loss = self.online_scale * loss

        # Add regularization: (λ/2) * ||w||²/d = (λ/2) * q
        # _compute_regularization_online returns λ * ||w||²/d
        if self.reg_param > 0:
            reg = self._compute_regularization_online(model, d=d)
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
        Use `for_replica()` (sum over samples) or `for_online()`
        (single-sample loss with (λ/2)||w||²/d regularization) for
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
