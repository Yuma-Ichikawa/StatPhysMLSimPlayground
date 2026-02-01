"""Classification loss functions."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from statphys.loss.base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """
    Binary Cross-Entropy loss.

    L(y, ŷ) = -y*log(σ(ŷ)) - (1-y)*log(1-σ(ŷ))

    For binary classification with labels in {0, 1}.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize CrossEntropyLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.
            label_smoothing: Label smoothing factor.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.label_smoothing = label_smoothing

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        # Convert {-1, 1} labels to {0, 1} if needed
        if y_true.min() < 0:
            y_true = (y_true + 1) / 2

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE with logits for numerical stability
        return F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

    def get_config(self) -> dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["label_smoothing"] = self.label_smoothing
        return config


class LogisticLoss(BaseLoss):
    """
    Logistic loss (equivalent to BCE but for {-1, +1} labels).

    L(y, ŷ) = log(1 + exp(-y*ŷ))

    The standard loss for logistic regression analysis.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize LogisticLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logistic loss."""
        # For numerical stability: log(1 + exp(-y*ŷ)) = softplus(-y*ŷ)
        return F.softplus(-y_true * y_pred)


class HingeLoss(BaseLoss):
    """
    Hinge loss (SVM loss).

    L(y, ŷ) = max(0, 1 - y*ŷ)

    The loss function for Support Vector Machines.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize HingeLoss.

        Args:
            margin: Margin parameter. Defaults to 1.0.
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.margin = margin

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hinge loss."""
        return torch.relu(self.margin - y_true * y_pred)

    def get_config(self) -> dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["margin"] = self.margin
        return config


class SquaredHingeLoss(BaseLoss):
    """
    Squared Hinge loss.

    L(y, ŷ) = max(0, 1 - y*ŷ)²

    Smooth version of hinge loss, differentiable everywhere.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize SquaredHingeLoss.

        Args:
            margin: Margin parameter.
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.margin = margin

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute squared hinge loss."""
        return torch.relu(self.margin - y_true * y_pred) ** 2


class PerceptronLoss(BaseLoss):
    """
    Perceptron loss.

    L(y, ŷ) = max(0, -y*ŷ)

    The classic perceptron learning rule loss.
    Zero margin hinge loss.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize PerceptronLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptron loss."""
        return torch.relu(-y_true * y_pred)


class ExponentialLoss(BaseLoss):
    """
    Exponential loss (AdaBoost loss).

    L(y, ŷ) = exp(-y*ŷ)

    Used in AdaBoost and exponential family analysis.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize ExponentialLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exponential loss."""
        # Clip to avoid overflow
        return torch.exp(torch.clamp(-y_true * y_pred, max=50))


class RampLoss(BaseLoss):
    """
    Ramp loss (robust SVM).

    L(y, ŷ) = min(1, max(0, 1 - y*ŷ))

    Bounded hinge loss, more robust to outliers.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize RampLoss.

        Args:
            margin: Margin parameter.
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.margin = margin

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ramp loss."""
        hinge = torch.relu(self.margin - y_true * y_pred)
        return torch.clamp(hinge, max=1.0)


class ProbitLoss(BaseLoss):
    """
    Probit loss for binary classification.

    L(y, ŷ) = -log(Phi(y*ŷ))

    where Phi is the Gaussian CDF.

    This loss is analytically convenient for replica calculations
    because the Gaussian integral over the label distribution
    yields closed-form expressions.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-7,
        **kwargs: Any,
    ):
        """
        Initialize ProbitLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.
            eps: Small value for numerical stability.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.eps = eps

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute probit loss.

        L(y, ŷ) = -log(Phi(y*ŷ)) = -log(0.5 * (1 + erf(y*ŷ / sqrt(2))))
        """
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        margin = y_true * y_pred
        phi = 0.5 * (1.0 + torch.erf(margin / np.sqrt(2.0)))

        # Clamp for numerical stability
        phi = torch.clamp(phi, min=self.eps, max=1.0 - self.eps)

        return -torch.log(phi)


class SoftmaxCrossEntropyLoss(BaseLoss):
    """
    Softmax Cross-Entropy loss for multi-class classification.

    L(y, z) = -log(exp(z_y) / sum_k exp(z_k))
           = -z_y + log(sum_k exp(z_k))

    where y is the true class index and z are the logits.

    Standard loss for multi-class softmax regression.
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize SoftmaxCrossEntropyLoss.

        Args:
            reg_param: L2 regularization parameter.
            reduction: Reduction method.
            label_smoothing: Label smoothing factor (0 to 1).

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.label_smoothing = label_smoothing

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax cross-entropy loss.

        Args:
            y_pred: Logits of shape (batch_size, n_classes) or (n_classes,).
            y_true: Class indices of shape (batch_size,) or scalar.

        Returns:
            Loss values.

        """
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        # Convert to long type for indexing
        y_true = y_true.long()

        if self.label_smoothing > 0:
            # Label smoothing
            n_classes = y_pred.size(-1)
            smooth_targets = torch.zeros_like(y_pred)
            smooth_targets.scatter_(1, y_true.unsqueeze(1), 1.0)
            smooth_targets = (
                smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            )

            log_probs = F.log_softmax(y_pred, dim=-1)
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(y_pred, y_true, reduction="none")

        return loss

    def get_config(self) -> dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["label_smoothing"] = self.label_smoothing
        return config


class MultiMarginLoss(BaseLoss):
    """
    Multi-class Hinge Loss (Crammer-Singer).

    L(y, z) = sum_{k != y} max(0, 1 - z_y + z_k) / (K-1)

    Multi-class extension of SVM hinge loss.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reg_param: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ):
        """
        Initialize MultiMarginLoss.

        Args:
            margin: Margin parameter.
            reg_param: L2 regularization parameter.
            reduction: Reduction method.

        """
        super().__init__(reg_param=reg_param, reduction=reduction, **kwargs)
        self.margin = margin

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-class hinge loss.

        Args:
            y_pred: Logits of shape (batch_size, n_classes).
            y_true: Class indices of shape (batch_size,).

        Returns:
            Loss values.

        """
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        y_true = y_true.long()
        n_classes = y_pred.size(-1)

        # Get correct class scores
        correct_scores = y_pred.gather(1, y_true.unsqueeze(1))  # (batch, 1)

        # Compute margins for all classes
        margins = self.margin - correct_scores + y_pred  # (batch, K)

        # Zero out correct class
        margins = margins.scatter(1, y_true.unsqueeze(1), 0.0)

        # Apply hinge
        losses = torch.relu(margins).sum(dim=-1) / (n_classes - 1)

        return losses
