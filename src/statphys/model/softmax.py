"""
Softmax (Multinomial Logistic) Regression model.

Standard model for multi-class classification with K classes.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from statphys.model.base import BaseModel


class SoftmaxRegression(BaseModel):
    """
    Softmax (Multinomial Logistic) Regression for multi-class classification.

    Architecture:
        z_k = (1/sqrt(d)) * w_k^T @ x  for k = 1, ..., K
        P(y=k|x) = exp(z_k) / sum_j exp(z_j)

    Uses the standard softmax parameterization with K weight vectors.

    Attributes:
        d: Input dimension.
        n_classes: Number of classes (K).
        W: Weight matrix (K, d).
    """

    def __init__(
        self,
        d: int,
        n_classes: int = 10,
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize SoftmaxRegression.

        Args:
            d: Input dimension.
            n_classes: Number of classes (K).
            init_scale: Scale for weight initialization.
            init_method: Initialization method ('normal', 'xavier', 'orthogonal').
        """
        super().__init__(d=d, **kwargs)

        self.n_classes = n_classes
        self.init_scale = init_scale
        self.init_method = init_method

        # Weight matrix (K, d)
        self.W = nn.Parameter(torch.empty(n_classes, d))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        if self.init_method == "normal":
            nn.init.normal_(self.W, std=self.init_scale / np.sqrt(self.d))
        elif self.init_method == "xavier":
            nn.init.xavier_normal_(self.W, gain=self.init_scale)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.W, gain=self.init_scale)
        else:
            nn.init.normal_(self.W, std=self.init_scale / np.sqrt(self.d))

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).
            return_logits: If True, return logits instead of probabilities.

        Returns:
            Probabilities or logits of shape (batch_size, K) or (K,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute logits: z_k = (1/sqrt(d)) * w_k^T @ x
        logits = (x @ self.W.T) / np.sqrt(self.d)  # (batch_size, K)

        if return_logits:
            output = logits
        else:
            output = F.softmax(logits, dim=-1)

        if squeeze_output:
            return output.squeeze(0)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor.

        Returns:
            Predicted class indices.
        """
        logits = self.forward(x, return_logits=True)
        return torch.argmax(logits, dim=-1)

    def get_weight_vector(self) -> torch.Tensor:
        """Return flattened weight matrix."""
        return self.W.flatten()

    def get_class_weights(self, k: int) -> torch.Tensor:
        """Return weight vector for class k."""
        return self.W[k]

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """
        Compute order parameters for multi-class model.

        For softmax regression, we compute per-class overlaps:
        - Q_kk': (1/d) * w_k^T @ w_k'
        - R_km: (1/d) * w_k^T @ w*_m (if teacher weights available)

        Args:
            teacher_params: Dictionary containing teacher parameters.
            include_generalization_error: Whether to estimate error.

        Returns:
            Dictionary of order parameters.
        """
        # Self-overlap matrix
        Q = (self.W @ self.W.T / self.d).detach()

        result = {
            "Q_diag_mean": Q.diag().mean().item(),  # Average self-overlap
            "Q_offdiag_mean": (Q.sum() - Q.diag().sum()).item() / (self.n_classes * (self.n_classes - 1)),
        }

        # Teacher overlap if available
        if "means" in teacher_params:
            # For GMM teacher
            means = teacher_params["means"]
            R = (self.W @ means.T / self.d).detach()
            result["R_mean"] = R.diag().mean().item()

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
            "init_scale": self.init_scale,
            "init_method": self.init_method,
        })
        return config


class SoftmaxRegressionWithBias(SoftmaxRegression):
    """
    Softmax Regression with bias terms.

    Architecture:
        z_k = (1/sqrt(d)) * w_k^T @ x + b_k
        P(y=k|x) = exp(z_k) / sum_j exp(z_j)
    """

    def __init__(
        self,
        d: int,
        n_classes: int = 10,
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """Initialize with bias."""
        super().__init__(
            d=d,
            n_classes=n_classes,
            init_scale=init_scale,
            init_method=init_method,
            **kwargs
        )

        # Bias terms
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass with bias."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        logits = (x @ self.W.T) / np.sqrt(self.d) + self.bias

        if return_logits:
            output = logits
        else:
            output = F.softmax(logits, dim=-1)

        if squeeze_output:
            return output.squeeze(0)
        return output
