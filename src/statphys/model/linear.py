"""Linear models for regression and classification."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from statphys.model.base import BaseModel


class LinearRegression(BaseModel):
    """
    Linear regression model: y = (1/sqrt(d)) * W^T @ x.

    This is the standard model for studying ridge regression,
    LASSO, and other linear learning problems.

    Attributes:
        d: Input dimension.
        W: Weight vector of shape (d, 1).

    """

    def __init__(
        self,
        d: int,
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize LinearRegression.

        Args:
            d: Input dimension.
            init_scale: Scale for weight initialization.
            init_method: Initialization method ('normal', 'zero', 'uniform').

        """
        super().__init__(d=d, **kwargs)

        self.init_scale = init_scale
        self.init_method = init_method

        # Initialize weights
        self.W = nn.Parameter(torch.empty(d, 1))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights based on specified method."""
        if self.init_method == "normal":
            nn.init.normal_(self.W, mean=0.0, std=self.init_scale)
        elif self.init_method == "zero":
            nn.init.zeros_(self.W)
        elif self.init_method == "uniform":
            bound = self.init_scale * np.sqrt(3.0)
            nn.init.uniform_(self.W, -bound, bound)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).

        Returns:
            Predictions of shape (batch_size,) or scalar.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # y = (1/sqrt(d)) * W^T @ x^T = (1/sqrt(d)) * x @ W
        output = (x @ self.W / np.sqrt(self.d)).squeeze(-1)
        return output

    def get_weight_vector(self) -> torch.Tensor:
        """Return weight vector."""
        return self.W.flatten()

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """
        Compute order parameters for linear regression.

        Args:
            teacher_params: Dictionary with 'W0', 'rho', 'eta'.
            include_generalization_error: Whether to compute E_g.

        Returns:
            Dictionary with 'm', 'q', and optionally 'eg'.

        """
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)
        teacher_params.get("eta", 0.0)

        w = self.W

        # m = (1/d) * w^T @ W0
        m = (w.T @ W0 / self.d).item() if W0 is not None else 0.0

        # q = (1/d) * w^T @ w
        q = (w.T @ w / self.d).item()

        result = {"m": m, "q": q}

        if include_generalization_error and W0 is not None:
            # E_g = 0.5 * (rho - 2*sqrt(rho)*m + q) for normalized teacher
            # or E_g = 0.5 * (rho - 2*m + q) if teacher is already scaled
            eg = 0.5 * (rho - 2 * m + q)
            result["eg"] = eg

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "init_scale": self.init_scale,
                "init_method": self.init_method,
            }
        )
        return config


class RidgeRegression(LinearRegression):
    """
    Linear regression with built-in ridge regularization tracking.

    This is a convenience wrapper that stores the regularization parameter.
    """

    def __init__(
        self,
        d: int,
        reg_param: float = 0.01,
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize RidgeRegression.

        Args:
            d: Input dimension.
            reg_param: Ridge regularization parameter λ.
            init_scale: Scale for weight initialization.
            init_method: Initialization method.

        """
        super().__init__(d=d, init_scale=init_scale, init_method=init_method, **kwargs)
        self.reg_param = reg_param

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config["reg_param"] = self.reg_param
        return config


class LinearClassifier(BaseModel):
    """
    Linear classifier: y = sign(W^T @ x / sqrt(d)).

    Used for studying perceptron learning, SVM, and logistic regression.

    Attributes:
        d: Input dimension.
        W: Weight vector of shape (d, 1).

    """

    def __init__(
        self,
        d: int,
        init_scale: float = 1.0,
        init_method: str = "normal",
        output_type: str = "sign",
        **kwargs: Any,
    ):
        """
        Initialize LinearClassifier.

        Args:
            d: Input dimension.
            init_scale: Scale for weight initialization.
            init_method: Initialization method.
            output_type: Output type ('sign', 'logit', 'prob').

        """
        super().__init__(d=d, **kwargs)

        self.init_scale = init_scale
        self.init_method = init_method
        self.output_type = output_type

        self.W = nn.Parameter(torch.empty(d, 1))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        if self.init_method == "normal":
            nn.init.normal_(self.W, mean=0.0, std=self.init_scale)
        elif self.init_method == "zero":
            nn.init.zeros_(self.W)
        elif self.init_method == "uniform":
            bound = self.init_scale * np.sqrt(3.0)
            nn.init.uniform_(self.W, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Predictions based on output_type.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logit = (x @ self.W / np.sqrt(self.d)).squeeze(-1)

        if self.output_type == "sign":
            return torch.sign(logit)
        elif self.output_type == "logit":
            return logit
        elif self.output_type == "prob":
            return torch.sigmoid(logit)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def get_weight_vector(self) -> torch.Tensor:
        """Return weight vector."""
        return self.W.flatten()

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """
        Compute order parameters for linear classifier.

        For classification, generalization error is the classification error rate.
        """
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)

        w = self.W

        m = (w.T @ W0 / self.d).item() if W0 is not None else 0.0
        q = (w.T @ w / self.d).item()

        result = {"m": m, "q": q}

        if include_generalization_error and W0 is not None:
            # Classification error: P(sign(w^T x) != sign(W0^T x))
            # = (1/pi) * arccos(m / sqrt(q * rho))
            if q > 0 and rho > 0:
                cos_angle = m / np.sqrt(q * rho)
                cos_angle = np.clip(cos_angle, -1, 1)
                eg = np.arccos(cos_angle) / np.pi
            else:
                eg = 0.5  # Random guessing
            result["eg"] = eg

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "init_scale": self.init_scale,
                "init_method": self.init_method,
                "output_type": self.output_type,
            }
        )
        return config
