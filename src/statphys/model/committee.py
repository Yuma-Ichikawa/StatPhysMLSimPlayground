"""Committee machine models."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from statphys.model.base import BaseModel


class CommitteeMachine(BaseModel):
    """
    Committee machine with K hidden units.

    Architecture: y = (1/sqrt(K)) * sum_k sign(W_k^T @ x / sqrt(d))

    This is the canonical model for studying hidden layer learning
    in neural networks from a statistical mechanics perspective.

    Attributes:
        d: Input dimension.
        k: Number of hidden units (committee size).
        W: Weight matrix of shape (K, d).

    """

    def __init__(
        self,
        d: int,
        k: int = 2,
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize CommitteeMachine.

        Args:
            d: Input dimension.
            k: Number of hidden units.
            init_scale: Scale for weight initialization.
            init_method: Initialization method.

        """
        super().__init__(d=d, **kwargs)

        self.k = k
        self.init_scale = init_scale
        self.init_method = init_method

        # Weight matrix: (K, d)
        self.W = nn.Parameter(torch.empty(k, d))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        if self.init_method == "normal":
            nn.init.normal_(self.W, mean=0.0, std=self.init_scale)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.W, gain=self.init_scale)
        elif self.init_method == "uniform":
            bound = self.init_scale * np.sqrt(3.0)
            nn.init.uniform_(self.W, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).

        Returns:
            Output tensor of shape (batch_size,) or scalar.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Hidden layer: (batch, K) = (batch, d) @ (d, K)
        hidden = x @ self.W.T / np.sqrt(self.d)

        # Sign activation and aggregation
        activated = torch.sign(hidden)
        output = activated.sum(dim=-1) / np.sqrt(self.k)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return flattened weight matrix."""
        return self.W.flatten()

    def get_weight_vectors(self) -> torch.Tensor:
        """Return weight matrix (K, d)."""
        return self.W

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """
        Compute order parameters for committee machine.

        Returns overlap matrices Q (student-student) and M (student-teacher).
        """
        W0 = teacher_params.get("W0")  # Teacher weights (K0, d) or (d, 1)
        teacher_params.get("k", 1)  # Teacher committee size

        # Student self-overlap matrix Q: Q_ij = (1/d) * W_i^T @ W_j
        Q = self.W @ self.W.T / self.d

        result = {
            "Q": Q.detach().cpu().numpy().tolist(),
            "q_diag": torch.diag(Q).mean().item(),  # Average self-norm
        }

        if W0 is not None:
            if W0.dim() == 2 and W0.shape[0] > 1:
                # Multi-output teacher
                # M_ij = (1/d) * W_i^T @ W0_j
                M = self.W @ W0.T / self.d
                result["M"] = M.detach().cpu().numpy().tolist()
                result["m_avg"] = M.mean().item()
            else:
                # Single output teacher
                W0_flat = W0.flatten()
                m_vec = self.W @ W0_flat / self.d
                result["m_vec"] = m_vec.detach().cpu().numpy().tolist()
                result["m_avg"] = m_vec.mean().item()

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "init_scale": self.init_scale,
                "init_method": self.init_method,
            }
        )
        return config


class SoftCommitteeMachine(BaseModel):
    """
    Soft committee machine with differentiable activation.

    Architecture: y = (1/sqrt(K)) * sum_k g(W_k^T @ x / sqrt(d))

    where g is a differentiable activation (erf, tanh, sigmoid).

    This model allows gradient-based training while maintaining
    the committee machine structure.
    """

    def __init__(
        self,
        d: int,
        k: int = 2,
        activation: str = "erf",
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize SoftCommitteeMachine.

        Args:
            d: Input dimension.
            k: Number of hidden units.
            activation: Activation function ('erf', 'tanh', 'sigmoid', 'relu').
            init_scale: Scale for weight initialization.
            init_method: Initialization method.

        """
        super().__init__(d=d, **kwargs)

        self.k = k
        self.activation_name = activation
        self.init_scale = init_scale
        self.init_method = init_method

        self.W = nn.Parameter(torch.empty(k, d))
        self._init_weights()

        # Set activation function
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> callable:
        """Get activation function by name."""
        if name == "erf":
            return lambda x: torch.erf(x / np.sqrt(2))
        elif name == "tanh":
            return torch.tanh
        elif name == "sigmoid":
            return torch.sigmoid
        elif name == "relu":
            return torch.relu
        elif name == "softplus":
            return nn.functional.softplus
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self) -> None:
        """Initialize weights."""
        if self.init_method == "normal":
            nn.init.normal_(self.W, mean=0.0, std=self.init_scale)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.W, gain=self.init_scale)
        elif self.init_method == "xavier":
            nn.init.xavier_normal_(self.W, gain=self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        hidden = x @ self.W.T / np.sqrt(self.d)
        activated = self.activation(hidden)
        output = activated.sum(dim=-1) / np.sqrt(self.k)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return flattened weights."""
        return self.W.flatten()

    def get_weight_vectors(self) -> torch.Tensor:
        """Return weight matrix."""
        return self.W

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """Compute order parameters."""
        W0 = teacher_params.get("W0")

        Q = self.W @ self.W.T / self.d

        result = {
            "Q": Q.detach().cpu().numpy().tolist(),
            "q_diag": torch.diag(Q).mean().item(),
        }

        if W0 is not None:
            if W0.dim() == 2 and W0.shape[0] > 1:
                M = self.W @ W0.T / self.d
                result["M"] = M.detach().cpu().numpy().tolist()
                result["m_avg"] = M.mean().item()
            else:
                W0_flat = W0.flatten()
                m_vec = self.W @ W0_flat / self.d
                result["m_vec"] = m_vec.detach().cpu().numpy().tolist()
                result["m_avg"] = m_vec.mean().item()

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "activation": self.activation_name,
                "init_scale": self.init_scale,
                "init_method": self.init_method,
            }
        )
        return config
