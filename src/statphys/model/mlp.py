"""Multi-layer perceptron models."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from statphys.model.base import BaseModel


class TwoLayerNetwork(BaseModel):
    """
    Two-layer neural network.

    Architecture: y = (1/sqrt(K)) * a^T @ g(W @ x / sqrt(d))

    where:
        - W: (K, d) first layer weights
        - a: (K,) second layer weights (may be fixed or learnable)
        - g: activation function

    This is the canonical model for studying feature learning
    in the lazy/rich regimes.
    """

    def __init__(
        self,
        d: int,
        k: int = 100,
        activation: str = "relu",
        second_layer_fixed: bool = False,
        second_layer_init: str = "ones",
        init_scale: float = 1.0,
        init_method: str = "normal",
        **kwargs: Any,
    ):
        """
        Initialize TwoLayerNetwork.

        Args:
            d: Input dimension.
            k: Number of hidden units.
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'erf').
            second_layer_fixed: If True, don't train second layer.
            second_layer_init: Init for second layer ('ones', 'random', 'alternating').
            init_scale: Scale for weight initialization.
            init_method: Initialization method for first layer.

        """
        super().__init__(d=d, **kwargs)

        self.k = k
        self.activation_name = activation
        self.second_layer_fixed = second_layer_fixed
        self.init_scale = init_scale
        self.init_method = init_method

        # First layer weights: (K, d)
        self.W = nn.Parameter(torch.empty(k, d))

        # Second layer weights: (K,)
        if second_layer_fixed:
            self.register_buffer("a", torch.empty(k))
        else:
            self.a = nn.Parameter(torch.empty(k))

        self._init_weights(second_layer_init)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> callable:
        """Get activation function."""
        if name == "relu":
            return torch.relu
        elif name == "tanh":
            return torch.tanh
        elif name == "sigmoid":
            return torch.sigmoid
        elif name == "erf":
            return lambda x: torch.erf(x / np.sqrt(2))
        elif name == "leaky_relu":
            return nn.functional.leaky_relu
        elif name == "gelu":
            return nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self, second_layer_init: str) -> None:
        """Initialize weights."""
        # First layer initialization
        if self.init_method == "normal":
            nn.init.normal_(self.W, mean=0.0, std=self.init_scale / np.sqrt(self.d))
        elif self.init_method == "xavier":
            nn.init.xavier_normal_(self.W, gain=self.init_scale)
        elif self.init_method == "kaiming":
            nn.init.kaiming_normal_(self.W, nonlinearity="relu")
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.W, gain=self.init_scale)

        # Second layer initialization
        if second_layer_init == "ones":
            nn.init.ones_(self.a)
        elif second_layer_init == "random":
            nn.init.normal_(self.a, mean=0.0, std=1.0)
        elif second_layer_init == "alternating":
            # +1, -1, +1, -1, ...
            self.a.data = torch.tensor([(-1) ** i for i in range(self.k)], dtype=self.W.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).

        Returns:
            Output tensor.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # First layer: (batch, K) = (batch, d) @ (d, K)
        hidden = x @ self.W.T / np.sqrt(self.d)

        # Activation
        activated = self.activation(hidden)

        # Second layer: (batch,) = (batch, K) @ (K,)
        output = (activated @ self.a) / np.sqrt(self.k)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return first layer weights (flattened)."""
        return self.W.flatten()

    def get_all_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both layer weights."""
        return self.W, self.a

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """
        Compute order parameters for two-layer network.

        Computes Q (student-student overlap) and M (student-teacher overlap).
        """
        W0 = teacher_params.get("W0")
        teacher_params.get("a0")  # Teacher second layer

        # Student self-overlap
        Q = self.W @ self.W.T / self.d

        result = {
            "Q_diag_mean": torch.diag(Q).mean().item(),
            "Q_offdiag_mean": (
                (Q.sum() - torch.diag(Q).sum()).item() / (self.k * (self.k - 1))
                if self.k > 1
                else 0.0
            ),
        }

        if W0 is not None:
            if W0.dim() == 2 and W0.shape[0] > 1:
                # Teacher also has hidden layer
                M = self.W @ W0.T / self.d
                result["M_mean"] = M.mean().item()
            else:
                # Linear teacher
                W0_flat = W0.flatten()
                m_vec = self.W @ W0_flat / self.d
                result["m_vec_mean"] = m_vec.mean().item()

        # Second layer norms
        result["a_norm"] = (self.a**2).sum().item() / self.k

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "activation": self.activation_name,
                "second_layer_fixed": self.second_layer_fixed,
                "init_scale": self.init_scale,
                "init_method": self.init_method,
            }
        )
        return config


class TwoLayerNetworkReLU(TwoLayerNetwork):
    """
    Two-layer network with ReLU activation.

    Convenience class with ReLU as default activation.
    """

    def __init__(
        self,
        d: int,
        k: int = 100,
        second_layer_fixed: bool = False,
        second_layer_init: str = "alternating",
        init_scale: float = 1.0,
        init_method: str = "kaiming",
        **kwargs: Any,
    ):
        """
        Initialize TwoLayerNetworkReLU.

        Args:
            d: Input dimension.
            k: Number of hidden units.
            second_layer_fixed: If True, don't train second layer.
            second_layer_init: Initialization for second layer.
            init_scale: Scale for weight initialization.
            init_method: Initialization method.

        """
        super().__init__(
            d=d,
            k=k,
            activation="relu",
            second_layer_fixed=second_layer_fixed,
            second_layer_init=second_layer_init,
            init_scale=init_scale,
            init_method=init_method,
            **kwargs,
        )


class DeepNetwork(BaseModel):
    """
    Deep fully-connected network with L layers.

    Architecture: y = W_L @ g(...g(W_2 @ g(W_1 @ x / sqrt(d_0)) / sqrt(d_1))...) / sqrt(d_{L-1})

    Useful for studying depth effects and feature learning.
    """

    def __init__(
        self,
        d: int,
        hidden_dims: list[int],
        activation: str = "relu",
        init_scale: float = 1.0,
        init_method: str = "kaiming",
        **kwargs: Any,
    ):
        """
        Initialize DeepNetwork.

        Args:
            d: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function.
            init_scale: Scale for weight initialization.
            init_method: Initialization method.

        """
        super().__init__(d=d, **kwargs)

        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.init_scale = init_scale
        self.init_method = init_method
        self.n_layers = len(hidden_dims) + 1

        # Build layers
        dims = [d] + hidden_dims + [1]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], bias=False)
            self.layers.append(layer)

        self._init_weights()
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> callable:
        """Get activation function."""
        activations = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "gelu": nn.functional.gelu,
            "leaky_relu": nn.functional.leaky_relu,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def _init_weights(self) -> None:
        """Initialize all layers."""
        for layer in self.layers:
            if self.init_method == "kaiming":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif self.init_method == "xavier":
                nn.init.xavier_normal_(layer.weight, gain=self.init_scale)
            elif self.init_method == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = x
        for _i, layer in enumerate(self.layers[:-1]):
            h = layer(h) / np.sqrt(layer.in_features)
            h = self.activation(h)

        # Output layer (no activation)
        output = self.layers[-1](h) / np.sqrt(self.layers[-1].in_features)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return first layer weights."""
        return self.layers[0].weight.flatten()

    def get_all_weights(self) -> list[torch.Tensor]:
        """Return all layer weights."""
        return [layer.weight for layer in self.layers]

    def compute_order_params(
        self,
        teacher_params: dict[str, Any],
        include_generalization_error: bool = True,
    ) -> dict[str, float]:
        """Compute order parameters (first layer only for simplicity)."""
        W = self.layers[0].weight
        W0 = teacher_params.get("W0")

        Q = W @ W.T / self.d

        result = {
            "Q_diag_mean": torch.diag(Q).mean().item(),
        }

        if W0 is not None and W0.shape == W.shape:
            M = W @ W0.T / self.d
            result["M_mean"] = M.mean().item()

        return result

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "hidden_dims": self.hidden_dims,
                "activation": self.activation_name,
                "init_scale": self.init_scale,
                "init_method": self.init_method,
            }
        )
        return config
