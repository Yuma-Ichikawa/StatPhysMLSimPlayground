"""
Random Features and Kernel models.

These models fix the first layer randomly and only train the final linear layer.
This is the basis for kernel ridge regression and neural tangent kernel analysis.
"""

from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import numpy as np

from statphys.model.base import BaseModel


class RandomFeaturesModel(BaseModel):
    """
    Random Features model with fixed random first layer.

    Architecture:
        h(x) = phi((1/sqrt(d)) * B @ x)  where B is fixed random
        f(x) = (1/sqrt(p)) * a^T @ h(x)  where a is learned

    This is equivalent to kernel ridge regression in the limit p -> infinity.

    Attributes:
        d: Input dimension.
        p: Number of random features.
        activation: Activation function.
        B: Fixed random projection matrix (p, d).
        a: Learnable output weights (p,).
    """

    def __init__(
        self,
        d: int,
        p: int = 1000,
        activation: str = "relu",
        init_scale: float = 1.0,
        feature_scale: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize RandomFeaturesModel.

        Args:
            d: Input dimension.
            p: Number of random features.
            activation: Activation function ('relu', 'erf', 'tanh', 'sign', 'linear').
            init_scale: Scale for output weight initialization.
            feature_scale: Scale for random feature matrix B.
            device: Computation device.
            dtype: Data type.
        """
        super().__init__(d=d, **kwargs)

        self.p = p
        self.activation_name = activation
        self.init_scale = init_scale
        self.feature_scale = feature_scale

        # Fixed random projection matrix (not trainable)
        self.register_buffer(
            "B",
            torch.randn(p, d, dtype=dtype) * np.sqrt(feature_scale)
        )

        # Learnable output weights
        self.a = nn.Parameter(
            torch.randn(p, dtype=dtype) * init_scale / np.sqrt(p)
        )

        # Activation function
        self._setup_activation(activation)

    def _setup_activation(self, activation: str) -> None:
        """Setup activation function."""
        if activation == "relu":
            self.activation = torch.relu
        elif activation == "erf":
            self.activation = lambda x: torch.erf(x / np.sqrt(2.0))
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sign":
            self.activation = torch.sign
        elif activation == "linear":
            self.activation = lambda x: x
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")

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
            squeeze_output = True
        else:
            squeeze_output = False

        # h = phi((1/sqrt(d)) * B @ x^T)  shape: (p, batch_size)
        preactivation = self.B @ x.T / np.sqrt(self.d)
        h = self.activation(preactivation)

        # f = (1/sqrt(p)) * a^T @ h  shape: (batch_size,)
        output = (self.a @ h) / np.sqrt(self.p)

        if squeeze_output:
            return output.squeeze(0)
        return output

    def get_weight_vector(self) -> torch.Tensor:
        """Return the learnable output weights."""
        return self.a

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get random feature representation.

        Args:
            x: Input tensor of shape (batch_size, d) or (d,).

        Returns:
            Feature tensor of shape (batch_size, p) or (p,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        preactivation = self.B @ x.T / np.sqrt(self.d)
        h = self.activation(preactivation).T  # (batch_size, p)

        if squeeze_output:
            return h.squeeze(0)
        return h

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """
        Compute order parameters.

        For random features, we track the self-overlap of output weights.

        Args:
            teacher_params: Dictionary containing teacher parameters.
            include_generalization_error: Whether to compute gen error.

        Returns:
            Dictionary of order parameters.
        """
        q = torch.sum(self.a ** 2).item() / self.p

        result = {
            "q": q,
        }

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "p": self.p,
            "activation": self.activation_name,
            "init_scale": self.init_scale,
            "feature_scale": self.feature_scale,
        })
        return config


class KernelRidgeModel(RandomFeaturesModel):
    """
    Kernel Ridge Regression using Random Features approximation.

    This is a convenience wrapper around RandomFeaturesModel that
    emphasizes its connection to kernel methods.
    """

    def __init__(
        self,
        d: int,
        p: int = 2000,
        kernel: str = "rbf",
        init_scale: float = 1.0,
        **kwargs: Any,
    ):
        """
        Initialize KernelRidgeModel.

        Args:
            d: Input dimension.
            p: Number of random features for kernel approximation.
            kernel: Kernel type ('rbf' -> relu, 'ntk' -> relu, 'polynomial' -> linear+squared).
            init_scale: Scale for weight initialization.
        """
        # Map kernel to activation
        kernel_to_activation = {
            "rbf": "relu",
            "ntk": "relu",
            "laplace": "relu",
            "polynomial": "linear",
            "arc_cosine": "relu",
        }

        activation = kernel_to_activation.get(kernel, "relu")

        super().__init__(
            d=d,
            p=p,
            activation=activation,
            init_scale=init_scale,
            **kwargs
        )

        self.kernel_type = kernel


class DeepLinearNetwork(BaseModel):
    """
    Deep Linear Network (identity activation).

    Architecture:
        f(x) = (1/d^{L/2}) * W^{(L)} @ ... @ W^{(2)} @ W^{(1)} @ x

    This is the simplest "deep" network that admits exact statistical mechanics analysis.
    Despite being linear in the end-to-end mapping, the optimization landscape
    and learning dynamics are non-trivial.

    Reference: Saxe et al. "Exact solutions to the nonlinear dynamics of learning
    in deep linear neural networks" (2013)

    Attributes:
        d: Input dimension.
        depth: Number of layers.
        widths: List of layer widths.
    """

    def __init__(
        self,
        d: int,
        depth: int = 3,
        width: Optional[int] = None,
        widths: Optional[List[int]] = None,
        init_scale: float = 1.0,
        init_method: str = "orthogonal",
        **kwargs: Any,
    ):
        """
        Initialize DeepLinearNetwork.

        Args:
            d: Input dimension.
            depth: Number of layers (L).
            width: Width for all hidden layers (if widths not specified).
            widths: List of layer widths [d_1, d_2, ..., d_{L-1}].
                   Input is d_0 = d, output is d_L = 1.
            init_scale: Scale for weight initialization.
            init_method: Initialization method ('orthogonal', 'normal', 'identity').
        """
        super().__init__(d=d, **kwargs)

        self.depth = depth
        self.init_scale = init_scale
        self.init_method = init_method

        # Determine layer widths
        if widths is not None:
            self.widths = [d] + list(widths) + [1]
        elif width is not None:
            self.widths = [d] + [width] * (depth - 1) + [1]
        else:
            # Default: all layers have width d except output
            self.widths = [d] * depth + [1]

        # Create weight matrices
        self.layers = nn.ModuleList()
        for l in range(depth):
            d_in = self.widths[l]
            d_out = self.widths[l + 1]
            layer = nn.Linear(d_in, d_out, bias=False)
            self._init_layer(layer, d_in, d_out)
            self.layers.append(layer)

    def _init_layer(self, layer: nn.Linear, d_in: int, d_out: int) -> None:
        """Initialize layer weights."""
        if self.init_method == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=self.init_scale)
        elif self.init_method == "normal":
            nn.init.normal_(layer.weight, std=self.init_scale / np.sqrt(d_in))
        elif self.init_method == "identity":
            # Initialize close to identity (for square matrices)
            with torch.no_grad():
                if d_in == d_out:
                    layer.weight.copy_(
                        torch.eye(d_out) * self.init_scale + 
                        torch.randn(d_out, d_in) * 0.01
                    )
                else:
                    nn.init.normal_(layer.weight, std=self.init_scale / np.sqrt(d_in))
        else:
            nn.init.normal_(layer.weight, std=self.init_scale / np.sqrt(d_in))

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
            squeeze_output = True
        else:
            squeeze_output = False

        # Apply layers with proper scaling
        h = x / np.sqrt(self.d)  # Initial scaling
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l < len(self.layers) - 1:
                # Scale by 1/sqrt(width) between layers
                h = h / np.sqrt(self.widths[l + 1])

        output = h.squeeze(-1)

        if squeeze_output:
            return output.squeeze(0)
        return output

    def get_weight_vector(self) -> torch.Tensor:
        """
        Return effective end-to-end weight matrix.

        For deep linear: W_eff = W^{(L)} @ ... @ W^{(1)}
        """
        W_eff = self.layers[0].weight
        for l in range(1, len(self.layers)):
            W_eff = self.layers[l].weight @ W_eff
        return W_eff.flatten()

    def get_layer_weights(self) -> List[torch.Tensor]:
        """Return list of weight matrices for each layer."""
        return [layer.weight for layer in self.layers]

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """
        Compute order parameters.

        For deep linear networks, we track:
        - q: self-overlap of effective weight
        - m: overlap with teacher
        - Layer-wise statistics

        Args:
            teacher_params: Dictionary containing teacher parameters.
            include_generalization_error: Whether to compute gen error.

        Returns:
            Dictionary of order parameters.
        """
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)

        # Effective weight
        w_eff = self.get_weight_vector()
        q = torch.sum(w_eff ** 2).item() / self.d

        result = {"q": q}

        if W0 is not None:
            w0_flat = W0.flatten()
            m = torch.dot(w_eff, w0_flat).item() / self.d
            result["m"] = m

            if include_generalization_error:
                eg = 0.5 * (rho - 2 * m + q)
                result["eg"] = eg

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "widths": self.widths,
            "init_scale": self.init_scale,
            "init_method": self.init_method,
        })
        return config
