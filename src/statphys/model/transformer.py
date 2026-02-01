"""
Transformer models for statistical mechanics analysis.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from statphys.model.base import BaseModel


class SingleLayerAttention(BaseModel):
    """
    Single-layer attention mechanism.

    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    For sequence-to-scalar tasks common in statistical mechanics studies.
    """

    def __init__(
        self,
        d: int,
        d_model: int = 64,
        n_heads: int = 1,
        init_scale: float = 1.0,
        **kwargs: Any,
    ):
        """
        Initialize SingleLayerAttention.

        Args:
            d: Input dimension (also sequence length for simple cases).
            d_model: Model dimension.
            n_heads: Number of attention heads.
            init_scale: Scale for initialization.
        """
        super().__init__(d=d, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.init_scale = init_scale

        # Query, Key, Value projections
        self.W_q = nn.Parameter(torch.empty(d_model, d))
        self.W_k = nn.Parameter(torch.empty(d_model, d))
        self.W_v = nn.Parameter(torch.empty(d_model, d))

        # Output projection
        self.W_o = nn.Parameter(torch.empty(1, d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        scale = self.init_scale / np.sqrt(self.d)
        nn.init.normal_(self.W_q, mean=0.0, std=scale)
        nn.init.normal_(self.W_k, mean=0.0, std=scale)
        nn.init.normal_(self.W_v, mean=0.0, std=scale)
        nn.init.normal_(self.W_o, mean=0.0, std=self.init_scale / np.sqrt(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d) or (seq_len, d) or (d,).

        Returns:
            Output scalar or (batch,) tensor.
        """
        # Handle different input shapes
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, d)

        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = x @ self.W_q.T  # (batch, seq, d_model)
        K = x @ self.W_k.T
        V = x @ self.W_v.T

        # Multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        context = attn @ V  # (batch, n_heads, seq, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Pool over sequence (mean pooling)
        pooled = context.mean(dim=1)  # (batch, d_model)

        # Output projection
        output = pooled @ self.W_o.T  # (batch, 1)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return concatenated QKV weights."""
        return torch.cat([self.W_q.flatten(), self.W_k.flatten(), self.W_v.flatten()])

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor.

        Returns:
            Attention weights tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q.T
        K = x @ self.W_k.T

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        return attn

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """Compute order parameters for attention model."""
        result = {
            "q_norm": (self.W_q ** 2).sum().item() / self.W_q.numel(),
            "k_norm": (self.W_k ** 2).sum().item() / self.W_k.numel(),
            "v_norm": (self.W_v ** 2).sum().item() / self.W_v.numel(),
            "o_norm": (self.W_o ** 2).sum().item() / self.W_o.numel(),
        }
        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "init_scale": self.init_scale,
        })
        return config


class SingleLayerTransformer(BaseModel):
    """
    Single-layer transformer block.

    Architecture:
        h = x + Attention(x)
        y = h + MLP(h)

    Simplified transformer for statistical mechanics analysis.
    """

    def __init__(
        self,
        d: int,
        d_model: int = 64,
        d_ff: int = 256,
        n_heads: int = 4,
        activation: str = "gelu",
        init_scale: float = 1.0,
        use_layer_norm: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize SingleLayerTransformer.

        Args:
            d: Input dimension.
            d_model: Model dimension.
            d_ff: Feed-forward hidden dimension.
            n_heads: Number of attention heads.
            activation: Activation in feed-forward ('gelu', 'relu').
            init_scale: Initialization scale.
            use_layer_norm: Whether to use layer normalization.
        """
        super().__init__(d=d, **kwargs)

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.init_scale = init_scale
        self.use_layer_norm = use_layer_norm
        self.activation_name = activation

        # Input projection
        self.input_proj = nn.Linear(d, d_model, bias=False)

        # Attention
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1, bias=False)

        # Layer norms
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = self._get_activation(activation)

        self._init_weights()

    def _get_activation(self, name: str) -> callable:
        """Get activation function."""
        if name == "gelu":
            return F.gelu
        elif name == "relu":
            return F.relu
        elif name == "silu":
            return F.silu
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self) -> None:
        """Initialize weights."""
        scale = self.init_scale / np.sqrt(self.d_model)
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(module.weight, mean=0.0, std=scale)

        nn.init.normal_(self.ff1.weight, mean=0.0, std=self.init_scale / np.sqrt(self.d_model))
        nn.init.normal_(self.ff2.weight, mean=0.0, std=self.init_scale / np.sqrt(self.d_ff))
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.init_scale / np.sqrt(self.d))
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.init_scale / np.sqrt(self.d_model))

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head attention."""
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = attn @ V

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)

    def _feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward network."""
        return self.ff2(self.activation(self.ff1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output scalar or (batch,) tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        # Input projection
        h = self.input_proj(x)

        # Attention block with residual
        if self.use_layer_norm:
            h = h + self._attention(self.ln1(h))
            h = h + self._feedforward(self.ln2(h))
        else:
            h = h + self._attention(h)
            h = h + self._feedforward(h)

        # Pool and project
        pooled = h.mean(dim=1)
        output = self.output_proj(pooled)

        return output.squeeze()

    def get_weight_vector(self) -> torch.Tensor:
        """Return attention weights."""
        return torch.cat([
            self.W_q.weight.flatten(),
            self.W_k.weight.flatten(),
            self.W_v.weight.flatten(),
        ])

    def compute_order_params(
        self,
        teacher_params: Dict[str, Any],
        include_generalization_error: bool = True,
    ) -> Dict[str, float]:
        """Compute order parameters."""
        result = {
            "attn_q_norm": (self.W_q.weight ** 2).mean().item(),
            "attn_k_norm": (self.W_k.weight ** 2).mean().item(),
            "attn_v_norm": (self.W_v.weight ** 2).mean().item(),
            "ff1_norm": (self.ff1.weight ** 2).mean().item(),
            "ff2_norm": (self.ff2.weight ** 2).mean().item(),
        }
        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_heads": self.n_heads,
            "activation": self.activation_name,
            "use_layer_norm": self.use_layer_norm,
            "init_scale": self.init_scale,
        })
        return config
