"""Small but realistic decoder Transformer with instrumented branches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = inputs.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (inputs * scale.to(inputs.dtype)) * self.weight


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, activation: str) -> None:
        super().__init__()
        self.activation = activation.lower()
        self.hidden = int(hidden)
        if self.activation == "none" or hidden <= 0:
            self.up = None
            self.down = None
            return
        multiplier = 2 if self.activation in {"geglu", "swiglu"} else 1
        self.up = nn.Linear(width, multiplier * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.up is None or self.down is None:
            empty = inputs.new_zeros((*inputs.shape[:-1], 1))
            return torch.zeros_like(inputs), empty, empty
        projected = self.up(inputs)
        gate = inputs.new_zeros((*inputs.shape[:-1], 1))
        if self.activation == "linear":
            hidden = projected
        elif self.activation == "relu":
            hidden = F.relu(projected)
        elif self.activation == "gelu":
            hidden = F.gelu(projected)
        elif self.activation in {"geglu", "swiglu"}:
            value, gate = projected.chunk(2, dim=-1)
            activated = F.gelu(gate) if self.activation == "geglu" else F.silu(gate)
            hidden = value * activated
        else:
            raise ValueError(f"unsupported feed-forward activation: {self.activation}")
        return self.down(hidden), hidden, gate


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        reshape = lambda tensor: tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = map(reshape, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=inputs.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        context = (attention @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(context), attention


def _normalization(name: str, width: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name.endswith("rmsnorm"):
        return RMSNorm(width)
    return nn.LayerNorm(width)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        hidden: int,
        activation: str,
        normalization: str,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.residual_scale = residual_scale
        self.attention = CausalAttention(width, heads)
        self.mlp = FeedForward(width, hidden, activation)
        self.norm1 = _normalization(normalization, width)
        self.norm2 = _normalization(normalization, width)

    @property
    def is_pre_norm(self) -> bool:
        return self.normalization.startswith("pre_")

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        ablate_attention: bool = False,
        ablate_mlp: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attention_input = self.norm1(inputs) if self.is_pre_norm else inputs
        attention_branch, attention_map = self.attention(attention_input)
        applied_attention = torch.zeros_like(attention_branch) if ablate_attention else attention_branch
        hidden = inputs + self.residual_scale * applied_attention
        if not self.is_pre_norm and self.normalization != "none":
            hidden = self.norm1(hidden)
        mlp_input = self.norm2(hidden) if self.is_pre_norm else hidden
        mlp_branch, mlp_activation, mlp_gate = self.mlp(mlp_input)
        applied_mlp = torch.zeros_like(mlp_branch) if ablate_mlp else mlp_branch
        output = hidden + self.residual_scale * applied_mlp
        if not self.is_pre_norm and self.normalization != "none":
            output = self.norm2(output)
        return output, {
            "attention": attention_map,
            "attention_branch": attention_branch,
            "mlp_branch": mlp_branch,
            "mlp_activation": mlp_activation,
            "mlp_gate": mlp_gate,
            "mlp_input": mlp_input,
            "output": output,
        }


@dataclass(frozen=True)
class TransformerConfig:
    vocabulary: int = 258
    width: int = 64
    sequence_length: int = 64
    heads: int = 4
    layers: int = 2
    ff_ratio: float = 4.0
    activation: str = "gelu"
    normalization: str = "pre_rmsnorm"
    residual_scale: float = 1.0
    tie_embeddings: bool = True


class PhaseTensorTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = 0 if config.activation == "none" else max(1, int(round(config.ff_ratio * config.width)))
        self.token_embedding = nn.Embedding(config.vocabulary, config.width)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_length, config.width))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.width,
                config.heads,
                hidden,
                config.activation,
                config.normalization,
                config.residual_scale / math.sqrt(max(config.layers, 1)),
            )
            for _ in range(config.layers)
        )
        self.final_norm = RMSNorm(config.width)
        self.readout = nn.Linear(config.width, config.vocabulary, bias=False)
        if config.tie_embeddings:
            self.readout.weight = self.token_embedding.weight
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        ablate_attention: bool = False,
        ablate_mlp: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        hidden = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        diagnostics: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            hidden, block_diagnostics = block(
                hidden,
                ablate_attention=ablate_attention,
                ablate_mlp=ablate_mlp,
            )
            diagnostics.append(block_diagnostics)
        logits = self.readout(self.final_norm(hidden))
        if not return_diagnostics:
            return logits
        return logits, {
            "attention": torch.stack([item["attention"] for item in diagnostics]),
            "attention_branch": torch.stack([item["attention_branch"] for item in diagnostics]),
            "mlp_branch": torch.stack([item["mlp_branch"] for item in diagnostics]),
            "mlp_activation": torch.stack([item["mlp_activation"] for item in diagnostics]),
            "mlp_gate": torch.stack([item["mlp_gate"] for item in diagnostics]),
            "mlp_input": torch.stack([item["mlp_input"] for item in diagnostics]),
            "layer_representation": torch.stack([item["output"] for item in diagnostics]),
            "representation": hidden,
        }
