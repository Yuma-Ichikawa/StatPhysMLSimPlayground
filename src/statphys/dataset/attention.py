"""
Attention-Indexed Datasets.

Datasets where attention matrices serve as indices for the output.
Key models for Bayes-optimal analysis of Transformer learning.

All models follow the scaling convention: z = w^T x / √d = O(1).

References:
- AIM: arXiv 2025 (Bayes optimal learning of attention-indexed models)

"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDataset


class AttentionIndexedModelDataset(BaseDataset):
    """
    Attention-Indexed Model (AIM) Dataset (arXiv 2025).

    Generates data where attention matrices serve as "indices" for the output.
    Key model for Bayes-optimal analysis of Transformer learning.

    Data generation:
        1. Sample tokens: x_a ~ N(0, I_d) for a = 1,...,T
        2. Sample attention matrices: S_ℓ ~ P_S (GOE, low-rank, etc.)
        3. Compute attention indices:
           h^{(ℓ)}_{ab} = (x_a^T S_ℓ x_b - δ_{ab} Tr(S_ℓ)) / √d
        4. Output: y = g({h^{(ℓ)}})

    Args:
        d: Embedding dimension
        n_tokens: Number of tokens (T)
        n_indices: Number of attention indices (L)
        attention_prior: Prior for S_ℓ ('goe', 'lowrank', 'sparse', 'identity')
        attention_rank: Rank for low-rank prior
        output_fn: Output function g ('linear', 'sign', 'softmax', 'trace')
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        n_tokens: int = 10,
        n_indices: int = 3,
        attention_prior: str = "goe",
        attention_rank: int = 5,
        output_fn: str = "linear",
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.n_tokens = n_tokens
        self.n_indices = n_indices
        self.attention_prior = attention_prior
        self.attention_rank = attention_rank
        self.output_fn = output_fn

        # Sample teacher attention matrices
        self.S_teacher = self._sample_attention_matrices()

    def _sample_attention_matrices(self) -> list[torch.Tensor]:
        """Sample attention matrices from prior."""
        matrices = []

        for _ in range(self.n_indices):
            if self.attention_prior == "goe":
                M = torch.randn(self.d, self.d, device=self.device)
                S = (M + M.T) / (2 * np.sqrt(self.d))

            elif self.attention_prior == "lowrank":
                U = torch.randn(self.d, self.attention_rank, device=self.device)
                S = U @ U.T / self.d

            elif self.attention_prior == "sparse":
                sparsity = 0.1
                M = torch.randn(self.d, self.d, device=self.device)
                mask = torch.rand(self.d, self.d, device=self.device) < sparsity
                M = M * mask.float()
                S = (M + M.T) / (2 * np.sqrt(self.d * sparsity))

            elif self.attention_prior == "identity":
                S = torch.eye(self.d, device=self.device) / self.d

            else:
                raise ValueError(f"Unknown attention prior: {self.attention_prior}")

            matrices.append(S)

        return matrices

    def _compute_attention_index(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Compute attention index h_{ab} = (x_a^T S x_b - δ_{ab} Tr(S)) / √d."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, T, d = x.shape

        Sx = torch.einsum("btd,de->bte", x, S)
        h = torch.einsum("btd,bsd->bts", x, Sx) / np.sqrt(d)

        trace_S = torch.trace(S)
        h = h - trace_S / np.sqrt(d) * torch.eye(T, device=self.device)

        if squeeze:
            h = h.squeeze(0)

        return h

    def _apply_output_fn(self, h_list: list[torch.Tensor]) -> torch.Tensor:
        """Apply output function to attention indices."""
        h_stack = torch.stack(h_list, dim=1)

        if self.output_fn == "linear":
            y = h_stack.sum(dim=1)
            return y.reshape(y.shape[0], -1)

        elif self.output_fn == "sign":
            y = torch.sign(h_stack.sum(dim=1))
            return y.reshape(y.shape[0], -1)

        elif self.output_fn == "softmax":
            y = F.softmax(h_stack.sum(dim=1), dim=-1)
            return y.reshape(y.shape[0], -1)

        elif self.output_fn == "trace":
            traces = []
            for l in range(len(h_list)):
                tr = torch.diagonal(h_list[l], dim1=-2, dim2=-1).sum(dim=-1)
                traces.append(tr)
            return torch.stack(traces, dim=-1)

        else:
            raise ValueError(f"Unknown output function: {self.output_fn}")

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        """Generate a single sample."""
        x = torch.randn(self.n_tokens, self.d, device=self.device)
        h_list = [self._compute_attention_index(x, S) for S in self.S_teacher]
        y = self._apply_output_fn([h.unsqueeze(0) for h in h_list]).squeeze(0)
        return x, y, h_list

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate a batch of samples."""
        x = torch.randn(batch_size, self.n_tokens, self.d, device=self.device)
        h_list = [self._compute_attention_index(x, S) for S in self.S_teacher]
        y = self._apply_output_fn(h_list)

        return {
            "x": x,
            "y": y,
            "attention_indices": torch.stack(h_list, dim=1),
            "S_teacher": self.S_teacher,
        }

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["x"].reshape(n_samples, -1)
        y = batch["y"]
        if y.dim() > 1:
            y = y[:, 0]
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "n_indices": self.n_indices,
            "attention_prior": self.attention_prior,
            "attention_rank": self.attention_rank,
            "output_fn": self.output_fn,
        }

    def __repr__(self) -> str:
        return (
            f"AttentionIndexedModelDataset(d={self.d}, n_tokens={self.n_tokens}, "
            f"n_indices={self.n_indices}, attention_prior={self.attention_prior})"
        )
