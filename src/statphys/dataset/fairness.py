"""
Fairness/Bias Analysis Datasets.

Datasets for studying fairness and bias in high-dimensional learning,
with multiple groups having different teachers or distributions.

All models follow the scaling convention: z = w^T x / √d = O(1).

References:
- Teacher-Mixture: ICML 2024 / APS 2025 (fairness/bias analysis)

"""

from typing import Any

import numpy as np
import torch

from .base import BaseDataset


class TeacherMixtureFairnessDataset(BaseDataset):
    """
    Teacher-Mixture (T-M) Model for Fairness Analysis (ICML 2024 / APS 2025).

    Generates data with two groups having different teachers to study
    bias and fairness in high-dimensional learning.

    Data generation:
        1. Sample group: μ ∈ {+, -} with P(+) = ρ
        2. Sample input: x ~ N(±v/√d, Δ_± I_d) where v ~ N(0, I_d)
        3. Sample label: y = sign(W_T^± · x / √d + b_T^±)

    The teacher correlation q_T = E[W_T^+ · W_T^-] / d controls
    how different the two group rules are.

    Args:
        d: Input dimension
        group_ratio: Probability of group + (ρ)
        teacher_correlation: Correlation between group teachers (q_T)
        group_variances: Variances for each group (Δ_+, Δ_-)
        signal_strength: Strength of group mean shift
        teacher_bias: Bias terms for each group (b_T^+, b_T^-)
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        group_ratio: float = 0.5,
        teacher_correlation: float = 0.5,
        group_variances: tuple[float, float] = (1.0, 1.0),
        signal_strength: float = 1.0,
        teacher_bias: tuple[float, float] = (0.0, 0.0),
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.group_ratio = group_ratio
        self.teacher_correlation = teacher_correlation
        self.group_variances = group_variances
        self.signal_strength = signal_strength
        self.teacher_bias = teacher_bias

        self._generate_teacher_weights()

        self.v = torch.randn(d, device=device)
        self.v = self.v / self.v.norm() * np.sqrt(d)

    def _generate_teacher_weights(self):
        """Generate correlated teacher weights for two groups."""
        d = self.d
        q_T = self.teacher_correlation

        self.W_plus = torch.randn(d, device=self.device)
        Z = torch.randn(d, device=self.device)
        self.W_minus = q_T * self.W_plus + np.sqrt(max(0, 1 - q_T**2)) * Z

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        """Generate a single sample."""
        group = 1 if torch.rand(1).item() < self.group_ratio else -1

        if group == 1:
            mean = self.v / np.sqrt(self.d) * self.signal_strength
            std = np.sqrt(self.group_variances[0])
            W_T = self.W_plus
            b_T = self.teacher_bias[0]
        else:
            mean = -self.v / np.sqrt(self.d) * self.signal_strength
            std = np.sqrt(self.group_variances[1])
            W_T = self.W_minus
            b_T = self.teacher_bias[1]

        z = torch.randn(self.d, device=self.device)
        x = mean + std * z

        pre_activation = (W_T @ x) / np.sqrt(self.d) + b_T
        y = torch.sign(pre_activation)

        return x, y, torch.tensor(group, device=self.device)

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate a batch of samples."""
        groups = torch.where(
            torch.rand(batch_size, device=self.device) < self.group_ratio,
            torch.ones(batch_size, device=self.device),
            -torch.ones(batch_size, device=self.device),
        )

        x = torch.zeros(batch_size, self.d, device=self.device)
        y = torch.zeros(batch_size, device=self.device)

        mask_plus = groups == 1
        n_plus = mask_plus.sum().item()
        if n_plus > 0:
            mean_plus = self.v / np.sqrt(self.d) * self.signal_strength
            std_plus = np.sqrt(self.group_variances[0])
            z_plus = torch.randn(n_plus, self.d, device=self.device)
            x[mask_plus] = mean_plus + std_plus * z_plus
            pre_act_plus = (x[mask_plus] @ self.W_plus) / np.sqrt(self.d) + self.teacher_bias[0]
            y[mask_plus] = torch.sign(pre_act_plus)

        mask_minus = groups == -1
        n_minus = mask_minus.sum().item()
        if n_minus > 0:
            mean_minus = -self.v / np.sqrt(self.d) * self.signal_strength
            std_minus = np.sqrt(self.group_variances[1])
            z_minus = torch.randn(n_minus, self.d, device=self.device)
            x[mask_minus] = mean_minus + std_minus * z_minus
            pre_act_minus = (x[mask_minus] @ self.W_minus) / np.sqrt(self.d) + self.teacher_bias[1]
            y[mask_minus] = torch.sign(pre_act_minus)

        return {
            "x": x,
            "y": y,
            "groups": groups,
            "W_plus": self.W_plus,
            "W_minus": self.W_minus,
        }

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        return batch["x"], batch["y"]

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "group_ratio": self.group_ratio,
            "teacher_correlation": self.teacher_correlation,
            "group_variances": self.group_variances,
            "signal_strength": self.signal_strength,
            "teacher_bias": self.teacher_bias,
        }

    def compute_fairness_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
    ) -> dict[str, float]:
        """Compute fairness metrics given predictions."""
        metrics = {}

        for g, name in [(1, "plus"), (-1, "minus")]:
            mask = groups == g
            if mask.any():
                acc = (predictions[mask] == labels[mask]).float().mean().item()
                metrics[f"accuracy_{name}"] = acc

        pred_rate_plus = (
            (predictions[groups == 1] == 1).float().mean().item() if (groups == 1).any() else 0
        )
        pred_rate_minus = (
            (predictions[groups == -1] == 1).float().mean().item() if (groups == -1).any() else 0
        )
        metrics["demographic_parity_gap"] = abs(pred_rate_plus - pred_rate_minus)

        return metrics

    def __repr__(self) -> str:
        return (
            f"TeacherMixtureFairnessDataset(d={self.d}, group_ratio={self.group_ratio}, "
            f"teacher_correlation={self.teacher_correlation})"
        )
