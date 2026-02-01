"""
Noisy Label Datasets.

Datasets with label noise and class imbalance for studying
robust learning and self-distillation.

All models follow the scaling convention: z = w^T x / √d = O(1).

References:
- Noisy GMM: OpenReview 2025 (self-distillation analysis)
"""

from typing import Optional, Tuple, Dict, Any, List

import torch
import numpy as np

from .base import BaseDataset


class NoisyGMMSelfDistillationDataset(BaseDataset):
    """
    Noisy Gaussian Mixture Model for Self-Distillation Analysis (OpenReview 2025).
    
    Generates data with label noise and class imbalance to study
    multi-stage self-distillation via replica methods.
    
    Data generation:
        1. Sample true label: y_true ~ Bernoulli(ρ)
        2. Flip with probability θ: P(y ≠ y_true) = θ
        3. Sample input: x = (2y_true - 1) * v/√d + √Δ * z
           where z ~ N(0, I_d)
    
    Args:
        d: Input dimension (N in paper)
        class_prior: Probability of class 1 (ρ)
        label_noise: Label flip probability (θ)
        signal_strength: SNR parameter (||v||²/d)
        noise_variance: Input noise variance (Δ)
        device: Device for tensors
    """
    
    def __init__(
        self,
        d: int,
        class_prior: float = 0.5,
        label_noise: float = 0.0,
        signal_strength: float = 1.0,
        noise_variance: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.class_prior = class_prior
        self.label_noise = label_noise
        self.signal_strength = signal_strength
        self.noise_variance = noise_variance
        
        self.v = torch.randn(d, device=device)
        self.v = self.v / self.v.norm() * np.sqrt(d * signal_strength)
    
    def generate_sample(self) -> Tuple[torch.Tensor, ...]:
        """Generate a single sample."""
        y_true = 1 if torch.rand(1).item() < self.class_prior else 0
        
        if torch.rand(1).item() < self.label_noise:
            y_observed = 1 - y_true
        else:
            y_observed = y_true
        
        sign = 2 * y_true - 1
        mean = sign * self.v / np.sqrt(self.d)
        z = torch.randn(self.d, device=self.device)
        x = mean + np.sqrt(self.noise_variance) * z
        
        return (x, 
                torch.tensor(y_observed, dtype=torch.float32, device=self.device),
                torch.tensor(y_true, dtype=torch.float32, device=self.device))
    
    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of samples."""
        y_true = (torch.rand(batch_size, device=self.device) < self.class_prior).float()
        
        flip_mask = torch.rand(batch_size, device=self.device) < self.label_noise
        y_observed = torch.where(flip_mask, 1 - y_true, y_true)
        
        signs = 2 * y_true - 1
        mean = signs.unsqueeze(-1) * self.v / np.sqrt(self.d)
        z = torch.randn(batch_size, self.d, device=self.device)
        x = mean + np.sqrt(self.noise_variance) * z
        
        return {
            'x': x,
            'y': y_observed,
            'y_true': y_true,
            'flipped': flip_mask.float(),
        }
    
    def generate_dataset(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        return batch['x'], batch['y']
    
    def get_teacher_params(self) -> Dict[str, Any]:
        return {
            'class_prior': self.class_prior,
            'label_noise': self.label_noise,
            'signal_strength': self.signal_strength,
            'noise_variance': self.noise_variance,
        }
    
    def compute_distillation_metrics(self, stage_predictions: List[torch.Tensor],
                                      true_labels: torch.Tensor) -> Dict[str, List[float]]:
        """Compute metrics across distillation stages."""
        accuracies = []
        for preds in stage_predictions:
            acc = ((preds > 0.5).float() == true_labels).float().mean().item()
            accuracies.append(acc)
        
        return {'stage_accuracies': accuracies}
    
    def __repr__(self) -> str:
        return (f"NoisyGMMSelfDistillationDataset(d={self.d}, class_prior={self.class_prior}, "
                f"label_noise={self.label_noise}, signal_strength={self.signal_strength})")
