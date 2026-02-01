"""
In-Context Learning (ICL) Datasets.

Datasets for studying in-context learning in transformers and related models.
Includes task-based data generation where each context contains examples
from a randomly sampled task (teacher).

All models follow the scaling convention: z = w^T x / √d = O(1).

References:
- Mahankali et al. (2024): One Step of Gradient Descent is Provably the Optimal In-Context Learner
- Lu et al. (2024): Asymptotic theory of in-context learning by linear attention
- Zhang et al. (2025): Training Dynamics of In-Context Learning in Linear Attention
"""

from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np

from .base import BaseDataset


class ICLLinearRegressionDataset(BaseDataset):
    """
    In-Context Learning dataset for linear regression tasks.
    
    Each task has a randomly sampled teacher weight θ ~ N(0, I_d).
    Context contains ℓ examples: (x_t, y_t) where y_t = θ^T x_t / √d + noise.
    Query x_{ℓ+1} is provided to predict y_{ℓ+1}.
    
    This is the canonical setup for analyzing Linear Self-Attention (LSA).
    
    Data format:
        - context_x: (batch, context_len, d) - context inputs
        - context_y: (batch, context_len) - context labels  
        - query_x: (batch, d) - query input
        - query_y: (batch,) - query target
    
    Args:
        d: Input dimension
        context_len: Number of examples in context (ℓ)
        noise_std: Standard deviation of observation noise (σ)
        task_prior_std: Standard deviation of task prior (default 1.0)
        device: Device for tensors
    """
    
    def __init__(
        self,
        d: int,
        context_len: int = 10,
        noise_std: float = 0.1,
        task_prior_std: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.context_len = context_len
        self.noise_std = noise_std
        self.task_prior_std = task_prior_std
        
    def generate_sample(self) -> Tuple[torch.Tensor, ...]:
        """
        Generate a single ICL sample (one task).
        
        Returns:
            context_x: (context_len, d) - context inputs
            context_y: (context_len,) - context labels
            query_x: (d,) - query input
            query_y: scalar - query target
            theta: (d,) - true task parameter
        """
        theta = torch.randn(self.d, device=self.device) * self.task_prior_std
        
        context_x = torch.randn(self.context_len, self.d, device=self.device)
        context_y = (context_x @ theta) / np.sqrt(self.d)
        if self.noise_std > 0:
            context_y = context_y + self.noise_std * torch.randn(self.context_len, device=self.device)
        
        query_x = torch.randn(self.d, device=self.device)
        query_y = (query_x @ theta) / np.sqrt(self.d)
        if self.noise_std > 0:
            query_y = query_y + self.noise_std * torch.randn(1, device=self.device).squeeze()
        
        return context_x, context_y, query_x, query_y, theta
    
    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of ICL samples.
        
        Args:
            batch_size: Number of tasks to sample
            
        Returns:
            Dictionary with keys:
                - context_x: (batch, context_len, d)
                - context_y: (batch, context_len)
                - query_x: (batch, d)
                - query_y: (batch,)
                - theta: (batch, d) - true task parameters
        """
        context_x = torch.randn(batch_size, self.context_len, self.d, device=self.device)
        query_x = torch.randn(batch_size, self.d, device=self.device)
        
        theta = torch.randn(batch_size, self.d, device=self.device) * self.task_prior_std
        
        context_y = torch.einsum('btd,bd->bt', context_x, theta) / np.sqrt(self.d)
        query_y = torch.einsum('bd,bd->b', query_x, theta) / np.sqrt(self.d)
        
        if self.noise_std > 0:
            context_y = context_y + self.noise_std * torch.randn_like(context_y)
            query_y = query_y + self.noise_std * torch.randn_like(query_y)
        
        return {
            'context_x': context_x,
            'context_y': context_y,
            'query_x': query_x,
            'query_y': query_y,
            'theta': theta,
        }
    
    def generate_dataset(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset in standard format for compatibility."""
        batch = self.generate_batch(n_samples)
        X = torch.cat([
            batch['context_x'].reshape(n_samples, -1),
            batch['query_x']
        ], dim=1)
        y = batch['query_y']
        return X, y
    
    def get_teacher_params(self) -> Dict[str, Any]:
        return {
            'context_len': self.context_len,
            'noise_std': self.noise_std,
            'task_prior_std': self.task_prior_std,
        }
    
    def __repr__(self) -> str:
        return (f"ICLLinearRegressionDataset(d={self.d}, context_len={self.context_len}, "
                f"noise_std={self.noise_std}, task_prior_std={self.task_prior_std})")


class ICLNonlinearRegressionDataset(BaseDataset):
    """
    ICL dataset with nonlinear teacher (e.g., 2-layer network).
    
    Teacher: y = (1/√K) Σ_k a_k φ(w_k^T x / √d) + noise
    
    Args:
        d: Input dimension
        k: Number of hidden units in teacher
        context_len: Number of examples in context
        noise_std: Observation noise
        activation: Teacher activation ('relu', 'tanh', 'erf', 'sigmoid')
        device: Device for tensors
    """
    
    def __init__(
        self,
        d: int,
        k: int = 5,
        context_len: int = 10,
        noise_std: float = 0.1,
        activation: str = 'relu',
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.k = k
        self.context_len = context_len
        self.noise_std = noise_std
        self.activation = activation
        
        if activation == 'relu':
            self.phi = torch.relu
        elif activation == 'tanh':
            self.phi = torch.tanh
        elif activation == 'erf':
            self.phi = lambda x: torch.erf(x / np.sqrt(2))
        elif activation == 'sigmoid':
            self.phi = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _teacher_forward(self, x: torch.Tensor, W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute teacher output: y = (1/√k) Σ_k a_k φ(w_k^T x / √d)"""
        h = self.phi(x @ W.T / np.sqrt(self.d))
        y = (h @ a) / np.sqrt(self.k)
        return y
    
    def generate_sample(self) -> Tuple[torch.Tensor, ...]:
        """Generate a single ICL sample with nonlinear teacher."""
        W = torch.randn(self.k, self.d, device=self.device)
        a = torch.randn(self.k, device=self.device)
        
        context_x = torch.randn(self.context_len, self.d, device=self.device)
        context_y = self._teacher_forward(context_x, W, a)
        
        query_x = torch.randn(self.d, device=self.device)
        query_y = self._teacher_forward(query_x.unsqueeze(0), W, a).squeeze()
        
        if self.noise_std > 0:
            context_y = context_y + self.noise_std * torch.randn_like(context_y)
            query_y = query_y + self.noise_std * torch.randn(1, device=self.device).squeeze()
        
        return context_x, context_y, query_x, query_y, W, a
    
    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of ICL samples with nonlinear teacher."""
        W = torch.randn(batch_size, self.k, self.d, device=self.device)
        a = torch.randn(batch_size, self.k, device=self.device)
        
        context_x = torch.randn(batch_size, self.context_len, self.d, device=self.device)
        query_x = torch.randn(batch_size, self.d, device=self.device)
        
        context_y = torch.zeros(batch_size, self.context_len, device=self.device)
        for b in range(batch_size):
            context_y[b] = self._teacher_forward(context_x[b], W[b], a[b])
        
        query_y = torch.zeros(batch_size, device=self.device)
        for b in range(batch_size):
            query_y[b] = self._teacher_forward(query_x[b].unsqueeze(0), W[b], a[b]).squeeze()
        
        if self.noise_std > 0:
            context_y = context_y + self.noise_std * torch.randn_like(context_y)
            query_y = query_y + self.noise_std * torch.randn_like(query_y)
        
        return {
            'context_x': context_x,
            'context_y': context_y,
            'query_x': query_x,
            'query_y': query_y,
            'W': W,
            'a': a,
        }
    
    def generate_dataset(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = torch.cat([
            batch['context_x'].reshape(n_samples, -1),
            batch['query_x']
        ], dim=1)
        y = batch['query_y']
        return X, y
    
    def get_teacher_params(self) -> Dict[str, Any]:
        return {
            'k': self.k,
            'context_len': self.context_len,
            'noise_std': self.noise_std,
            'activation': self.activation,
        }
    
    def __repr__(self) -> str:
        return (f"ICLNonlinearRegressionDataset(d={self.d}, k={self.k}, "
                f"context_len={self.context_len}, activation={self.activation})")
