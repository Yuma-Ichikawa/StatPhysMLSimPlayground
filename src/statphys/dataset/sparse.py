"""Sparse dataset implementations."""

from typing import Any

import numpy as np
import torch

from statphys.dataset.base import BaseDataset


class SparseDataset(BaseDataset):
    """
    Dataset with sparse inputs.

    The input distribution is:
        x_i ~ Bernoulli(p) * N(0, 1/p)

    This maintains E[x_i^2] = 1 while having sparse support.

    Attributes:
        d: Input dimension.
        sparsity: Sparsity level (fraction of non-zero elements).
        rho: Teacher weight norm.
        eta: Noise variance.

    """

    def __init__(
        self,
        d: int,
        sparsity: float = 0.1,
        rho: float = 1.0,
        eta: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize SparseDataset.

        Args:
            d: Input dimension.
            sparsity: Fraction of non-zero elements (0 < sparsity <= 1).
            rho: Teacher weight norm. Defaults to 1.0.
            eta: Noise variance. Defaults to 0.0.
            W0: Teacher weights. If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        assert 0 < sparsity <= 1, "Sparsity must be in (0, 1]"

        self.sparsity = sparsity
        self.rho = rho
        self.eta = eta

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
            "sparsity": self.sparsity,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single sparse (x, y) sample.

        Returns:
            Tuple of sparse input x (d,) and output y (scalar).

        """
        # Generate sparse mask
        mask = torch.bernoulli(
            torch.full((self.d,), self.sparsity, device=self.device, dtype=self.dtype)
        )

        # Generate Gaussian values scaled to maintain unit variance
        gaussian = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = mask * gaussian / np.sqrt(self.sparsity)

        # Teacher output
        signal = (self.W0.T @ x / np.sqrt(self.d)).squeeze()

        if self.eta > 0:
            noise = torch.randn(1, device=self.device, dtype=self.dtype).squeeze() * np.sqrt(
                self.eta
            )
            y = signal + noise
        else:
            y = signal

        return x, y

    def get_teacher_params(self) -> dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update(
            {
                "sparsity": self.sparsity,
                "rho": self.rho,
                "eta": self.eta,
            }
        )
        return config


class BernoulliGaussianDataset(BaseDataset):
    """
    Dataset with Bernoulli-Gaussian (spike-and-slab) inputs.

    The input distribution is:
        x_i = b_i * g_i
        where b_i ~ Bernoulli(p), g_i ~ N(mu, sigma^2)

    Useful for studying sparse regression problems.
    """

    def __init__(
        self,
        d: int,
        p: float = 0.1,
        mu: float = 0.0,
        sigma: float = 1.0,
        rho: float = 1.0,
        eta: float = 0.0,
        sparse_teacher: bool = False,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize BernoulliGaussianDataset.

        Args:
            d: Input dimension.
            p: Bernoulli probability (sparsity).
            mu: Gaussian mean.
            sigma: Gaussian standard deviation.
            rho: Teacher weight norm.
            eta: Noise variance.
            sparse_teacher: If True, teacher is also sparse.
            W0: Teacher weights.
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.eta = eta
        self.sparse_teacher = sparse_teacher

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            if sparse_teacher:
                # Sparse teacher
                mask = torch.bernoulli(torch.full((d, 1), p, device=self.device, dtype=self.dtype))
                self.W0 = (
                    mask
                    * torch.randn(d, 1, device=self.device, dtype=self.dtype)
                    * np.sqrt(rho / p)
                )
            else:
                self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
            "p": self.p,
            "sparse_teacher": self.sparse_teacher,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a single Bernoulli-Gaussian sample."""
        # Bernoulli mask
        mask = torch.bernoulli(torch.full((self.d,), self.p, device=self.device, dtype=self.dtype))

        # Gaussian component
        gaussian = self.mu + self.sigma * torch.randn(self.d, device=self.device, dtype=self.dtype)

        x = mask * gaussian

        # Teacher output
        signal = (self.W0.T @ x / np.sqrt(self.d)).squeeze()

        if self.eta > 0:
            noise = torch.randn(1, device=self.device, dtype=self.dtype).squeeze() * np.sqrt(
                self.eta
            )
            y = signal + noise
        else:
            y = signal

        return x, y

    def get_teacher_params(self) -> dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params
