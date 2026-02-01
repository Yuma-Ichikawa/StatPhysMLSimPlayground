"""Structured dataset implementations with correlated inputs."""

from typing import Any

import numpy as np
import torch

from statphys.dataset.base import BaseDataset


class StructuredDataset(BaseDataset):
    """
    Dataset with structured (non-iid) inputs.

    The covariance structure is specified by a covariance matrix Sigma:
        x ~ N(0, Sigma)

    Useful for studying effects of input correlations.
    """

    def __init__(
        self,
        d: int,
        cov_matrix: torch.Tensor | None = None,
        rho: float = 1.0,
        eta: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize StructuredDataset.

        Args:
            d: Input dimension.
            cov_matrix: Covariance matrix (d, d). If None, uses identity.
            rho: Teacher weight norm.
            eta: Noise variance.
            W0: Teacher weights.
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.rho = rho
        self.eta = eta

        # Set covariance matrix
        if cov_matrix is not None:
            self.cov_matrix = cov_matrix.to(device=self.device, dtype=self.dtype)
        else:
            self.cov_matrix = torch.eye(d, device=self.device, dtype=self.dtype)

        # Compute Cholesky decomposition for sampling
        self._cholesky = torch.linalg.cholesky(self.cov_matrix)

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
            "cov_matrix": self.cov_matrix,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a single sample with correlated inputs."""
        # Sample z ~ N(0, I) and transform to x ~ N(0, Sigma)
        z = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = self._cholesky @ z

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


class CorrelatedGaussianDataset(BaseDataset):
    """
    Dataset with exponentially decaying correlations.

    The covariance structure is:
        Sigma_ij = rho^|i-j|

    where rho is the correlation parameter.
    """

    def __init__(
        self,
        d: int,
        correlation: float = 0.5,
        teacher_rho: float = 1.0,
        eta: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize CorrelatedGaussianDataset.

        Args:
            d: Input dimension.
            correlation: Correlation parameter (0 <= correlation < 1).
            teacher_rho: Teacher weight norm.
            eta: Noise variance.
            W0: Teacher weights.
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        assert 0 <= correlation < 1, "Correlation must be in [0, 1)"

        self.correlation = correlation
        self.teacher_rho = teacher_rho
        self.eta = eta

        # Build Toeplitz covariance matrix
        indices = torch.arange(d, device=self.device, dtype=self.dtype)
        self.cov_matrix = correlation ** torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))

        # Cholesky decomposition
        self._cholesky = torch.linalg.cholesky(self.cov_matrix)

        # Initialize teacher
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(teacher_rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.teacher_rho,
            "eta": self.eta,
            "correlation": self.correlation,
            "cov_matrix": self.cov_matrix,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a sample with exponentially correlated inputs."""
        z = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = self._cholesky @ z

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


class SpikedCovarianceDataset(BaseDataset):
    """
    Dataset with spiked covariance model.

    The covariance is:
        Sigma = I + sum_i lambda_i * v_i @ v_i^T

    where v_i are spike directions and lambda_i are spike strengths.
    Useful for studying PCA and related problems.
    """

    def __init__(
        self,
        d: int,
        n_spikes: int = 1,
        spike_strengths: list[float] | None = None,
        spike_directions: torch.Tensor | None = None,
        rho: float = 1.0,
        eta: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize SpikedCovarianceDataset.

        Args:
            d: Input dimension.
            n_spikes: Number of spikes.
            spike_strengths: List of spike strengths. Defaults to [1.0] * n_spikes.
            spike_directions: Spike directions (n_spikes, d). If None, random orthogonal.
            rho: Teacher weight norm.
            eta: Noise variance.
            W0: Teacher weights.
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.n_spikes = n_spikes
        self.rho = rho
        self.eta = eta

        # Set spike strengths
        if spike_strengths is None:
            spike_strengths = [1.0] * n_spikes
        self.spike_strengths = spike_strengths

        # Set spike directions (orthonormal)
        if spike_directions is not None:
            self.spike_directions = spike_directions.to(device=self.device, dtype=self.dtype)
        else:
            # Random orthonormal directions via QR decomposition
            random_matrix = torch.randn(d, n_spikes, device=self.device, dtype=self.dtype)
            Q, _ = torch.linalg.qr(random_matrix)
            self.spike_directions = Q.T  # (n_spikes, d)

        # Build covariance matrix: I + sum_i lambda_i * v_i @ v_i^T
        self.cov_matrix = torch.eye(d, device=self.device, dtype=self.dtype)
        for i in range(n_spikes):
            v = self.spike_directions[i].unsqueeze(1)  # (d, 1)
            self.cov_matrix = self.cov_matrix + spike_strengths[i] * (v @ v.T)

        # Cholesky
        self._cholesky = torch.linalg.cholesky(self.cov_matrix)

        # Teacher
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
            "spike_directions": self.spike_directions,
            "spike_strengths": self.spike_strengths,
            "cov_matrix": self.cov_matrix,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a sample from spiked covariance model."""
        z = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = self._cholesky @ z

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
