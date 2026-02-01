"""Gaussian dataset implementations for regression and classification."""

from typing import Any

import numpy as np
import torch

from statphys.dataset.base import BaseDataset


class GaussianDataset(BaseDataset):
    """
    Dataset with iid Gaussian inputs and linear teacher (regression).

    The data generation follows:
        x ~ N(0, I_d)
        y = (1/sqrt(d)) * W0^T @ x + noise

    where W0 is the teacher weight vector and noise ~ N(0, eta).

    Attributes:
        d: Input dimension.
        rho: Norm of teacher weights (||W0||^2 / d).
        eta: Noise variance.
        W0: Teacher weight vector.

    """

    def __init__(
        self,
        d: int,
        rho: float = 1.0,
        eta: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize GaussianDataset.

        Args:
            d: Input dimension.
            rho: Teacher weight norm (||W0||^2 / d). Defaults to 1.0.
            eta: Noise variance. Defaults to 0.0 (noiseless).
            W0: Teacher weights. If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.rho = rho
        self.eta = eta

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            # Sample teacher from N(0, rho*I)
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        # Store teacher params
        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and output y (scalar).

        """
        # Generate input x ~ N(0, I)
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)

        # Generate output y = (1/sqrt(d)) * W0^T @ x + noise
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
        return {
            "W0": self.W0,
            "rho": self.rho,
            "eta": self.eta,
        }

    def get_config(self) -> dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "eta": self.eta,
            }
        )
        return config


class GaussianClassificationDataset(BaseDataset):
    """
    Dataset with iid Gaussian inputs and linear teacher (binary classification).

    The data generation follows:
        x ~ N(0, I_d)
        y = sign((1/sqrt(d)) * W0^T @ x)  [noiseless]
        or
        y = sign((1/sqrt(d)) * W0^T @ x + noise)  [noisy]

    Attributes:
        d: Input dimension.
        rho: Norm of teacher weights.
        flip_prob: Label flip probability for noisy classification.
        W0: Teacher weight vector.

    """

    def __init__(
        self,
        d: int,
        rho: float = 1.0,
        flip_prob: float = 0.0,
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize GaussianClassificationDataset.

        Args:
            d: Input dimension.
            rho: Teacher weight norm. Defaults to 1.0.
            flip_prob: Probability of flipping labels. Defaults to 0.0.
            W0: Teacher weights. If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.rho = rho
        self.flip_prob = flip_prob

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "flip_prob": self.flip_prob,
        }

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and label y (+1 or -1).

        """
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)

        # Teacher prediction
        teacher_output = (self.W0.T @ x / np.sqrt(self.d)).squeeze()
        y = torch.sign(teacher_output)

        # Apply label noise
        if self.flip_prob > 0 and torch.rand(1).item() < self.flip_prob:
            y = -y

        # Handle exact zero case
        if y == 0:
            y = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        return x, y

    def get_teacher_params(self) -> dict[str, Any]:
        """Return teacher parameters."""
        return {
            "W0": self.W0,
            "rho": self.rho,
            "flip_prob": self.flip_prob,
        }

    def get_config(self) -> dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "flip_prob": self.flip_prob,
            }
        )
        return config


class GaussianMultiOutputDataset(BaseDataset):
    """
    Dataset with Gaussian inputs and multi-output teacher.

    Supports committee machines and other multi-output architectures.

    The data generation follows:
        x ~ N(0, I_d)
        y = f(W0 @ x / sqrt(d))

    where W0 is (k, d) for k outputs and f is an activation function.
    """

    def __init__(
        self,
        d: int,
        k: int,
        rho: float = 1.0,
        eta: float = 0.0,
        activation: str = "linear",
        aggregation: str = "mean",
        W0: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize GaussianMultiOutputDataset.

        Args:
            d: Input dimension.
            k: Number of hidden units/outputs in teacher.
            rho: Teacher weight norm per unit. Defaults to 1.0.
            eta: Output noise variance. Defaults to 0.0.
            activation: Activation function ('linear', 'relu', 'sign', 'erf').
            aggregation: How to aggregate hidden outputs ('mean', 'sum', 'none').
            W0: Teacher weights (k, d). If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.

        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.k = k
        self.rho = rho
        self.eta = eta
        self.activation = activation
        self.aggregation = aggregation

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(k, d, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "k": self.k,
            "rho": self.rho,
            "eta": self.eta,
            "activation": self.activation,
        }

    def _apply_activation(self, z: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "linear":
            return z
        elif self.activation == "relu":
            return torch.relu(z)
        elif self.activation == "sign":
            return torch.sign(z)
        elif self.activation == "erf":
            return torch.erf(z / np.sqrt(2))
        elif self.activation == "tanh":
            return torch.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and output y (scalar or k-dim).

        """
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)

        # Teacher forward: (k, d) @ (d,) -> (k,)
        preactivation = self.W0 @ x / np.sqrt(self.d)
        hidden = self._apply_activation(preactivation)

        # Aggregate outputs
        if self.aggregation == "mean":
            y = hidden.mean()
        elif self.aggregation == "sum":
            y = hidden.sum() / np.sqrt(self.k)
        elif self.aggregation == "none":
            y = hidden
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Add noise
        if self.eta > 0:
            if isinstance(y, torch.Tensor) and y.dim() > 0:
                noise = torch.randn_like(y) * np.sqrt(self.eta)
            else:
                noise = torch.randn(1, device=self.device, dtype=self.dtype).squeeze() * np.sqrt(
                    self.eta
                )
            y = y + noise

        return x, y

    def get_teacher_params(self) -> dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "rho": self.rho,
                "eta": self.eta,
                "activation": self.activation,
                "aggregation": self.aggregation,
            }
        )
        return config
