"""
GLM (Generalized Linear Model) dataset implementations.

Provides datasets with probabilistic label generation following:
- Logistic: P(y=1|u) = sigmoid(u)
- Probit: P(y=1|u) = Phi(u) (Gaussian CDF)
- Custom link functions
"""

from typing import Any, Dict, Optional, Tuple, Callable

import torch
import numpy as np
from scipy import special

from statphys.dataset.base import BaseDataset, TeacherType


class LogisticTeacherDataset(BaseDataset):
    """
    Dataset with logistic teacher for binary classification.

    The data generation follows:
        x ~ N(0, I_d)
        u = (1/sqrt(d)) * W0^T @ x
        P(y=1|u) = sigmoid(u) = 1 / (1 + exp(-u))
        y ~ Bernoulli(sigmoid(u)), output in {-1, +1}

    This is the standard GLM teacher for logistic regression analysis.

    Attributes:
        d: Input dimension.
        rho: Norm of teacher weights (||W0||^2 / d).
        W0: Teacher weight vector.
    """

    def __init__(
        self,
        d: int,
        rho: float = 1.0,
        W0: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize LogisticTeacherDataset.

        Args:
            d: Input dimension.
            rho: Teacher weight norm (||W0||^2 / d). Defaults to 1.0.
            W0: Teacher weights. If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.
        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.rho = rho

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "teacher_type": "logistic",
        }

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and label y (+1 or -1).
        """
        # Generate input x ~ N(0, I)
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)

        # Compute teacher score u = (1/sqrt(d)) * W0^T @ x
        u = (self.W0.T @ x / np.sqrt(self.d)).squeeze()

        # Generate label from Bernoulli(sigmoid(u))
        prob = torch.sigmoid(u)
        y_binary = torch.bernoulli(prob)

        # Convert to {-1, +1}
        y = 2.0 * y_binary - 1.0

        return x, y.to(dtype=self.dtype)

    def get_teacher_params(self) -> Dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "teacher_type": "logistic",
        })
        return config


class ProbitTeacherDataset(BaseDataset):
    """
    Dataset with probit teacher for binary classification.

    The data generation follows:
        x ~ N(0, I_d)
        u = (1/sqrt(d)) * W0^T @ x
        P(y=1|u) = Phi(u) (Gaussian CDF)
        y ~ Bernoulli(Phi(u)), output in {-1, +1}

    Probit model is analytically convenient for replica calculations.

    Attributes:
        d: Input dimension.
        rho: Norm of teacher weights.
        W0: Teacher weight vector.
    """

    def __init__(
        self,
        d: int,
        rho: float = 1.0,
        W0: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize ProbitTeacherDataset.

        Args:
            d: Input dimension.
            rho: Teacher weight norm. Defaults to 1.0.
            W0: Teacher weights. If None, sampled from N(0, rho*I).
            device: Computation device.
            dtype: Data type.
        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.rho = rho

        # Initialize teacher weights
        if W0 is not None:
            self.W0 = W0.to(device=self.device, dtype=self.dtype)
        else:
            self.W0 = torch.randn(d, 1, device=self.device, dtype=self.dtype) * np.sqrt(rho)

        self._teacher_params = {
            "W0": self.W0,
            "rho": self.rho,
            "teacher_type": "probit",
        }

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and label y (+1 or -1).
        """
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)

        # Compute teacher score
        u = (self.W0.T @ x / np.sqrt(self.d)).squeeze()

        # Generate label from Bernoulli(Phi(u))
        # Phi(u) = 0.5 * (1 + erf(u / sqrt(2)))
        prob = 0.5 * (1.0 + torch.erf(u / np.sqrt(2.0)))
        y_binary = torch.bernoulli(prob)

        # Convert to {-1, +1}
        y = 2.0 * y_binary - 1.0

        return x, y.to(dtype=self.dtype)

    def get_teacher_params(self) -> Dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "teacher_type": "probit",
        })
        return config


class GaussianMixtureDataset(BaseDataset):
    """
    Gaussian Mixture dataset for classification.

    The data generation follows:
        y ~ Uniform({-1, +1}) or specified prior
        x | y ~ N(y * mu, I_d)

    This is the standard toy model for DMFT analysis of SGD.

    Attributes:
        d: Input dimension.
        mu: Class mean direction (scaled so ||mu||^2/d = signal).
        signal: Signal-to-noise ratio (||mu||^2 / d).
    """

    def __init__(
        self,
        d: int,
        signal: float = 1.0,
        mu: Optional[torch.Tensor] = None,
        prior: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize GaussianMixtureDataset.

        Args:
            d: Input dimension.
            signal: Signal strength (||mu||^2 / d). Defaults to 1.0.
            mu: Class mean direction. If None, uses e_1 direction.
            prior: Prior probability of y=+1. Defaults to 0.5.
            device: Computation device.
            dtype: Data type.
        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.signal = signal
        self.prior = prior

        # Initialize mean direction
        if mu is not None:
            self.mu = mu.to(device=self.device, dtype=self.dtype)
            # Rescale to have correct signal strength
            mu_norm_sq = torch.sum(self.mu ** 2).item()
            self.mu = self.mu * np.sqrt(signal * d / mu_norm_sq)
        else:
            # Default: mu = sqrt(signal) * e_1 direction
            self.mu = torch.zeros(d, device=self.device, dtype=self.dtype)
            self.mu[0] = np.sqrt(signal * d)

        self._teacher_params = {
            "mu": self.mu,
            "signal": self.signal,
            "prior": self.prior,
            "teacher_type": "gaussian_mixture",
        }

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and label y (+1 or -1).
        """
        # Generate label y ~ Bernoulli(prior)
        if torch.rand(1).item() < self.prior:
            y = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            y = torch.tensor(-1.0, device=self.device, dtype=self.dtype)

        # Generate x | y ~ N(y * mu / sqrt(d), I_d)
        # Note: mu is already scaled so ||mu||^2 = signal * d
        noise = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = y * self.mu / np.sqrt(self.d) + noise

        return x, y

    def get_bayes_optimal_error(self) -> float:
        """
        Compute Bayes optimal classification error.

        For balanced GMM with signal r = ||mu||^2/d:
            P_error = Phi(-sqrt(r/2))

        Returns:
            Bayes optimal error probability.
        """
        from scipy.stats import norm
        return norm.cdf(-np.sqrt(self.signal / 2.0))

    def get_teacher_params(self) -> Dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update({
            "signal": self.signal,
            "prior": self.prior,
            "teacher_type": "gaussian_mixture",
        })
        return config


class MulticlassGaussianMixtureDataset(BaseDataset):
    """
    Multi-class Gaussian Mixture dataset.

    The data generation follows:
        y ~ Categorical(pi_1, ..., pi_K)
        x | y=k ~ N(mu_k, I_d)

    Each class has its own mean mu_k.

    Attributes:
        d: Input dimension.
        n_classes: Number of classes.
        means: Class means (n_classes, d).
    """

    def __init__(
        self,
        d: int,
        n_classes: int = 3,
        signal: float = 1.0,
        priors: Optional[torch.Tensor] = None,
        means: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize MulticlassGaussianMixtureDataset.

        Args:
            d: Input dimension.
            n_classes: Number of classes (K).
            signal: Average signal strength per class.
            priors: Class priors (K,). Defaults to uniform.
            means: Class means (K, d). If None, uses orthogonal directions.
            device: Computation device.
            dtype: Data type.
        """
        super().__init__(d=d, device=device, dtype=dtype, **kwargs)

        self.n_classes = n_classes
        self.signal = signal

        # Initialize priors
        if priors is not None:
            self.priors = priors.to(device=self.device, dtype=self.dtype)
        else:
            self.priors = torch.ones(n_classes, device=self.device, dtype=self.dtype) / n_classes

        # Initialize means
        if means is not None:
            self.means = means.to(device=self.device, dtype=self.dtype)
        else:
            # Default: orthogonal directions with equal signal
            self.means = torch.zeros(n_classes, d, device=self.device, dtype=self.dtype)
            for k in range(min(n_classes, d)):
                self.means[k, k] = np.sqrt(signal * d)

        self._teacher_params = {
            "means": self.means,
            "priors": self.priors,
            "n_classes": self.n_classes,
            "signal": self.signal,
            "teacher_type": "multiclass_gmm",
        }

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single (x, y) sample.

        Returns:
            Tuple of input x (d,) and label y (class index 0 to K-1).
        """
        # Sample class from categorical
        y_idx = torch.multinomial(self.priors, 1).item()
        y = torch.tensor(float(y_idx), device=self.device, dtype=self.dtype)

        # Generate x | y ~ N(mu_y / sqrt(d), I_d)
        noise = torch.randn(self.d, device=self.device, dtype=self.dtype)
        x = self.means[y_idx] / np.sqrt(self.d) + noise

        return x, y

    def get_teacher_params(self) -> Dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
            "signal": self.signal,
            "teacher_type": "multiclass_gmm",
        })
        return config
