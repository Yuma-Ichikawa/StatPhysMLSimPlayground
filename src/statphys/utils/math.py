"""
Mathematical helper functions for statistical mechanics calculations.

This module provides functions commonly used in:
- Replica calculations
- Online learning dynamics
- Order parameter computations
"""

from collections.abc import Callable

import numpy as np
import torch
from scipy.special import erf


def H_function(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """
    Complementary Gaussian CDF (tail probability).

    H(x) = P(Z > x) where Z ~ N(0, 1)
    H(x) = 0.5 * erfc(x / sqrt(2)) = 0.5 * (1 - erf(x / sqrt(2)))

    Args:
        x: Input value(s).

    Returns:
        H(x) value(s).

    Example:
        >>> H_function(0)  # Should be 0.5
        0.5

    """
    if isinstance(x, torch.Tensor):
        return 0.5 * torch.erfc(x / np.sqrt(2.0))
    elif isinstance(x, np.ndarray):
        return 0.5 * (1 - erf(x / np.sqrt(2.0)))
    else:
        return 0.5 * (1 - erf(x / np.sqrt(2.0)))


def erf_scaled(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """
    Scaled error function: erf(x / sqrt(2)).

    Args:
        x: Input value(s).

    Returns:
        erf(x / sqrt(2)) value(s).

    """
    if isinstance(x, torch.Tensor):
        return torch.erf(x / np.sqrt(2.0))
    elif isinstance(x, np.ndarray):
        return erf(x / np.sqrt(2.0))
    else:
        return erf(x / np.sqrt(2.0))


def sigmoid(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + exp(-x)).

    Args:
        x: Input value(s).

    Returns:
        Sigmoid of x.

    """
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    else:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """
    ReLU activation function: max(0, x).

    Args:
        x: Input value(s).

    Returns:
        ReLU of x.

    """
    if isinstance(x, torch.Tensor):
        return torch.relu(x)
    else:
        return np.maximum(0, x)


def gaussian_integral(
    func: Callable[[float], float],
    mean: float = 0.0,
    std: float = 1.0,
    n_points: int = 100,
    method: str = "quadrature",
) -> float:
    """
    Compute Gaussian integral: E[f(z)] where z ~ N(mean, std^2).

    Uses Gauss-Hermite quadrature for efficiency.

    Args:
        func: Function to integrate.
        mean: Mean of Gaussian. Defaults to 0.0.
        std: Standard deviation of Gaussian. Defaults to 1.0.
        n_points: Number of quadrature points. Defaults to 100.
        method: Integration method ('quadrature' or 'monte_carlo').

    Returns:
        The integral value.

    Example:
        >>> # E[z^2] for z ~ N(0, 1) should be 1
        >>> gaussian_integral(lambda z: z**2)
        1.0

    """
    if method == "quadrature":
        # Gauss-Hermite quadrature
        points, weights = np.polynomial.hermite.hermgauss(n_points)
        # Transform for general Gaussian
        transformed_points = np.sqrt(2) * std * points + mean
        result = sum(w * func(x) for x, w in zip(transformed_points, weights, strict=False))
        return result / np.sqrt(np.pi)
    elif method == "monte_carlo":
        samples = np.random.normal(mean, std, n_points)
        return np.mean([func(s) for s in samples])
    else:
        raise ValueError(f"Unknown method: {method}")


def double_gaussian_integral(
    func: Callable[[float, float], float],
    mean1: float = 0.0,
    mean2: float = 0.0,
    cov: np.ndarray | None = None,
    n_points: int = 50,
) -> float:
    """
    Compute 2D Gaussian integral: E[f(z1, z2)] where (z1, z2) ~ N(μ, Σ).

    Args:
        func: Function of two variables to integrate.
        mean1: Mean of first variable.
        mean2: Mean of second variable.
        cov: 2x2 covariance matrix. Defaults to identity.
        n_points: Number of quadrature points per dimension.

    Returns:
        The integral value.

    """
    if cov is None:
        cov = np.eye(2)

    # Cholesky decomposition for correlated Gaussians
    L = np.linalg.cholesky(cov)

    points, weights = np.polynomial.hermite.hermgauss(n_points)
    points = np.sqrt(2) * points

    result = 0.0
    for _i, (x1, w1) in enumerate(zip(points, weights, strict=False)):
        for _j, (x2, w2) in enumerate(zip(points, weights, strict=False)):
            # Transform to correlated Gaussian
            z = L @ np.array([x1, x2]) + np.array([mean1, mean2])
            result += w1 * w2 * func(z[0], z[1])

    return result / np.pi


def compute_overlap(
    w1: torch.Tensor,
    w2: torch.Tensor,
    normalize: bool = True,
) -> float:
    """
    Compute overlap between two weight vectors.

    Args:
        w1: First weight vector.
        w2: Second weight vector.
        normalize: If True, normalize by dimension. Defaults to True.

    Returns:
        Overlap value: (1/d) * w1^T @ w2 if normalized, else w1^T @ w2.

    """
    overlap = torch.dot(w1.flatten(), w2.flatten()).item()
    if normalize:
        return overlap / w1.numel()
    return overlap


def proximal_operator(
    w: torch.Tensor,
    reg_type: str,
    reg_param: float,
    lr: float,
) -> torch.Tensor:
    """
    Apply proximal operator for regularization.

    Args:
        w: Weight tensor.
        reg_type: Type of regularization ('l1', 'l2', 'elastic').
        reg_param: Regularization parameter.
        lr: Learning rate (step size).

    Returns:
        Updated weight tensor after proximal step.

    """
    if reg_type == "l1":
        # Soft thresholding
        threshold = reg_param * lr
        return torch.sign(w) * torch.maximum(torch.abs(w) - threshold, torch.zeros_like(w))
    elif reg_type == "l2":
        # Shrinkage
        return w / (1 + 2 * reg_param * lr)
    elif reg_type == "elastic":
        # Combination of L1 and L2
        alpha = 0.5  # mixing parameter
        w_l2 = w / (1 + 2 * (1 - alpha) * reg_param * lr)
        threshold = alpha * reg_param * lr
        return torch.sign(w_l2) * torch.maximum(torch.abs(w_l2) - threshold, torch.zeros_like(w_l2))
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def moreau_envelope(
    func: Callable[[float], float],
    x: float,
    gamma: float,
    n_points: int = 100,
) -> float:
    """
    Compute Moreau envelope of a function.

    M_γf(x) = min_y [f(y) + (1/2γ)||x - y||^2]

    Args:
        func: Function to compute envelope of.
        x: Point to evaluate at.
        gamma: Smoothing parameter.
        n_points: Number of points for numerical minimization.

    Returns:
        Moreau envelope value at x.

    """
    y_vals = np.linspace(x - 5 * np.sqrt(gamma), x + 5 * np.sqrt(gamma), n_points)
    envelope_vals = [func(y) + 0.5 / gamma * (x - y) ** 2 for y in y_vals]
    return min(envelope_vals)
