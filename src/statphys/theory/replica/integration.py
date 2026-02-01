"""Numerical integration utilities for replica calculations."""

from collections.abc import Callable

import numpy as np
from scipy.integrate import dblquad, quad


def gaussian_integral(
    func: Callable[[float], float],
    mean: float = 0.0,
    var: float = 1.0,
    n_points: int = 100,
    method: str = "quadrature",
    limits: tuple[float, float] = (-10, 10),
) -> float:
    """
    Compute Gaussian integral: E_z[f(z)] where z ~ N(mean, var).

    Args:
        func: Function to integrate.
        mean: Mean of Gaussian.
        var: Variance of Gaussian.
        n_points: Number of quadrature points.
        method: Integration method ('quadrature', 'hermite', 'monte_carlo').
        limits: Integration limits for quadrature.

    Returns:
        Integral value.

    Example:
        >>> # E[z^2] for z ~ N(0, 1)
        >>> gaussian_integral(lambda z: z**2)
        1.0

    """
    std = np.sqrt(var)

    if method == "quadrature":

        def integrand(z):
            return func(z) * np.exp(-((z - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        result, _ = quad(integrand, limits[0] * std + mean, limits[1] * std + mean)
        return result

    elif method == "hermite":
        # Gauss-Hermite quadrature
        points, weights = np.polynomial.hermite.hermgauss(n_points)
        # Transform to general Gaussian
        transformed_points = np.sqrt(2 * var) * points + mean
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
    var1: float = 1.0,
    var2: float = 1.0,
    cov: float = 0.0,
    n_points: int = 50,
    method: str = "hermite",
) -> float:
    """
    Compute 2D Gaussian integral.

    E_{z1, z2}[f(z1, z2)] where (z1, z2) ~ N(μ, Σ)

    Args:
        func: Function of two variables.
        mean1, mean2: Means.
        var1, var2: Variances.
        cov: Covariance.
        n_points: Points per dimension.
        method: Integration method.

    Returns:
        Integral value.

    """
    # Build covariance matrix and Cholesky decomposition
    sigma = np.array([[var1, cov], [cov, var2]])
    L = np.linalg.cholesky(sigma)

    if method == "hermite":
        points, weights = np.polynomial.hermite.hermgauss(n_points)
        points = np.sqrt(2) * points

        result = 0.0
        for _i, (x1, w1) in enumerate(zip(points, weights, strict=False)):
            for _j, (x2, w2) in enumerate(zip(points, weights, strict=False)):
                z = L @ np.array([x1, x2]) + np.array([mean1, mean2])
                result += w1 * w2 * func(z[0], z[1])

        return result / np.pi

    elif method == "quadrature":

        def integrand(z2, z1):
            pdf = np.exp(
                -0.5
                * (
                    (z1 - mean1) ** 2 / var1
                    + (z2 - mean2) ** 2 / var2
                    - 2 * cov * (z1 - mean1) * (z2 - mean2) / (var1 * var2)
                )
            )
            pdf /= 2 * np.pi * np.sqrt(var1 * var2 - cov**2)
            return func(z1, z2) * pdf

        result, _ = dblquad(
            integrand,
            mean1 - 5 * np.sqrt(var1),
            mean1 + 5 * np.sqrt(var1),
            lambda x: mean2 - 5 * np.sqrt(var2),
            lambda x: mean2 + 5 * np.sqrt(var2),
        )
        return result

    else:
        raise ValueError(f"Unknown method: {method}")


def moreau_envelope(
    func: Callable[[float], float],
    x: float,
    gamma: float,
    n_points: int = 200,
) -> float:
    """
    Compute Moreau envelope of a function.

    M_γf(x) = min_y [f(y) + (1/2γ)||x - y||^2]

    Args:
        func: Function to compute envelope of.
        x: Point to evaluate at.
        gamma: Smoothing parameter.
        n_points: Points for numerical optimization.

    Returns:
        Moreau envelope value.

    """
    # Grid search for minimum
    search_range = 5 * np.sqrt(gamma)
    y_vals = np.linspace(x - search_range, x + search_range, n_points)
    envelope_vals = [func(y) + 0.5 / gamma * (x - y) ** 2 for y in y_vals]
    return min(envelope_vals)


def proximal_operator(
    func: Callable[[float], float],
    x: float,
    gamma: float,
    n_points: int = 200,
) -> float:
    """
    Compute proximal operator of a function.

    prox_γf(x) = argmin_y [f(y) + (1/2γ)||x - y||^2]

    Args:
        func: Function.
        x: Point.
        gamma: Step size.
        n_points: Points for optimization.

    Returns:
        Proximal operator value.

    """
    search_range = 5 * np.sqrt(gamma)
    y_vals = np.linspace(x - search_range, x + search_range, n_points)
    envelope_vals = [func(y) + 0.5 / gamma * (x - y) ** 2 for y in y_vals]
    min_idx = np.argmin(envelope_vals)
    return y_vals[min_idx]


def soft_threshold(x: float, threshold: float) -> float:
    """
    Soft thresholding operator (proximal for L1).

    S_λ(x) = sign(x) * max(|x| - λ, 0)

    Args:
        x: Input value.
        threshold: Threshold (λ).

    Returns:
        Soft-thresholded value.

    """
    return np.sign(x) * max(abs(x) - threshold, 0)


def hard_threshold(x: float, threshold: float) -> float:
    """
    Hard thresholding operator (proximal for L0).

    H_λ(x) = x if |x| > λ else 0

    Args:
        x: Input value.
        threshold: Threshold.

    Returns:
        Hard-thresholded value.

    """
    return x if abs(x) > threshold else 0


def compute_free_entropy_density(
    order_params: dict,
    alpha: float,
    params: dict,
) -> float:
    """
    Compute free entropy density for replica symmetric solution.

    This is problem-specific; this function provides a template.

    Args:
        order_params: Dictionary of order parameters.
        alpha: Sample ratio.
        params: Problem parameters.

    Returns:
        Free entropy density.

    """
    # Template implementation for ridge regression
    m = order_params.get("m", 0)
    q = order_params.get("q", 1)
    rho = params.get("rho", 1)
    lam = params.get("reg_param", 0.01)

    # Energetic term
    e_term = -0.5 * alpha * (rho - 2 * m + q)

    # Entropic term (simplified)
    s_term = 0.5 * np.log(2 * np.pi * q) if q > 0 else 0

    # Regularization term
    r_term = -0.5 * lam * q

    return e_term + s_term + r_term
