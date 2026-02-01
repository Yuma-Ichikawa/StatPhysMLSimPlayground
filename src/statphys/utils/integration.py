"""
Numerical integration utilities for statistical mechanics.

This module provides:
- Univariate and multivariate Gaussian integrals
- Gauss-Hermite quadrature (efficient for Gaussian integrals)
- Standard numerical quadrature (scipy quad)
- Monte Carlo integration

All functions support:
- Custom mean vector and covariance matrix
- Choice of integration method (quadrature, hermite, monte_carlo)
"""

from collections.abc import Callable

import numpy as np
from scipy import integrate
from scipy.special import roots_hermite

# =============================================================================
# Univariate Gaussian Integration
# =============================================================================


def gaussian_integral_1d(
    func: Callable[[float], float],
    mean: float = 0.0,
    variance: float = 1.0,
    method: str = "hermite",
    n_points: int = 100,
    limits: tuple[float, float] = (-np.inf, np.inf),
) -> float:
    """
    Compute univariate Gaussian integral: E[f(z)] where z ~ N(mean, variance).

    ∫ f(z) * (1/sqrt(2π*var)) * exp(-(z-μ)²/(2*var)) dz

    Args:
        func: Function to integrate.
        mean: Mean of Gaussian distribution.
        variance: Variance of Gaussian distribution.
        method: Integration method.
            - 'hermite': Gauss-Hermite quadrature (recommended for smooth functions)
            - 'quad': Scipy adaptive quadrature (for difficult integrands)
            - 'monte_carlo': Monte Carlo sampling (for high-dimensional)
        n_points: Number of quadrature/sample points.
        limits: Integration limits (for quad method).

    Returns:
        Integral value.

    Example:
        >>> # E[z^2] for z ~ N(0, 1) should be 1
        >>> gaussian_integral_1d(lambda z: z**2, mean=0, variance=1)
        1.0
        >>> # E[z] for z ~ N(3, 2) should be 3
        >>> gaussian_integral_1d(lambda z: z, mean=3, variance=2)
        3.0

    """
    std = np.sqrt(variance)

    if method == "hermite":
        # Gauss-Hermite quadrature
        # Hermite quadrature computes ∫ f(x) * exp(-x²) dx
        # For N(μ, σ²), substitute x = (z - μ) / (σ√2)
        points, weights = roots_hermite(n_points)

        # Transform points: z = σ√2 * x + μ
        z_points = std * np.sqrt(2) * points + mean

        # Compute weighted sum
        # Weight includes 1/√π from Hermite quadrature normalization
        result = np.sum(weights * np.array([func(z) for z in z_points])) / np.sqrt(np.pi)

        return result

    elif method == "quad":
        # Scipy adaptive quadrature
        def integrand(z):
            return func(z) * np.exp(-0.5 * ((z - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        if limits[0] == -np.inf and limits[1] == np.inf:
            # Use specialized infinite interval integration
            result, _ = integrate.quad(integrand, -np.inf, np.inf)
        else:
            result, _ = integrate.quad(integrand, limits[0], limits[1])

        return result

    elif method == "monte_carlo":
        # Monte Carlo sampling
        samples = np.random.normal(mean, std, n_points)
        return np.mean([func(s) for s in samples])

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'hermite', 'quad', 'monte_carlo'.")


def gaussian_expectation(
    func: Callable[[float], float],
    mean: float = 0.0,
    variance: float = 1.0,
    **kwargs,
) -> float:
    """
    Alias for gaussian_integral_1d with cleaner name.

    E[f(Z)] where Z ~ N(mean, variance)
    """
    return gaussian_integral_1d(func, mean, variance, **kwargs)


# =============================================================================
# Bivariate Gaussian Integration
# =============================================================================


def gaussian_integral_2d(
    func: Callable[[float, float], float],
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    method: str = "hermite",
    n_points: int = 50,
) -> float:
    """
    Compute bivariate Gaussian integral: E[f(z1, z2)] where (z1, z2) ~ N(μ, Σ).

    Args:
        func: Function of two variables to integrate.
        mean: Mean vector (2,). Defaults to [0, 0].
        cov: Covariance matrix (2, 2). Defaults to identity.
        method: Integration method ('hermite', 'quad', 'monte_carlo').
        n_points: Number of quadrature points per dimension.

    Returns:
        Integral value.

    Example:
        >>> # E[z1 * z2] for independent standard normals should be 0
        >>> gaussian_integral_2d(lambda z1, z2: z1 * z2)
        0.0
        >>> # E[z1 * z2] with correlation 0.5 should be 0.5
        >>> cov = np.array([[1, 0.5], [0.5, 1]])
        >>> gaussian_integral_2d(lambda z1, z2: z1 * z2, cov=cov)
        0.5

    """
    if mean is None:
        mean = np.zeros(2)
    if cov is None:
        cov = np.eye(2)

    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if method == "hermite":
        # Gauss-Hermite quadrature with Cholesky decomposition
        # For correlated Gaussians: z = L @ x + μ where L = cholesky(Σ)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # If not positive definite, add small regularization
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(2))

        points, weights = roots_hermite(n_points)
        points = np.sqrt(2) * points  # Scale for standard Hermite

        result = 0.0
        for _i, (x1, w1) in enumerate(zip(points, weights, strict=False)):
            for _j, (x2, w2) in enumerate(zip(points, weights, strict=False)):
                # Transform to correlated Gaussian
                x_vec = np.array([x1, x2])
                z = L @ x_vec + mean
                result += w1 * w2 * func(z[0], z[1])

        return result / np.pi

    elif method == "quad":
        # Scipy double quadrature
        # Compute Cholesky for change of variables
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(2))

        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)

        def integrand(z2, z1):
            z = np.array([z1, z2])
            diff = z - mean
            exponent = -0.5 * diff @ inv_cov @ diff
            return func(z1, z2) * np.exp(exponent) / (2 * np.pi * np.sqrt(det_cov))

        result, _ = integrate.dblquad(
            integrand,
            mean[0] - 10 * np.sqrt(cov[0, 0]),
            mean[0] + 10 * np.sqrt(cov[0, 0]),
            lambda z1: mean[1] - 10 * np.sqrt(cov[1, 1]),
            lambda z1: mean[1] + 10 * np.sqrt(cov[1, 1]),
        )

        return result

    elif method == "monte_carlo":
        samples = np.random.multivariate_normal(mean, cov, n_points)
        return np.mean([func(s[0], s[1]) for s in samples])

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Multivariate Gaussian Integration (General Dimension)
# =============================================================================


def gaussian_integral_nd(
    func: Callable[[np.ndarray], float],
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    dim: int = 2,
    method: str = "monte_carlo",
    n_points: int = 10000,
) -> float:
    """
    Compute multivariate Gaussian integral: E[f(z)] where z ~ N(μ, Σ).

    For dimensions > 2, Monte Carlo is recommended due to curse of dimensionality.

    Args:
        func: Function of d-dimensional vector to integrate.
        mean: Mean vector (d,). Defaults to zeros.
        cov: Covariance matrix (d, d). Defaults to identity.
        dim: Dimension (only used if mean/cov not provided).
        method: Integration method.
            - 'hermite': Tensor product quadrature (expensive for d > 3)
            - 'monte_carlo': Monte Carlo sampling (recommended for d > 2)
        n_points: Number of points.

    Returns:
        Integral value.

    """
    # Determine dimension from mean if provided
    if mean is not None:
        mean = np.asarray(mean)
        d = len(mean)
    else:
        d = dim
        mean = np.zeros(d)

    cov = np.eye(d) if cov is None else np.asarray(cov)

    if method == "monte_carlo":
        samples = np.random.multivariate_normal(mean, cov, n_points)
        return np.mean([func(s) for s in samples])

    elif method == "hermite":
        if d > 4:
            print(f"Warning: Hermite quadrature with d={d} is expensive. Consider 'monte_carlo'.")

        # Tensor product quadrature
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(d))

        # Reduce n_points per dimension for tractability
        n_per_dim = max(3, int(n_points ** (1.0 / d)))
        points, weights = roots_hermite(n_per_dim)
        points = np.sqrt(2) * points

        # Generate all tensor product combinations
        from itertools import product

        grid = list(product(range(n_per_dim), repeat=d))

        result = 0.0
        for indices in grid:
            x_vec = np.array([points[i] for i in indices])
            w = np.prod([weights[i] for i in indices])
            z = L @ x_vec + mean
            result += w * func(z)

        return result / (np.sqrt(np.pi) ** d)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Specialized Integrals for Statistical Mechanics
# =============================================================================


def teacher_student_integral(
    func: Callable[[float, float], float],
    m: float,
    q: float,
    rho: float = 1.0,
    method: str = "hermite",
    n_points: int = 50,
) -> float:
    """
    Compute integral over joint teacher-student field distribution.

    For teacher field u ~ N(0, rho) and student field z with
    correlation m/sqrt(q*rho), compute E[f(u, z)].

    This is the standard integral in perceptron/committee analysis.

    Args:
        func: Function of (teacher_field, student_field).
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.
        method: Integration method.
        n_points: Number of quadrature points.

    Returns:
        Integral value.

    """
    # Covariance matrix for (u, z)
    # Var(u) = rho, Var(z) = q, Cov(u, z) = m
    cov = np.array([[rho, m], [m, q]])

    # Ensure positive definiteness
    if rho * q - m**2 < 0:
        # Clip correlation to valid range
        max_m = np.sqrt(rho * q) * 0.999
        m_clipped = np.clip(m, -max_m, max_m)
        cov = np.array([[rho, m_clipped], [m_clipped, q]])

    return gaussian_integral_2d(func, mean=np.zeros(2), cov=cov, method=method, n_points=n_points)


def conditional_expectation(
    func: Callable[[float], float],
    given_value: float,
    m: float,
    q: float,
    rho: float = 1.0,
    conditional_on: str = "teacher",
    method: str = "hermite",
    n_points: int = 100,
) -> float:
    """
    Compute conditional expectation E[f(z) | u = given_value] or E[f(u) | z = given_value].

    For joint Gaussian (u, z) with Cov = [[rho, m], [m, q]]:
    - E[z | u] = (m/rho) * u, Var(z | u) = q - m²/rho
    - E[u | z] = (m/q) * z, Var(u | z) = rho - m²/q

    Args:
        func: Function to compute expectation of.
        given_value: The conditioned value.
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.
        conditional_on: 'teacher' or 'student'.
        method: Integration method.
        n_points: Number of quadrature points.

    Returns:
        Conditional expectation value.

    """
    if conditional_on == "teacher":
        # E[f(z) | u = given_value]
        cond_mean = (m / (rho + 1e-10)) * given_value
        cond_var = max(q - m**2 / (rho + 1e-10), 1e-10)
    else:
        # E[f(u) | z = given_value]
        cond_mean = (m / (q + 1e-10)) * given_value
        cond_var = max(rho - m**2 / (q + 1e-10), 1e-10)

    return gaussian_integral_1d(
        func, mean=cond_mean, variance=cond_var, method=method, n_points=n_points
    )


# =============================================================================
# Integration with Specific Distributions
# =============================================================================


def logistic_gaussian_integral(
    func: Callable[[float, float], float],
    m: float,
    q: float,
    rho: float = 1.0,
    method: str = "hermite",
    n_points: int = 50,
) -> float:
    """
    Compute integral E[f(y, z)] where y is generated from logistic teacher.

    y = sign(u) with probability sigmoid(u), u ~ N(0, rho)
    z ~ N(m/sqrt(rho) * u, q - m²/rho) conditionally

    Args:
        func: Function of (label y ∈ {-1, +1}, student field z).
        m, q, rho: Order parameters.
        method: Integration method.
        n_points: Number of points.

    Returns:
        Integral value.

    """
    from statphys.utils.special_functions import sigmoid

    def integrand(u, z_noise):
        # Student field given teacher field
        cond_var = max(q - m**2 / (rho + 1e-10), 1e-10)
        z = (m / np.sqrt(rho + 1e-10)) * u + np.sqrt(cond_var) * z_noise

        # Logistic probability
        prob_plus = sigmoid(u)

        # Expected value over y
        return prob_plus * func(1.0, z) + (1 - prob_plus) * func(-1.0, z)

    # Integrate over independent standard normals
    cov = np.array([[rho, 0], [0, 1]])
    return gaussian_integral_2d(
        integrand, mean=np.zeros(2), cov=cov, method=method, n_points=n_points
    )
