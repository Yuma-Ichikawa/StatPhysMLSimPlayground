"""
Special functions for statistical mechanics calculations.

This module provides special functions commonly used in:
- Replica method saddle-point equations
- Online learning ODE dynamics
- Committee machine analysis
- DMFT calculations

Reference: Engel & Van den Broeck, "Statistical Mechanics of Learning"
"""

from collections.abc import Callable

import numpy as np
from scipy.special import erf, erfc, erfinv

# =============================================================================
# Gaussian Distribution Functions
# =============================================================================


def gaussian_pdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    """
    Gaussian probability density function.

    phi(x; mu, sigma) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2 / (2*sigma^2))

    Args:
        x: Input value(s).
        mu: Mean. Defaults to 0.0.
        sigma: Standard deviation. Defaults to 1.0.

    Returns:
        PDF value(s).

    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def gaussian_cdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    """
    Gaussian cumulative distribution function.

    Phi(x; mu, sigma) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

    Args:
        x: Input value(s).
        mu: Mean. Defaults to 0.0.
        sigma: Standard deviation. Defaults to 1.0.

    Returns:
        CDF value(s).

    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def gaussian_tail(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    """
    Gaussian tail probability (complementary CDF).

    H(x) = P(Z > x) = 0.5 * erfc((x - mu) / (sigma * sqrt(2)))

    Args:
        x: Input value(s).
        mu: Mean. Defaults to 0.0.
        sigma: Standard deviation. Defaults to 1.0.

    Returns:
        Tail probability value(s).

    """
    return 0.5 * erfc((x - mu) / (sigma * np.sqrt(2)))


# Aliases for standard notation
Phi = gaussian_cdf  # Standard normal CDF
H = gaussian_tail  # Standard tail function
phi = gaussian_pdf  # Standard normal PDF


def gaussian_quantile(
    p: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0
) -> float | np.ndarray:
    """
    Gaussian quantile function (inverse CDF).

    Args:
        p: Probability value(s) in (0, 1).
        mu: Mean. Defaults to 0.0.
        sigma: Standard deviation. Defaults to 1.0.

    Returns:
        Quantile value(s).

    """
    return mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)


# =============================================================================
# Activation Functions and Their Derivatives
# =============================================================================


def erf_activation(x: float | np.ndarray) -> float | np.ndarray:
    """
    Error function activation: erf(x / sqrt(2)).

    Standard choice in committee machine analysis because Gaussian
    integrals have closed-form solutions.

    Args:
        x: Input value(s).

    Returns:
        Activation value(s).

    """
    return erf(x / np.sqrt(2))


def erf_derivative(x: float | np.ndarray) -> float | np.ndarray:
    """
    Derivative of erf activation.

    d/dx erf(x/sqrt(2)) = sqrt(2/pi) * exp(-x^2/2)

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).

    """
    return np.sqrt(2 / np.pi) * np.exp(-(x**2) / 2)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """
    Sigmoid function with numerical stability.

    sigma(x) = 1 / (1 + exp(-x))

    Args:
        x: Input value(s).

    Returns:
        Sigmoid value(s).

    """
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(x: float | np.ndarray) -> float | np.ndarray:
    """
    Derivative of sigmoid function.

    d/dx sigma(x) = sigma(x) * (1 - sigma(x))

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).

    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x: float | np.ndarray) -> float | np.ndarray:
    """
    Derivative of tanh function.

    d/dx tanh(x) = 1 - tanh(x)^2 = sech(x)^2

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).

    """
    return 1 - np.tanh(x) ** 2


def relu(x: float | np.ndarray) -> float | np.ndarray:
    """
    ReLU activation function.

    Args:
        x: Input value(s).

    Returns:
        max(0, x).

    """
    return np.maximum(0, x)


def relu_derivative(x: float | np.ndarray) -> float | np.ndarray:
    """
    Derivative of ReLU (Heaviside step function).

    Args:
        x: Input value(s).

    Returns:
        1 if x > 0, else 0.

    """
    return np.where(x > 0, 1.0, 0.0)


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    """
    Softplus activation: log(1 + exp(x)).

    Smooth approximation to ReLU.

    Args:
        x: Input value(s).

    Returns:
        Softplus value(s).

    """
    # Numerical stability
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


# =============================================================================
# Committee Machine Correlation Functions
# =============================================================================


def I2(Q_ab: float, activation: str = "erf") -> float:
    """
    Two-point correlation function I_2 for committee machines.

    I_2(Q_ab) = E[g(u)g(v)] where (u, v) ~ N(0, [[1, Q_ab], [Q_ab, 1]])

    For erf activation:
        I_2(Q_ab) = (2/pi) * arcsin(Q_ab)

    Args:
        Q_ab: Correlation between two units.
        activation: Activation function ('erf', 'sign', 'relu').

    Returns:
        I_2 value.

    """
    if activation == "erf":
        # I_2 for erf activation: (2/pi) * arcsin(Q_ab)
        Q_ab = np.clip(Q_ab, -0.9999, 0.9999)
        return (2 / np.pi) * np.arcsin(Q_ab)
    elif activation == "sign":
        # I_2 for sign activation: 1 - (2/pi) * arccos(Q_ab)
        Q_ab = np.clip(Q_ab, -0.9999, 0.9999)
        return 1 - (2 / np.pi) * np.arccos(Q_ab)
    elif activation == "relu":
        # I_2 for ReLU: numerical integration
        Q_ab = np.clip(Q_ab, -0.9999, 0.9999)
        return Q_ab * (np.pi - np.arccos(Q_ab)) / (2 * np.pi) + np.sqrt(1 - Q_ab**2) / (2 * np.pi)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def I3(Q_ab: float, Q_ac: float, Q_bc: float, activation: str = "erf") -> float:
    """
    Three-point correlation function I_3 for committee machines.

    I_3 = E[g'(u)g(v)g(w)] where (u, v, w) have correlation Q.

    This appears in committee machine saddle-point equations.

    Args:
        Q_ab, Q_ac, Q_bc: Pairwise correlations.
        activation: Activation function.

    Returns:
        I_3 value.

    """
    if activation == "erf":
        # Simplified formula for symmetric case
        # General case requires numerical integration
        Q = (Q_ab + Q_ac + Q_bc) / 3  # Average correlation
        return (2 / np.pi) ** 1.5 / np.sqrt(1 + Q + 1e-10)
    else:
        # Numerical integration for other activations
        # Placeholder - implement full integration if needed
        return 0.0


def I4(Q_ab: float, Q_cd: float, Q_ac: float, activation: str = "erf") -> float:
    """
    Four-point correlation function I_4 for committee machines.

    I_4 = E[g'(u)g'(v)g(w)g(z)]

    Args:
        Q_ab, Q_cd, Q_ac: Correlations.
        activation: Activation function.

    Returns:
        I_4 value.

    """
    if activation == "erf":
        # Approximate formula
        return I2(Q_ab, "erf") * I2(Q_cd, "erf")
    else:
        return 0.0


# =============================================================================
# Proximal Operators and Related Functions
# =============================================================================


def soft_threshold(x: float | np.ndarray, threshold: float) -> float | np.ndarray:
    """
    Soft thresholding operator (proximal of L1 norm).

    S_λ(x) = sign(x) * max(|x| - λ, 0)

    Args:
        x: Input value(s).
        threshold: Threshold parameter λ.

    Returns:
        Soft-thresholded value(s).

    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def firm_threshold(x: float | np.ndarray, lambda1: float, lambda2: float) -> float | np.ndarray:
    """
    Firm thresholding operator (SCAD-like).

    Args:
        x: Input value(s).
        lambda1: Lower threshold.
        lambda2: Upper threshold.

    Returns:
        Firm-thresholded value(s).

    """
    abs_x = np.abs(x)
    result = np.where(abs_x <= lambda1, 0, x)
    result = np.where(
        (abs_x > lambda1) & (abs_x <= lambda2),
        np.sign(x) * (abs_x - lambda1) * lambda2 / (lambda2 - lambda1),
        result,
    )
    return result


def moreau_envelope(
    func: Callable[[float], float], x: float, gamma: float, n_points: int = 100
) -> tuple[float, float]:
    """
    Compute Moreau envelope and its proximal point.

    M_γf(x) = min_y [f(y) + (1/2γ)||x - y||^2]
    prox_γf(x) = argmin_y [f(y) + (1/2γ)||x - y||^2]

    Args:
        func: Function to compute envelope of.
        x: Point to evaluate at.
        gamma: Smoothing parameter.
        n_points: Number of points for optimization.

    Returns:
        Tuple of (envelope value, proximal point).

    """
    y_vals = np.linspace(x - 5 * np.sqrt(gamma), x + 5 * np.sqrt(gamma), n_points)
    envelope_vals = [func(y) + 0.5 / gamma * (x - y) ** 2 for y in y_vals]
    min_idx = np.argmin(envelope_vals)
    return envelope_vals[min_idx], y_vals[min_idx]


# =============================================================================
# Generalization Error Functions
# =============================================================================


def classification_error_linear(m: float, q: float, rho: float = 1.0) -> float:
    """
    Classification error for linear classifier with sign teacher.

    E_g = (1/π) * arccos(m / sqrt(q * rho))

    Args:
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.

    Returns:
        Classification error.

    """
    if q > 0 and rho > 0:
        cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
        return np.arccos(cos_angle) / np.pi
    return 0.5


def regression_error_linear(m: float, q: float, rho: float = 1.0, eta: float = 0.0) -> float:
    """
    Generalization error for linear regression.

    E_g = 0.5 * (rho - 2*m + q) + 0.5 * eta

    Args:
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.
        eta: Noise variance.

    Returns:
        Generalization error (MSE).

    """
    return 0.5 * (rho - 2 * m + q) + 0.5 * eta


def training_error_linear(
    m: float, q: float, rho: float, eta: float, alpha: float, reg_param: float = 0.0
) -> float:
    """
    Training error for ridge regression.

    Uses replica formula for training error.

    Args:
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.
        eta: Noise variance.
        alpha: Sample ratio.
        reg_param: Regularization parameter.

    Returns:
        Training error.

    """
    # Residual variance
    V = rho - 2 * m + q + eta

    # Training error from replica calculation
    e_train = V * reg_param**2 / (alpha * (reg_param + alpha) ** 2 + 1e-10) if alpha > 0 else 0.0

    return e_train


# =============================================================================
# Utility Functions
# =============================================================================


def compute_stability_parameter(m: float, q: float, rho: float = 1.0) -> float:
    """
    Compute stability parameter (normalized overlap).

    κ = m / sqrt(q * rho)

    This appears in perceptron generalization bounds.

    Args:
        m: Teacher-student overlap.
        q: Student self-overlap.
        rho: Teacher norm.

    Returns:
        Stability parameter κ.

    """
    if q > 0 and rho > 0:
        return m / np.sqrt(q * rho)
    return 0.0


def fisher_information_binary(m: float, q: float, rho: float = 1.0) -> float:
    """
    Fisher information for binary classification.

    Appears in asymptotic analysis of classification.

    Args:
        m, q, rho: Order parameters.

    Returns:
        Fisher information.

    """
    kappa = compute_stability_parameter(m, q, rho)
    kappa = np.clip(kappa, -10, 10)
    return gaussian_pdf(kappa) ** 2 / (gaussian_cdf(kappa) * gaussian_tail(kappa) + 1e-10)
