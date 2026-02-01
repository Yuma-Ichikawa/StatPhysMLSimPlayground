"""
Utility functions for StatPhys-ML.

This module provides:
- Random seed fixing and device management
- Mathematical helper functions for replica/online calculations
- Special functions (Gaussian CDF/PDF, activations, correlations)
- Numerical integration (Gaussian integrals, Gauss-Hermite quadrature)
- I/O utilities for saving/loading results
"""

from statphys.utils.seed import fix_seed, get_device
from statphys.utils.math import (
    gaussian_integral,
    double_gaussian_integral,
    H_function,
    sigmoid,
    relu,
    erf_scaled,
    compute_overlap,
    proximal_operator,
    moreau_envelope,
)
from statphys.utils.special_functions import (
    # Gaussian distribution functions
    gaussian_pdf,
    gaussian_cdf,
    gaussian_tail,
    gaussian_quantile,
    Phi,
    H,
    phi,
    # Activation functions
    erf_activation,
    erf_derivative,
    sigmoid_derivative,
    tanh_derivative,
    relu_derivative,
    softplus,
    # Committee machine functions
    I2,
    I3,
    I4,
    # Proximal operators
    soft_threshold,
    firm_threshold,
    # Generalization error functions
    classification_error_linear,
    regression_error_linear,
    training_error_linear,
    compute_stability_parameter,
    fisher_information_binary,
)
from statphys.utils.integration import (
    gaussian_integral_1d,
    gaussian_integral_2d,
    gaussian_integral_nd,
    gaussian_expectation,
    teacher_student_integral,
    conditional_expectation,
    logistic_gaussian_integral,
)
from statphys.utils.io import save_results, load_results, ResultsManager

__all__ = [
    # Seed utilities
    "fix_seed",
    "get_device",
    # Basic math utilities
    "gaussian_integral",
    "double_gaussian_integral",
    "H_function",
    "sigmoid",
    "relu",
    "erf_scaled",
    "compute_overlap",
    "proximal_operator",
    "moreau_envelope",
    # Gaussian distribution functions
    "gaussian_pdf",
    "gaussian_cdf",
    "gaussian_tail",
    "gaussian_quantile",
    "Phi",
    "H",
    "phi",
    # Activation functions
    "erf_activation",
    "erf_derivative",
    "sigmoid_derivative",
    "tanh_derivative",
    "relu_derivative",
    "softplus",
    # Committee machine correlation functions
    "I2",
    "I3",
    "I4",
    # Proximal operators
    "soft_threshold",
    "firm_threshold",
    # Generalization error functions
    "classification_error_linear",
    "regression_error_linear",
    "training_error_linear",
    "compute_stability_parameter",
    "fisher_information_binary",
    # Numerical integration
    "gaussian_integral_1d",
    "gaussian_integral_2d",
    "gaussian_integral_nd",
    "gaussian_expectation",
    "teacher_student_integral",
    "conditional_expectation",
    "logistic_gaussian_integral",
    # I/O utilities
    "save_results",
    "load_results",
    "ResultsManager",
]
