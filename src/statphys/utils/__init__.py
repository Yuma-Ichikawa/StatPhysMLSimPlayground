"""
Utility functions for StatPhys-ML.

This module provides:
- Random seed fixing and device management
- Mathematical helper functions for replica/online calculations
- Special functions (Gaussian CDF/PDF, activations, correlations)
- Numerical integration (Gaussian integrals, Gauss-Hermite quadrature)
- I/O utilities for saving/loading results
"""

from statphys.utils.integration import (
    conditional_expectation,
    gaussian_expectation,
    gaussian_integral_1d,
    gaussian_integral_2d,
    gaussian_integral_nd,
    logistic_gaussian_integral,
    teacher_student_integral,
)
from statphys.utils.io import ResultsManager, load_results, save_results
from statphys.utils.order_params import (
    ModelType,
    OrderParameterCalculator,
    TaskType,
    auto_calc_order_params,
    default_calculator,
    detailed_calculator,
)
from statphys.utils.math import (
    H_function,
    compute_overlap,
    double_gaussian_integral,
    erf_scaled,
    gaussian_integral,
    moreau_envelope,
    proximal_operator,
    relu,
    sigmoid,
)
from statphys.utils.seed import fix_seed, get_device
from statphys.utils.special_functions import (  # Gaussian distribution functions; Activation functions; Committee machine functions; Proximal operators; Generalization error functions
    I2,
    I3,
    I4,
    H,
    Phi,
    classification_error_linear,
    compute_stability_parameter,
    erf_activation,
    erf_derivative,
    firm_threshold,
    fisher_information_binary,
    gaussian_cdf,
    gaussian_pdf,
    gaussian_quantile,
    gaussian_tail,
    phi,
    regression_error_linear,
    relu_derivative,
    sigmoid_derivative,
    soft_threshold,
    softplus,
    tanh_derivative,
    training_error_linear,
)

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
    # Order parameter calculation
    "OrderParameterCalculator",
    "auto_calc_order_params",
    "ModelType",
    "TaskType",
    "default_calculator",
    "detailed_calculator",
]
