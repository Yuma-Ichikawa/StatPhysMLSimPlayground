"""
Central definitions of numerical constants.

All numerical-stability epsilons, clipping factors, integration bounds,
and solver defaults used across the package live here, so that tuning a
tolerance or a clip factor never requires hunting through scenario files.

Grouping:
    - EPS_*: lower bounds / division guards
    - CLIP_*: clipping factors for correlations
    - GAUSS_*: Gaussian integration domains
    - DEFAULT_*: solver and experiment defaults

"""

# --- Numerical-stability epsilons -------------------------------------------
# Lower bound for order parameters (q, rho) inside theory equations
EPS_ORDER_PARAM = 1e-6
# Guard against division by zero / log(0)
EPS_DIV = 1e-10
# Guard for vector-norm normalization
EPS_NORM = 1e-12
# Jitter added to covariance matrices before Cholesky factorization
CHOLESKY_JITTER = 1e-8

# --- Clipping factors ---------------------------------------------------------
# Keep |corr| = |m|/sqrt(q*rho) strictly inside (-1, 1) for conditional Gaussians
CORR_CLIP = 0.999

# --- Gaussian integration domains --------------------------------------------
# Integration bound in units of std; [-5, 5] captures > 99.9999% of the mass
GAUSS_INT_BOUND = 5.0
# Wider bound for integrands with polynomial growth (e.g. hinge (kappa - h)^2)
GAUSS_INT_BOUND_WIDE = 10.0
# Default number of Gauss-Hermite quadrature points
DEFAULT_GH_POINTS = 80

# --- Solver / dynamics defaults -----------------------------------------------
# Step size of heuristic gradient-flow relaxations in replica scenarios
DEFAULT_GRADFLOW_DAMPING = 0.1
# Default online-learning rate used by online ODE scenarios
DEFAULT_ONLINE_LR = 0.1

# --- Reproducibility -----------------------------------------------------------
DEFAULT_SEED = 42
