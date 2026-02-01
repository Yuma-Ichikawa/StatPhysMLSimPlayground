# Replica Method Module

This module provides solvers for **saddle-point equations** arising from the **replica trick** in statistical mechanics of learning. It enables theoretical analysis of high-dimensional learning problems in the proportional regime where both the number of samples $n$ and the dimension $d$ tend to infinity with fixed ratio $\alpha = n/d$.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Order Parameters](#order-parameters)
4. [Available Saddle-Point Equations](#available-saddle-point-equations)
5. [SaddlePointSolver](#saddlepointsolver)
6. [Integration Utilities](#integration-utilities)
7. [Usage Examples](#usage-examples)
8. [References](#references)

---

## Overview

The **replica method** is a powerful technique from statistical physics used to analyze the typical behavior of disordered systems. In machine learning, it allows us to compute:

- **Generalization error** as a function of sample complexity $\alpha$
- **Phase transitions** in learning (e.g., interpolation threshold)
- **Optimal regularization** parameters
- **Order parameters** characterizing the learned solution

### Key Components

| Component | Description |
|-----------|-------------|
| `SaddlePointSolver` | Fixed-point iteration solver with damping |
| `ReplicaEquations` | Abstract base class for saddle-point equations |
| `*Equations` classes | Pre-defined equations for common problems |
| `integration.py` | Gaussian integration utilities |

---

## Mathematical Background

### Teacher-Student Framework

We consider the **teacher-student** setup:

1. **Teacher**: Generates labels via a known rule

$$
y = f_{\text{teacher}}(\mathbf{x}; \mathbf{W}_0) + \varepsilon
$$

where $\mathbf{W}_0 \in \mathbb{R}^d$ is the teacher weight, $\varepsilon$ is noise.

2. **Student**: Learns from $n$ samples $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ by minimizing

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell\bigl(y_i, f_{\text{student}}(\mathbf{x}_i; \mathbf{w})\bigr) + \lambda R(\mathbf{w})
$$

### Scaling Convention

All models follow the **$1/\sqrt{d}$ scaling**:

$$
z = \frac{\mathbf{w}^\top \mathbf{x}}{\sqrt{d}} = O(1)
$$

This ensures that pre-activations remain $O(1)$ as $d \to \infty$.

### Replica Calculation

The replica method proceeds via:

1. **Average free energy**: 

$$
f = -\lim_{n \to 0} \frac{1}{n} \frac{\partial}{\partial n} \log \mathbb{E}[Z^n]
$$

2. **Replica symmetric ansatz**: Assume replica symmetry
3. **Saddle-point equations**: Extremize over order parameters
4. **Fixed-point iteration**: Solve self-consistent equations

---

## Order Parameters

The key order parameters characterize the learned solution:

| Symbol | Definition | Interpretation |
|--------|------------|----------------|
| $m$ | $\mathbf{w}^\top \mathbf{W}_0 / d$ | **Teacher-student overlap** (alignment) |
| $q$ | $\|\mathbf{w}\|^2 / d$ | **Self-overlap** (weight norm squared) |
| $\rho$ | $\|\mathbf{W}_0\|^2 / d$ | **Teacher norm** (signal strength) |
| $\eta$ | $\text{Var}(\varepsilon)$ | **Noise variance** |

### Generalization Error

For **regression** (MSE loss):

$$
E_g = \frac{1}{2}(\rho - 2m + q) = \frac{1}{2} \mathbb{E}\left[\left(\frac{(\mathbf{w} - \mathbf{W}_0)^\top \mathbf{x}}{\sqrt{d}}\right)^2\right]
$$

For **classification**:

$$
P(\text{error}) = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q\rho}}\right)
$$

---

## Available Saddle-Point Equations

### 1. `RidgeRegressionEquations`

**Problem**: Ridge regression with linear teacher

```python
from statphys.theory.replica import RidgeRegressionEquations

equations = RidgeRegressionEquations(
    rho=1.0,        # Teacher norm ||W₀||²/d
    eta=0.1,        # Noise variance
    reg_param=0.01  # Ridge parameter λ
)
```

**Saddle-point equations**:

Residual variance:

$$
V = \rho - 2m + q + \eta
$$

Conjugate variables:

$$
\hat{m} = \frac{\alpha \cdot m}{1 + \alpha q / (\lambda + \epsilon)}, \quad
\hat{q} = \frac{\alpha (V + m^2)}{(1 + \alpha q / (\lambda + \epsilon))^2}
$$

Update equations:

$$
m_{\text{new}} = \frac{\rho \cdot \hat{m}}{\lambda + \hat{q}}, \quad
q_{\text{new}} = \frac{\rho \cdot \hat{m}^2 + \hat{q}(\rho + \eta)}{(\lambda + \hat{q})^2}
$$

**Generalization error**: $E_g = \frac{1}{2}(\rho - 2m + q)$

---

### 2. `LassoEquations`

**Problem**: LASSO regression with L1 regularization

```python
from statphys.theory.replica import LassoEquations

equations = LassoEquations(
    rho=1.0,
    eta=0.1,
    reg_param=0.1  # L1 penalty strength
)
```

**Key feature**: Uses **soft-thresholding** proximal operator:

$$
\text{prox}_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)
$$

**Update equations** involve Gaussian integrals over the effective field distribution:

$$
m_{\text{new}} = \int \mathcal{D}z \, \text{prox}_{\lambda/\sqrt{\hat{q}}}(\omega + \sqrt{\hat{q}} z) \cdot \frac{\sqrt{\rho} \, m}{\sqrt{q}}
$$

$$
q_{\text{new}} = \int \mathcal{D}z \, \left[\text{prox}_{\lambda/\sqrt{\hat{q}}}(\omega + \sqrt{\hat{q}} z)\right]^2
$$

where $\mathcal{D}z = \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz$ is the Gaussian measure.

---

### 3. `LogisticRegressionEquations`

**Problem**: Binary classification with logistic loss

```python
from statphys.theory.replica import LogisticRegressionEquations

equations = LogisticRegressionEquations(
    rho=1.0,
    reg_param=0.01
)
```

**Teacher**: Generates binary labels $y \in \{-1, +1\}$ from a linear separator.

**Logistic loss**: $\ell(y, z) = \log(1 + e^{-yz})$

**Generalization error** (classification error rate):

$$
P(\text{error}) = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q\rho}}\right)
$$

---

### 4. `PerceptronEquations`

**Problem**: Perceptron/SVM learning with hinge loss

```python
from statphys.theory.replica import PerceptronEquations

equations = PerceptronEquations(
    rho=1.0,
    margin=0.0,     # Margin κ (for Gardner volume)
    reg_param=0.0
)
```

**Hinge loss**: $\ell(y, z) = \max(0, \kappa - yz)$

**Applications**:
- Perceptron learning rule ($\kappa = 0$)
- Support Vector Machine ($\kappa > 0$)
- Gardner storage capacity analysis

**Gardner volume** condition:

$$
y_i \cdot \frac{\mathbf{w}^\top \mathbf{x}_i}{\sqrt{d} \|\mathbf{w}\|} \geq \kappa, \quad \forall i
$$

---

### 5. `ProbitEquations`

**Problem**: Probit regression (Gaussian CDF teacher)

```python
from statphys.theory.replica import ProbitEquations

equations = ProbitEquations(
    rho=1.0,
    reg_param=0.01
)
```

**Teacher**: 

$$
P(y=1|\mathbf{x}) = \Phi\left(\frac{\mathbf{W}_0^\top \mathbf{x}}{\sqrt{d}}\right)
$$

where $\Phi(\cdot)$ is the Gaussian CDF.

**Advantage**: Gaussian integrals have closed-form solutions due to the Gaussian structure of the probit model.

---

### 6. `CommitteeMachineEquations`

**Problem**: Two-layer neural network (committee machine)

```python
from statphys.theory.replica import CommitteeMachineEquations

equations = CommitteeMachineEquations(
    K=2,            # Student hidden units
    M=2,            # Teacher hidden units
    rho=1.0,
    eta=0.0,
    activation='erf',  # 'erf', 'tanh', 'sign', 'relu'
    reg_param=0.01
)
```

**Architecture**:

$$
\text{Student}: \quad f(\mathbf{x}) = \frac{1}{\sqrt{K}} \sum_{k=1}^K \phi\left(\frac{\mathbf{v}_k^\top \mathbf{x}}{\sqrt{d}}\right)
$$

$$
\text{Teacher}: \quad y = \frac{1}{\sqrt{M}} \sum_{m=1}^M \phi\left(\frac{\mathbf{v}_m^{*\top} \mathbf{x}}{\sqrt{d}}\right)
$$

**Order parameters** (symmetric ansatz):
- $Q_{kk'} = \frac{1}{d} \mathbf{v}_k^\top \mathbf{v}_{k'}$ : Student-student overlaps
- $R_{km} = \frac{1}{d} \mathbf{v}_k^\top \mathbf{v}_m^*$ : Student-teacher overlaps  
- $T_{mm'} = \frac{1}{d} \mathbf{v}_m^{*\top} \mathbf{v}_{m'}^*$ : Teacher-teacher overlaps

---

## SaddlePointSolver

The `SaddlePointSolver` class implements **fixed-point iteration** with several advanced features:

### Basic Usage

```python
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# Create equations
equations = RidgeRegressionEquations(rho=1.0, eta=0.1, reg_param=0.01)

# Create solver
solver = SaddlePointSolver(
    equations=equations,
    order_params=['m', 'q'],
    damping=0.5,
    tol=1e-8,
    max_iter=10000,
    verbose=True
)

# Solve for range of α values
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
result = solver.solve(alpha_values, rho=1.0, eta=0.1, reg_param=0.01)

# Access results
print(result.order_params['m'])  # Teacher-student overlap
print(result.order_params['q'])  # Self-overlap
```

### Solver Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `damping` | 0.5 | Damping factor for updates: $x_{\text{new}} = \gamma f(x) + (1-\gamma) x$ |
| `adaptive_damping` | True | Automatically reduce damping when oscillating |
| `damping_decay` | 0.9 | Factor to reduce damping |
| `min_damping` | 0.01 | Minimum allowed damping |
| `tol` | $10^{-8}$ | Convergence tolerance |
| `max_iter` | 10000 | Maximum iterations |
| `n_restarts` | 3 | Random restarts for robustness |
| `use_continuation` | True | Use previous solution as initial guess |

### Solving with Generalization Error

```python
result = solver.solve_with_generalization_error(
    alpha_values,
    eg_formula=equations.generalization_error,
    rho=1.0,
    eta=0.1,
    reg_param=0.01
)

# Access generalization error
print(result.order_params['eg'])
```

### Custom Equations

You can define custom saddle-point equations:

```python
def my_equations(m, q, alpha, **params):
    rho = params.get('rho', 1.0)
    lam = params.get('reg_param', 0.01)
    
    # Your update equations here
    new_m = ...
    new_q = ...
    
    return new_m, new_q

solver = SaddlePointSolver(
    equations=my_equations,
    order_params=['m', 'q']
)
```

---

## Integration Utilities

The `integration.py` module provides numerical integration for Gaussian integrals.

### Univariate Gaussian Integral

Computes $\mathbb{E}_{z \sim \mathcal{N}(\mu, \sigma^2)}[f(z)]$:

```python
from statphys.theory.replica import gaussian_integral

# E[z²] for z ~ N(0, 1)
result = gaussian_integral(lambda z: z**2, mean=0.0, var=1.0)
# result ≈ 1.0

# Methods: 'quadrature', 'hermite', 'monte_carlo'
result = gaussian_integral(func, mean=0, var=1, method='hermite', n_points=100)
```

### Bivariate Gaussian Integral

Computes $\mathbb{E}_{(z_1, z_2) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})}[f(z_1, z_2)]$:

```python
from statphys.theory.replica import double_gaussian_integral

# E[z₁·z₂] for correlated Gaussians
result = double_gaussian_integral(
    lambda z1, z2: z1 * z2,
    mean1=0.0, mean2=0.0,
    var1=1.0, var2=1.0,
    cov=0.5  # Covariance
)
# result ≈ 0.5
```

### Proximal Operators

The **proximal operator** of a function $f$ is defined as:

$$
\text{prox}_{\gamma f}(x) = \arg\min_y \left[ f(y) + \frac{1}{2\gamma} \|x - y\|^2 \right]
$$

The **Moreau envelope** is:

$$
M_{\gamma f}(x) = \min_y \left[ f(y) + \frac{1}{2\gamma} \|x - y\|^2 \right]
$$

```python
from statphys.theory.replica import proximal_operator, moreau_envelope

# Soft thresholding (proximal of L1)
from statphys.theory.replica.integration import soft_threshold
x_thresholded = soft_threshold(x=2.0, threshold=0.5)  # = 1.5

# General proximal operator
prox_value = proximal_operator(func=lambda y: abs(y), x=2.0, gamma=1.0)

# Moreau envelope
envelope = moreau_envelope(func=lambda y: abs(y), x=2.0, gamma=1.0)
```

---

## Usage Examples

### Example 1: Ridge Regression Phase Diagram

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# Setup
rho, eta = 1.0, 0.1
lambda_values = [0.001, 0.01, 0.1, 1.0]
alpha_values = np.linspace(0.1, 5.0, 50)

fig, ax = plt.subplots()

for lam in lambda_values:
    equations = RidgeRegressionEquations(rho=rho, eta=eta, reg_param=lam)
    solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])
    
    result = solver.solve(alpha_values, rho=rho, eta=eta, reg_param=lam)
    
    # Compute generalization error
    m_vals = np.array(result.order_params['m'])
    q_vals = np.array(result.order_params['q'])
    eg = 0.5 * (rho - 2*m_vals + q_vals)
    
    ax.plot(alpha_values, eg, label=f'λ={lam}')

ax.set_xlabel(r'$\alpha = n/d$')
ax.set_ylabel(r'$E_g$ (Generalization Error)')
ax.set_title('Ridge Regression: Theory vs Sample Complexity')
ax.legend()
ax.set_yscale('log')
plt.show()
```

### Example 2: Interpolation Threshold

The **double descent** phenomenon occurs at $\alpha = 1$:

```python
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# Ridgeless regression (λ → 0)
equations = RidgeRegressionEquations(rho=1.0, eta=0.1, reg_param=1e-6)
solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])

alpha_values = np.linspace(0.5, 2.0, 100)
result = solver.solve(alpha_values, rho=1.0, eta=0.1, reg_param=1e-6)

# The generalization error diverges at α = 1 (interpolation threshold)
m_vals = np.array(result.order_params['m'])
q_vals = np.array(result.order_params['q'])
eg = 0.5 * (1.0 - 2*m_vals + q_vals)

# Plot shows peak at α = 1
```

For $\alpha < 1$ (underparameterized): unique minimum-norm interpolator  
For $\alpha > 1$ (overparameterized): generalization improves with more parameters

### Example 3: Classification Error

```python
from statphys.theory.replica import SaddlePointSolver, PerceptronEquations

equations = PerceptronEquations(rho=1.0, margin=0.0)
solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])

alpha_values = np.linspace(0.5, 10.0, 50)
result = solver.solve(alpha_values, rho=1.0)

# Classification error
m_vals = np.array(result.order_params['m'])
q_vals = np.array(result.order_params['q'])
rho = 1.0

error_rate = np.arccos(np.clip(m_vals / np.sqrt(q_vals * rho), -1, 1)) / np.pi
```

The classification error decreases as:

$$
P(\text{error}) \sim \frac{1}{\sqrt{\alpha}} \quad \text{as } \alpha \to \infty
$$

---

## References

### Foundational Papers

1. **Replica Method for Learning**
   - Seung, Sompolinsky, Tishby (1992). "Statistical mechanics of learning from examples." *Phys. Rev. A*

2. **Ridge Regression**
   - Advani, Saxe (2017). "High-dimensional dynamics of generalization error in neural networks." *arXiv:1710.03667*
   - Hastie, Montanari, Rosset, Tibshirani (2022). "Surprises in high-dimensional ridgeless least squares interpolation." *Ann. Statist.*

3. **LASSO**
   - Bayati, Montanari (2011). "The LASSO risk for Gaussian matrices." *IEEE Trans. Inf. Theory*
   - Thrampoulidis, Oymak, Hassibi (2018). "Precise error analysis of regularized M-estimators." *IEEE Trans. Inf. Theory*

4. **Classification**
   - Gardner (1988). "The space of interactions in neural network models." *J. Phys. A*
   - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics of support vector networks." *Phys. Rev. Lett.*

5. **Committee Machines**
   - Saad, Solla (1995). "Exact solution for on-line learning in multilayer neural networks." *Phys. Rev. Lett.*
   - Goldt, Mézard, Krzakala, Zdeborová (2020). "Modeling the influence of data structure on learning in neural networks." *Phys. Rev. X*

### Textbooks

- Engel, Van den Broeck (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
- Mézard, Montanari (2009). *Information, Physics, and Computation*. Oxford University Press.

---

## Default Supported Models

This module provides default saddle-point equations for the following models.

### Model List

| Model | Class | Description | File |
|-------|-------|-------------|------|
| **Ridge Regression** | `RidgeRegressionEquations` | Linear regression with L2 regularization | `models/linear.py` |
| **LASSO** | `LassoEquations` | Linear regression with L1 regularization | `models/lasso.py` |
| **Logistic Regression** | `LogisticRegressionEquations` | Binary classification (logistic loss) | `models/logistic.py` |
| **Perceptron/SVM** | `PerceptronEquations` | Perceptron/SVM (hinge loss) | `models/perceptron.py` |
| **Probit Regression** | `ProbitEquations` | Gaussian CDF teacher | `models/probit.py` |
| **Committee Machine** | `CommitteeMachineEquations` | Two-layer neural network | `models/committee.py` |

### File Structure

```
theory/replica/
├── __init__.py          # Module entry point
├── solver.py            # SaddlePointSolver
├── equations.py         # Legacy equations (backward compatibility)
├── integration.py       # Gaussian integration utilities
└── models/              # Model-specific saddle-point equations
    ├── __init__.py      # Model registry
    ├── base.py          # ReplicaEquations base class
    ├── linear.py        # RidgeRegressionEquations
    ├── lasso.py         # LassoEquations
    ├── logistic.py      # LogisticRegressionEquations
    ├── perceptron.py    # PerceptronEquations
    ├── probit.py        # ProbitEquations
    └── committee.py     # CommitteeMachineEquations
```

### Easy Model Access

```python
from statphys.theory.replica import get_replica_equations, REPLICA_MODELS

# Check available models
print(REPLICA_MODELS.keys())
# dict_keys(['ridge', 'lasso', 'logistic', 'perceptron', 'probit', 'committee'])

# Get model by name
equations = get_replica_equations("ridge", rho=1.0, eta=0.1, reg_param=0.01)
```

---

## Writing Custom Saddle-Point Equations

A detailed guide for adding new model saddle-point equations.

### Basic Structure

```python
# models/my_custom_model.py
import numpy as np
from statphys.theory.replica.models.base import ReplicaEquations

# Import special functions from utils (DO NOT implement your own)
from statphys.utils.special_functions import (
    gaussian_pdf, gaussian_cdf, gaussian_tail,
    sigmoid, soft_threshold,
    I2, I3, I4,
    classification_error_linear,
    regression_error_linear,
)
from statphys.utils.integration import (
    gaussian_integral_1d,
    gaussian_integral_2d,
    teacher_student_integral,
)


class MyCustomReplicaEquations(ReplicaEquations):
    """Custom saddle-point equations."""
    
    def __init__(self, rho=1.0, eta=0.0, reg_param=0.01, **params):
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
    
    def __call__(self, m: float, q: float, alpha: float, **kwargs) -> tuple[float, float]:
        """
        Compute fixed-point update (m_new, q_new) = F(m, q; α).
        
        The saddle-point equations in residual form: 0 = F(m,q) - (m,q)
        """
        rho = kwargs.get('rho', self.rho)
        eta = kwargs.get('eta', self.eta)
        lam = kwargs.get('reg_param', self.reg_param)
        
        V = rho - 2*m + q + eta
        denom = 1 + alpha * q / (lam + 1e-6)
        hat_m = alpha * m / denom
        hat_q = alpha * (V + m**2) / denom**2
        
        new_m = rho * hat_m / (lam + hat_q + 1e-6)
        new_q = (rho * hat_m**2 + hat_q*(rho + eta)) / (lam + hat_q + 1e-6)**2
        
        return new_m, new_q
    
    def residual(self, m: float, q: float, alpha: float, **kwargs) -> tuple[float, float]:
        """Residual form: 0 = F(m,q) - (m,q)"""
        new_m, new_q = self(m, q, alpha, **kwargs)
        return (new_m - m, new_q - q)
    
    def generalization_error(self, m: float, q: float, **kwargs) -> float:
        rho = kwargs.get('rho', self.rho)
        return regression_error_linear(m, q, rho)
```

### Using Special Functions (from utils)

**Important**: Always use functions from `statphys.utils` instead of implementing your own.

#### Available Functions (`statphys.utils.special_functions`)

| Function | Description |
|----------|-------------|
| `gaussian_pdf(x)` / `phi(x)` | Gaussian PDF |
| `gaussian_cdf(x)` / `Phi(x)` | Gaussian CDF |
| `gaussian_tail(x)` / `H(x)` | Tail probability H(x) = 1 - Φ(x) |
| `sigmoid(x)` | Sigmoid function |
| `soft_threshold(x, λ)` | Soft thresholding (L1 proximal) |
| `I2(Q, activation)` | Two-point correlation (committee) |
| `classification_error_linear(m, q, ρ)` | (1/π) arccos(m/√(qρ)) |
| `regression_error_linear(m, q, ρ)` | (1/2)(ρ - 2m + q) |

#### Available Integration Utilities (`statphys.utils.integration`)

| Function | Description |
|----------|-------------|
| `gaussian_integral_1d(f, μ, σ²)` | E[f(z)], z ~ N(μ, σ²) |
| `gaussian_integral_2d(f, μ, Σ)` | 2D Gaussian integral |
| `teacher_student_integral(f, m, q, ρ)` | E[f(u,z)] over joint |
| `conditional_expectation(f, u, m, q, ρ)` | E[f(z)|u=value] |
