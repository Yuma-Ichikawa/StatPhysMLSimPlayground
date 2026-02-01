# Online Learning Dynamics Module

This module provides **ODE solvers** for analyzing online learning dynamics in the high-dimensional limit. When data arrives sequentially and the model is updated after each sample, the learning dynamics can be described by ordinary differential equations (ODEs) for order parameters as $d \to \infty$.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [ODESolver](#odesolver)
4. [Available ODE Equations](#available-ode-equations)
5. [Usage Examples](#usage-examples)
6. [References](#references)

---

## Overview

In **online learning**, data arrives sequentially and the model is updated after observing each sample. In the high-dimensional limit $d \to \infty$ with $t = n/d$ (normalized time), the dynamics of order parameters are governed by deterministic ODEs.

### Key Components

| Component | Description |
|-----------|-------------|
| `ODESolver` | ODE solver using `scipy.integrate.solve_ivp` |
| `AdaptiveODESolver` | ODE solver with event detection |
| `OnlineEquations` | Abstract base class for ODE equations |
| `Online*Equations` | Pre-defined equations for common problems |

### Comparison with Replica Method

| Aspect | Replica Method | Online Learning |
|--------|----------------|-----------------|
| **Setting** | Batch learning (all data at once) | Sequential learning |
| **Analysis** | Fixed-point equations | ODEs |
| **Parameter** | $\alpha = n/d$ (sample ratio) | $t = n/d$ (normalized time) |
| **Result** | Equilibrium properties | Learning dynamics/trajectory |

---

## Mathematical Background

### Online SGD Setup

Consider the **teacher-student** framework with online SGD:

1. **Data stream**: At each step $\mu$, receive sample $(\mathbf{x}^\mu, y^\mu)$
2. **SGD update**:

$$
\mathbf{w}^{\mu+1} = \mathbf{w}^\mu - \eta \nabla_\mathbf{w} \ell(y^\mu, f(\mathbf{x}^\mu; \mathbf{w}^\mu))
$$

### High-Dimensional Limit

In the limit $d \to \infty$ with normalized time $t = \mu / d$, the order parameters evolve according to deterministic ODEs:

$$
\frac{dm}{dt} = F_m(m, q; \eta, \rho, \lambda, \ldots)
$$

$$
\frac{dq}{dt} = F_q(m, q; \eta, \rho, \lambda, \ldots)
$$

### Order Parameters

| Symbol | Definition | Interpretation |
|--------|------------|----------------|
| $m(t)$ | $\mathbf{w}(t)^\top \mathbf{W}_0 / d$ | Teacher-student overlap |
| $q(t)$ | $\|\mathbf{w}(t)\|^2 / d$ | Self-overlap (weight norm) |
| $\eta$ | Learning rate | Step size |
| $\rho$ | $\|\mathbf{W}_0\|^2 / d$ | Teacher norm |

### Generalization Error

For **regression**:

$$
E_g(t) = \frac{1}{2}(\rho - 2m(t) + q(t))
$$

For **classification**:

$$
P(\text{error}, t) = \frac{1}{\pi} \arccos\left(\frac{m(t)}{\sqrt{q(t)\rho}}\right)
$$

---

## ODESolver

The `ODESolver` class wraps `scipy.integrate.solve_ivp` for solving online learning ODEs.

### Basic Usage

```python
from statphys.theory.online import ODESolver, OnlineSGDEquations

# Create equations
equations = OnlineSGDEquations(
    rho=1.0,        # Teacher norm
    eta_noise=0.1,  # Noise variance
    lr=0.5,         # Learning rate
    reg_param=0.0   # Regularization
)

# Create solver
solver = ODESolver(
    equations=equations,
    order_params=['m', 'q'],
    method='RK45',
    tol=1e-8,
    verbose=True
)

# Solve ODE
result = solver.solve(
    t_span=(0, 10),           # Time range
    init_values=(0.0, 0.01),  # Initial (m₀, q₀)
    n_points=100              # Output points
)

# Access results
t_values = result.param_values  # Time points
m_values = result.order_params['m']  # m(t) trajectory
q_values = result.order_params['q']  # q(t) trajectory
```

### Solver Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `equations` | - | ODE right-hand side function |
| `order_params` | - | Names of order parameters |
| `method` | `'RK45'` | Integration method (`'RK45'`, `'RK23'`, `'Radau'`, `'BDF'`, `'LSODA'`) |
| `tol` | $10^{-8}$ | Relative tolerance |
| `max_step` | 0.1 | Maximum step size |

### Solving with Generalization Error

```python
result = solver.solve_with_generalization_error(
    t_span=(0, 10),
    eg_formula=equations.generalization_error,
    init_values=(0.0, 0.01),
    rho=1.0
)

# Access E_g trajectory
eg_values = result.order_params['eg']
```

### Comparing Multiple Learning Rates

```python
results = solver.solve_multiple_lr(
    t_span=(0, 10),
    learning_rates=[0.1, 0.5, 1.0, 2.0],
    init_values=(0.0, 0.01),
    rho=1.0
)

for lr, result in results.items():
    print(f"lr={lr}: final m = {result.order_params['m'][-1]:.4f}")
```

### AdaptiveODESolver with Event Detection

```python
from statphys.theory.online.solver import AdaptiveODESolver

# Define stopping event: stop when E_g < threshold
def eg_threshold_event(t, y, params):
    m, q = y
    rho = params.get('rho', 1.0)
    eg = 0.5 * (rho - 2*m + q)
    return eg - 0.01  # Stop when E_g = 0.01

solver = AdaptiveODESolver(
    equations=equations,
    order_params=['m', 'q'],
    events=[eg_threshold_event]
)

result = solver.solve(t_span=(0, 100), init_values=(0.0, 0.01), rho=1.0)
# Stops early if E_g reaches 0.01
```

---

## Available ODE Equations

### 1. `OnlineSGDEquations`

**Problem**: Online SGD for linear regression with MSE loss

```python
from statphys.theory.online import OnlineSGDEquations

equations = OnlineSGDEquations(
    rho=1.0,        # Teacher norm
    eta_noise=0.1,  # Noise variance σ²
    lr=0.5,         # Learning rate η
    reg_param=0.01  # L2 regularization λ
)
```

**ODE System**:

$$
\frac{dm}{dt} = \eta(\rho - m) - \eta\lambda m
$$

$$
\frac{dq}{dt} = \eta^2 V + 2\eta(m - q) - 2\eta\lambda q
$$

where $V = \rho - 2m + q + \sigma^2$ is the residual variance (training loss).

**Key dynamics**:
- $m(t) \to m^*$: Approaches optimal overlap
- $q(t) \to q^*$: Approaches optimal norm
- Convergence rate depends on $\eta$ and $\lambda$

---

### 2. `OnlinePerceptronEquations`

**Problem**: Online perceptron learning for binary classification

```python
from statphys.theory.online import OnlinePerceptronEquations

equations = OnlinePerceptronEquations(
    rho=1.0,   # Teacher norm
    lr=1.0     # Learning rate
)
```

**ODE System** (Saad & Solla style):

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \frac{\phi(\kappa)}{\sqrt{q}}
$$

$$
\frac{dq}{dt} = 2\eta^2 \epsilon(\kappa)
$$

where:
- $\kappa = m / \sqrt{q\rho}$ is the stability parameter
- $\phi(\kappa) = \frac{1}{\sqrt{2\pi}} e^{-\kappa^2/2}$ is the Gaussian PDF
- $\epsilon(\kappa) = H(\kappa)$ is the error rate (complementary Gaussian CDF)

**Classification error**:

$$
P(\text{error}) = \frac{1}{\pi} \arccos(\kappa)
$$

---

### 3. `OnlineRidgeEquations`

**Problem**: Online ridge regression (alias for `OnlineSGDEquations`)

```python
from statphys.theory.online import OnlineRidgeEquations

equations = OnlineRidgeEquations(
    rho=1.0,
    eta_noise=0.1,
    lr=0.5,
    reg_param=0.1  # Ridge parameter λ
)
```

Same ODE system as `OnlineSGDEquations` with explicit ridge regularization.

---

### 4. `OnlineLogisticEquations`

**Problem**: Online logistic regression for binary classification

```python
from statphys.theory.online import OnlineLogisticEquations

equations = OnlineLogisticEquations(
    rho=1.0,
    lr=0.1,
    reg_param=0.01
)
```

**Logistic loss**: $\ell(y, z) = \log(1 + e^{-yz})$

**ODE System**:

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \mathbb{E}[g(y,z) \cdot u / \sqrt{\rho}] - \eta\lambda m
$$

$$
\frac{dq}{dt} = 2\eta \sqrt{q} \cdot \mathbb{E}[g(y,z) \cdot z / \sqrt{q}] + \eta^2 \mathbb{E}[g(y,z)^2] - 2\eta\lambda q
$$

where $g(y, z) = y \cdot \sigma(-yz)$ is the logistic gradient and expectations are over the joint distribution of teacher field $u$ and student field $z$.

---

### 5. `OnlineHingeEquations`

**Problem**: Online SVM with hinge loss

```python
from statphys.theory.online import OnlineHingeEquations

equations = OnlineHingeEquations(
    rho=1.0,
    lr=0.1,
    margin=1.0,     # Hinge margin κ
    reg_param=0.01
)
```

**Hinge loss**: $\ell(y, z) = \max(0, \kappa - yz)$

**ODE System**:

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \frac{\phi(\theta)}{\Delta} - \eta\lambda m
$$

$$
\frac{dq}{dt} = 2\eta^2 H(-\theta) - 2\eta\lambda q
$$

where $\Delta$ and $\theta$ are derived from the margin condition.

---

### 6. `OnlineCommitteeEquations`

**Problem**: Online learning of committee machine (two-layer network)

```python
from statphys.theory.online import OnlineCommitteeEquations

equations = OnlineCommitteeEquations(
    k_student=2,      # Student hidden units
    k_teacher=2,      # Teacher hidden units
    rho=1.0,
    lr=0.1,
    activation='erf'  # 'erf', 'relu'
)
```

**Note**: This class provides a template. Full implementation requires handling overlap matrices $Q_{ij}$ (student-student) and $R_{in}$ (student-teacher) following Saad & Solla (1995).

**Order parameters**:
- $Q_{ij} = \frac{1}{d} \mathbf{w}_i^\top \mathbf{w}_j$ : Student-student overlaps
- $R_{in} = \frac{1}{d} \mathbf{w}_i^\top \mathbf{W}_n^*$ : Student-teacher overlaps

---

## Usage Examples

### Example 1: Learning Rate Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.online import ODESolver, OnlineSGDEquations

# Setup
rho = 1.0
eta_noise = 0.1
t_span = (0, 20)
learning_rates = [0.1, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for lr in learning_rates:
    equations = OnlineSGDEquations(rho=rho, eta_noise=eta_noise, lr=lr)
    solver = ODESolver(equations=equations, order_params=['m', 'q'])
    
    result = solver.solve(t_span=t_span, init_values=(0.0, 0.01), n_points=200)
    
    t = result.param_values
    m = result.order_params['m']
    q = result.order_params['q']
    eg = 0.5 * (rho - 2*np.array(m) + np.array(q))
    
    axes[0].plot(t, m, label=f'η={lr}')
    axes[1].plot(t, eg, label=f'η={lr}')

axes[0].set_xlabel('t = n/d')
axes[0].set_ylabel('m(t)')
axes[0].set_title('Teacher-Student Overlap')
axes[0].legend()

axes[1].set_xlabel('t = n/d')
axes[1].set_ylabel('$E_g(t)$')
axes[1].set_title('Generalization Error')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Example 2: Optimal Learning Rate

The optimal learning rate for online SGD without regularization is:

$$
\eta^* = \frac{1}{1 + \sigma^2/\rho}
$$

```python
import numpy as np
from statphys.theory.online import ODESolver, OnlineSGDEquations

rho = 1.0
sigma_sq = 0.1
eta_optimal = 1 / (1 + sigma_sq / rho)
print(f"Optimal learning rate: η* = {eta_optimal:.4f}")

# Compare optimal vs suboptimal
for lr in [0.5 * eta_optimal, eta_optimal, 2.0 * eta_optimal]:
    equations = OnlineSGDEquations(rho=rho, eta_noise=sigma_sq, lr=lr)
    solver = ODESolver(equations=equations, order_params=['m', 'q'])
    result = solver.solve(t_span=(0, 50), init_values=(0.0, 0.01))
    
    final_m = result.order_params['m'][-1]
    final_q = result.order_params['q'][-1]
    final_eg = 0.5 * (rho - 2*final_m + final_q)
    
    print(f"η={lr:.4f}: final E_g = {final_eg:.6f}")
```

### Example 3: Perceptron Learning Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.online import ODESolver, OnlinePerceptronEquations

equations = OnlinePerceptronEquations(rho=1.0, lr=1.0)
solver = ODESolver(equations=equations, order_params=['m', 'q'])

result = solver.solve(t_span=(0, 50), init_values=(0.01, 0.01), n_points=500)

t = np.array(result.param_values)
m = np.array(result.order_params['m'])
q = np.array(result.order_params['q'])

# Classification error
kappa = m / np.sqrt(q * 1.0)
error_rate = np.arccos(np.clip(kappa, -1, 1)) / np.pi

plt.figure(figsize=(8, 5))
plt.plot(t, error_rate)
plt.xlabel('t = n/d')
plt.ylabel('Classification Error')
plt.title('Online Perceptron Learning')
plt.yscale('log')
plt.grid(True)
plt.show()
```

### Example 4: Custom ODE Equations

```python
import numpy as np
from statphys.theory.online import ODESolver

def custom_equations(t, y, params):
    """Custom ODE for online learning with momentum."""
    m, q, v_m, v_q = y  # Include momentum variables
    
    rho = params.get('rho', 1.0)
    lr = params.get('lr', 0.1)
    beta = params.get('momentum', 0.9)
    
    # Gradient
    grad_m = rho - m
    grad_q = m - q
    
    # Momentum update
    dv_m = beta * v_m + lr * grad_m
    dv_q = beta * v_q + lr * grad_q
    
    # Parameter update
    dm = dv_m
    dq = 2 * dv_q
    
    return np.array([dm, dq, dv_m - v_m, dv_q - v_q])

solver = ODESolver(
    equations=custom_equations,
    order_params=['m', 'q', 'v_m', 'v_q']
)

result = solver.solve(
    t_span=(0, 20),
    init_values=(0.0, 0.01, 0.0, 0.0),
    rho=1.0, lr=0.1, momentum=0.9
)
```

---

## References

### Foundational Papers

1. **Online Learning Theory**
   - Saad, Solla (1995). "On-line learning in soft committee machines." *Phys. Rev. E*
   - Biehl, Schwarze (1995). "Learning by on-line gradient descent." *J. Phys. A*

2. **Linear Models**
   - Werfel, Xie, Seung (2005). "Learning curves for stochastic gradient descent in linear feedforward networks." *Neural Computation*

3. **Perceptron**
   - Opper (1996). "Online versus offline learning from random examples." *Europhys. Lett.*
   - Kinzel, Opper (1991). "Dynamics of learning." *Physics of Neural Networks*

4. **General Framework**
   - Engel, Van den Broeck (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
   - Chapter on "On-line Learning"

### Key Results

- **Optimal learning rate** for online SGD: $\eta^* = 1/(1 + \sigma^2/\rho)$
- **Asymptotic error** decay: $E_g(t) \sim 1/t$ for optimal $\eta$
- **Critical learning rate**: $\eta_c = 2$ (divergence threshold for SGD without regularization)

---

## Default Supported Models

This module provides default ODE equations for the following models.

### Model List

| Model | Class | Description | File |
|-------|-------|-------------|------|
| **Linear Regression (SGD)** | `OnlineSGDEquations` | Online SGD with MSE loss | `models/linear.py` |
| **Ridge Regression** | `OnlineRidgeEquations` | Online regression with L2 regularization | `models/linear.py` |
| **Perceptron** | `OnlinePerceptronEquations` | Online perceptron learning | `models/perceptron.py` |
| **Logistic Regression** | `OnlineLogisticEquations` | Online logistic regression | `models/logistic.py` |
| **SVM (Hinge Loss)** | `OnlineHingeEquations` | Online SVM/hinge loss learning | `models/hinge.py` |
| **Committee Machine** | `OnlineCommitteeEquations` | Two-layer network (template) | `models/committee.py` |

### File Structure

```
theory/online/
├── __init__.py          # Module entry point
├── solver.py            # ODESolver, AdaptiveODESolver
├── equations.py         # Legacy equations (backward compatibility)
├── README.md            # English documentation
├── README_ja.md         # Japanese documentation
└── models/              # Model-specific ODE equations
    ├── __init__.py      # Model registry
    ├── base.py          # OnlineEquations base class
    ├── linear.py        # OnlineSGDEquations, OnlineRidgeEquations
    ├── perceptron.py    # OnlinePerceptronEquations
    ├── logistic.py      # OnlineLogisticEquations
    ├── hinge.py         # OnlineHingeEquations
    └── committee.py     # OnlineCommitteeEquations
```

### Easy Model Access

```python
from statphys.theory.online import get_online_equations, ONLINE_MODELS

# Check available models
print(ONLINE_MODELS.keys())
# dict_keys(['sgd', 'ridge', 'perceptron', 'logistic', 'hinge', 'committee'])

# Get model by name
equations = get_online_equations("sgd", rho=1.0, lr=0.5)
```

---

## Writing Custom ODE Equations

A detailed guide for adding new model ODE equations.

### Basic Structure

```python
# models/my_custom_model.py
import numpy as np
from statphys.theory.online.models.base import OnlineEquations

# Import special functions from utils (DO NOT implement your own)
from statphys.utils.special_functions import (
    gaussian_pdf, gaussian_cdf, gaussian_tail,  # Gaussian functions
    sigmoid, erf_activation,                     # Activation functions
    soft_threshold,                              # Proximal operators
    I2, I3, I4,                                  # Committee machine correlations
    classification_error_linear,                 # Classification error
    regression_error_linear,                     # Regression error
)
from statphys.utils.integration import (
    gaussian_integral_1d,      # 1D Gaussian integral
    gaussian_integral_2d,      # 2D Gaussian integral
    teacher_student_integral,  # Teacher-Student field integral
    conditional_expectation,   # Conditional expectation
)


class MyCustomOnlineEquations(OnlineEquations):
    """Custom ODE equations for online learning."""
    
    def __init__(self, rho=1.0, eta_noise=0.0, lr=0.1, reg_param=0.0, **params):
        super().__init__(rho=rho, eta_noise=eta_noise, lr=lr, reg_param=reg_param, **params)
        self.rho = rho
        self.eta_noise = eta_noise
        self.lr = lr
        self.reg_param = reg_param
    
    def __call__(self, t: float, y: np.ndarray, params: dict) -> np.ndarray:
        """
        Compute ODE right-hand side dy/dt.
        
        Args:
            t: Normalized time t = τ/d
            y: Order parameters [m, q, ...] as numpy array
            params: Parameter dict (can override __init__ values)
        
        Returns:
            dy/dt = [dm/dt, dq/dt, ...] as numpy array
        """
        m, q = y
        rho = params.get('rho', self.rho)
        lr = params.get('lr', self.lr)
        lam = params.get('reg_param', self.reg_param)
        eta_noise = params.get('eta_noise', self.eta_noise)
        
        V = rho - 2*m + q + eta_noise  # Residual variance
        
        # Implement your ODE equations here
        dm_dt = lr * (rho - m) - lr * lam * m
        dq_dt = lr**2 * V + 2*lr*(m - q) - 2*lr*lam*q
        
        return np.array([dm_dt, dq_dt])
    
    def generalization_error(self, y: np.ndarray, **kwargs) -> float:
        m, q = y
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
| `gaussian_tail(x)` / `H(x)` | Tail probability |
| `sigmoid(x)` | Sigmoid function |
| `soft_threshold(x, λ)` | Soft thresholding |
| `classification_error_linear(m, q, ρ)` | Classification error |
| `regression_error_linear(m, q, ρ)` | Regression error |

#### Available Integration Utilities (`statphys.utils.integration`)

| Function | Description |
|----------|-------------|
| `gaussian_integral_1d(f, μ, σ²)` | E[f(z)], z ~ N(μ, σ²) |
| `gaussian_integral_2d(f, μ, Σ)` | 2D Gaussian integral |
| `teacher_student_integral(f, m, q, ρ)` | Teacher-Student integral |
| `conditional_expectation(f, u, m, q, ρ)` | E[f(z)|u] |
