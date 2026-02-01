# Online SGD Learning Dynamics: Dynamical Asymptotic Analysis

This note provides a comprehensive overview of the **dynamical asymptotic analysis** for online SGD in the high-dimensional limit.

---

## 1. Introduction

While the replica method analyzes **static properties** (post-training), dynamical analysis tracks the **entire learning trajectory**. This enables understanding of:

- Convergence rates and learning curves
- Transient dynamics and phase transitions
- Optimal learning rate schedules
- Early stopping strategies

The key insight is that high-dimensional stochastic dynamics concentrate to **deterministic ODEs** in the macroscopic variables.

---

## 2. General Framework

### 2.1 Online SGD Update Rule

At each discrete time step $t$, parameters are updated using a **single fresh sample** $(\mathbf{x}^t, y^t)$:

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \tau \nabla_{\boldsymbol{\theta}^t} r(\boldsymbol{\theta}^t; \mathbf{x}^t, y^t)$$

where:
- $\tau$: learning rate (step size)
- $r(\cdot)$: per-sample loss function

**Critical distinction from batch SGD:** Each sample is used exactly once, corresponding to infinite dataset streaming.

### 2.2 Macroscopic State Variables

Define the **macroscopic state** $\phi^t$ as a collection of summary statistics:

$$\phi^t = \{\phi_1^t, \phi_2^t, \ldots, \phi_M^t\}$$

For neural networks, these are typically **order parameters**:
- Student-teacher overlaps
- Student self-overlaps
- Other sufficient statistics

### 2.3 Stochastic Process Formulation

The macroscopic state evolves as a **stochastic process**:

$$\phi^{t+1} = \phi^t + \Delta \phi^{t+1}$$

where $\Delta \phi^{t+1}$ depends on $\phi^t$ and the random sample $(\mathbf{x}^t, y^t)$.

---

## 3. Concentration Theorem: From Stochastic to Deterministic

### 3.1 Key Conditions

For the stochastic process $\phi^t$ to concentrate to a deterministic trajectory, three conditions must hold:

**Condition 1: Mean-Incremental Concentration**
$$\mathbb{E}\left[\left\|\mathbb{E}_t[\phi^{t+1}] - \phi^t - \frac{1}{d}\mathbf{g}(\phi^t)\right\|\right] \leq \frac{C(T)}{d^{1+\epsilon_1}}$$

The conditional mean increment matches the ODE drift vector $\mathbf{g}$ up to $O(d^{-1-\epsilon_1})$ error.

**Condition 2: Mean-Variance Concentration**
$$\mathbb{E}\left[\left\|\phi^{t+1} - \mathbb{E}_t[\phi^{t+1}]\right\|^2\right] \leq \frac{C(T)}{d^{1+2\epsilon_2}}$$

The variance of increments decays faster than $O(d^{-1})$.

**Condition 3: Initial State Concentration**
$$\mathbb{E}\left[\|\phi^0 - \phi^*\|\right] \leq \frac{C}{d^{\epsilon_0}}$$

The initial state concentrates to a deterministic value.

### 3.2 Main Theorem

**Theorem (Concentration to ODE):** Under Conditions 1-3 with $\epsilon_1, \epsilon_2 > 0$ and $\epsilon_0 > 1/2$, for any fixed $T > 0$:

$$\max_{0 \leq t \leq dT} \mathbb{E}\|\phi^t - \phi(t/d)\| \leq \frac{C(T)}{\sqrt{d}}$$

where $\phi(t)$ is the unique solution of the **macroscopic ODE**:

$$\frac{d\phi}{dt} = \mathbf{g}(\phi)$$

### 3.3 Interpretation

- **Time rescaling:** Discrete time $t$ maps to continuous time $t/d$
- **Concentration rate:** $O(1/\sqrt{d})$ error
- **Deterministic dynamics:** Stochastic fluctuations vanish as $d \to \infty$

---

## 4. Derivation of Macroscopic ODEs

### 4.1 General Procedure

1. **Identify macroscopic variables** $\phi$ (order parameters)
2. **Compute conditional expectation** $\mathbb{E}_t[\phi^{t+1}|\phi^t]$
3. **Extract the $O(1/d)$ drift:** $\mathbf{g}(\phi) = d \cdot \mathbb{E}_t[\phi^{t+1} - \phi^t]$
4. **Verify concentration conditions**

### 4.2 Key Mathematical Tools

**Gaussian Equivalence:** For i.i.d. $\mathbf{x} \sim \mathcal{N}(0, I_d)$:

$$\mathbb{E}\left[f\left(\frac{\mathbf{w}^\top \mathbf{x}}{\sqrt{d}}\right)\right] = \mathbb{E}[f(\xi)], \quad \xi \sim \mathcal{N}(0, q)$$

where $q = \|\mathbf{w}\|^2/d$.

**Stein's Lemma:** For $\xi \sim \mathcal{N}(0, \sigma^2)$ and differentiable $g$:

$$\mathbb{E}[\xi \cdot g(\xi)] = \sigma^2 \mathbb{E}[g'(\xi)]$$

---

## 5. Case Study: Narrow Two-Layer Neural Networks

### 5.1 Model Architecture

A **narrow two-layer network** with $k = \Theta(1)$ hidden units:

$$\hat{y}(\mathbf{x}; \mathbf{a}, W, \mathbf{b}) = \mathbf{a}^\top \sigma\left(\frac{1}{\sqrt{d}} W^\top \mathbf{x} + \mathbf{b}\right)$$

where:
- $\mathbf{a} \in \mathbb{R}^k$: second-layer weights (fixed or trained)
- $W = (\mathbf{w}_l)_{l \in [k]} \in \mathbb{R}^{d \times k}$: first-layer weights
- $\mathbf{b} \in \mathbb{R}^k$: bias terms
- $\sigma: \mathbb{R} \to \mathbb{R}$: activation function (applied element-wise)

### 5.2 Teacher-Student Setup

**Teacher network** with $k^* = \Theta(1)$ hidden units:

$$y^* = (\mathbf{a}^*)^\top \sigma\left(\frac{1}{\sqrt{d}} (W^*)^\top \mathbf{x} + \mathbf{b}^*\right)$$

**Observed label with noise:**

$$y = y^* + \sigma_\epsilon \xi, \quad \xi \sim \mathcal{N}(0, 1)$$

### 5.3 Order Parameters

**Student self-overlap matrix** $Q \in \mathbb{R}^{k \times k}$:
$$Q_{ls} = \frac{1}{d} \mathbf{w}_l^\top \mathbf{w}_s$$

**Student-teacher overlap matrix** $M \in \mathbb{R}^{k \times k^*}$:
$$M_{lm} = \frac{1}{d} \mathbf{w}_l^\top \mathbf{w}_m^*$$

**Teacher self-overlap matrix** $P \in \mathbb{R}^{k^* \times k^*}$:
$$P_{mn} = \frac{1}{d} (\mathbf{w}_m^*)^\top \mathbf{w}_n^*$$

### 5.4 Pre-Activation Variables

For a data point $\mathbf{x}$, define:

**Student pre-activation:**
$$\boldsymbol{\Upsilon} = \frac{1}{\sqrt{d}} W^\top \mathbf{x} + \mathbf{b} \in \mathbb{R}^k$$

**Teacher pre-activation:**
$$\boldsymbol{\Upsilon}^* = \frac{1}{\sqrt{d}} (W^*)^\top \mathbf{x} + \mathbf{b}^* \in \mathbb{R}^{k^*}$$

**Joint distribution (LLMM data model):**
$$\begin{pmatrix} \boldsymbol{\Upsilon} \\ \boldsymbol{\Upsilon}^* \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mathbf{b} \\ \mathbf{b}^* \end{pmatrix}, \begin{pmatrix} Q & M \\ M^\top & P \end{pmatrix}\right)$$

### 5.5 Gradient Computation

**Prediction error:**
$$\Gamma = \hat{y} - y = \mathbf{a}^\top \sigma(\boldsymbol{\Upsilon}) - (\mathbf{a}^*)^\top \sigma(\boldsymbol{\Upsilon}^*) - \sigma_\epsilon \xi$$

**Gradient with respect to first-layer weights:**
$$\nabla_{\mathbf{w}_l} r = \frac{\partial r}{\partial \hat{y}} \cdot a_l \sigma'(\Upsilon_l) \cdot \frac{\mathbf{x}}{\sqrt{d}} + \lambda \nabla g(\mathbf{w}_l)$$

For squared loss $r = \frac{1}{2}(\hat{y} - y)^2$:
$$\frac{\partial r}{\partial \hat{y}} = \Gamma$$

### 5.6 SGD Update for First-Layer Weights

$$\mathbf{w}_l^{t+1} = \mathbf{w}_l^t - \tau \left(\Gamma \cdot a_l \sigma'(\Upsilon_l^t) \cdot \frac{\mathbf{x}^t}{\sqrt{d}} + \lambda \nabla g(\mathbf{w}_l^t)\right)$$

### 5.7 ODE for Order Parameter Q

**Update for $Q_{ls}$:**

$$Q_{ls}^{t+1} = Q_{ls}^t - \frac{\tau}{d}\left(\Gamma \cdot a_l \sigma'(\Upsilon_l^t) \cdot \frac{(\mathbf{w}_s^t)^\top \mathbf{x}^t}{\sqrt{d}} + a_s \sigma'(\Upsilon_s^t) \cdot \frac{(\mathbf{w}_l^t)^\top \mathbf{x}^t}{\sqrt{d}}\right) + \frac{\tau^2}{d} \Gamma^2 a_l a_s \sigma'(\Upsilon_l^t) \sigma'(\Upsilon_s^t) - \text{(reg. terms)}$$

Taking expectation and $d \to \infty$:

$$\boxed{\frac{dQ}{dt} = -\tau \mathbb{E}\left[\Gamma \cdot (\mathbf{a} \odot \sigma'(\boldsymbol{\Upsilon}))\boldsymbol{\Upsilon}^\top + \boldsymbol{\Upsilon}(\mathbf{a} \odot \sigma'(\boldsymbol{\Upsilon}))^\top\right] + \tau^2 \mathbb{E}\left[\Gamma^2 (\mathbf{a} \odot \sigma'(\boldsymbol{\Upsilon}))(\mathbf{a} \odot \sigma'(\boldsymbol{\Upsilon}))^\top\right] - 2\tau\lambda Q}$$

where $\odot$ denotes element-wise multiplication.

### 5.8 ODE for Order Parameter M

Similarly:

$$\boxed{\frac{dM}{dt} = -\tau \mathbb{E}\left[\Gamma \cdot (\mathbf{a} \odot \sigma'(\boldsymbol{\Upsilon})) (\boldsymbol{\Upsilon}^*)^\top\right] - \tau\lambda M}$$

### 5.9 Gaussian Expectations

The expectations involve Gaussian integrals over the joint distribution:

$$\mathbb{E}[\cdot] = \mathbb{E}_{(\boldsymbol{\Upsilon}, \boldsymbol{\Upsilon}^*, \xi) \sim \mathcal{N}}[\cdot]$$

These can be computed using:
- **Gauss-Hermite quadrature** for low-dimensional integrals
- **Monte Carlo sampling** for higher dimensions
- **Analytical formulas** for specific activations (linear, ReLU)

---

## 6. Special Case: Linear Student and Teacher

### 6.1 Model Simplification

For linear models ($\sigma(x) = x$, $k = k^* = 1$, $\mathbf{a} = \mathbf{a}^* = 1$, $\mathbf{b} = \mathbf{b}^* = 0$):

- $Q = q \in \mathbb{R}$ (student self-overlap)
- $M = m \in \mathbb{R}$ (student-teacher overlap)  
- $P = \rho \in \mathbb{R}$ (teacher norm)

### 6.2 Prediction and Error

$$\hat{y} = \frac{\mathbf{w}^\top \mathbf{x}}{\sqrt{d}}, \quad y = \frac{(\mathbf{w}^*)^\top \mathbf{x}}{\sqrt{d}} + \sigma_\epsilon \xi$$

$$\Gamma = \frac{(\mathbf{w} - \mathbf{w}^*)^\top \mathbf{x}}{\sqrt{d}} - \sigma_\epsilon \xi$$

### 6.3 Residual Variance

$$V = \mathbb{E}[\Gamma^2] = \rho - 2m + q + \sigma_\epsilon^2$$

This represents the **expected training loss** (without regularization).

### 6.4 Linear Regression ODEs

**ODE for $m$:**

$$\boxed{\frac{dm}{dt} = \tau(\rho - m) - \tau\lambda m = \tau\rho - \tau(1 + \lambda)m}$$

**ODE for $q$:**

$$\boxed{\frac{dq}{dt} = \tau^2 V + 2\tau(m - q) - 2\tau\lambda q = \tau^2 V + 2\tau m - 2\tau(1 + \lambda)q}$$

### 6.5 Analysis of Linear ODEs

**Steady state ($\lambda > 0$):**

$$m^* = \frac{\rho}{1 + \lambda}, \quad q^* = \frac{\rho}{1+\lambda} + \frac{\tau \sigma_\epsilon^2}{2(1+\lambda) - \tau(1+\lambda)^2}$$

**Stability condition:**

$$\tau < \tau_c = \frac{2}{1 + \lambda}$$

**Generalization error dynamics:**

$$E_g(t) = \frac{1}{2}(\rho - 2m(t) + q(t))$$

**Steady-state generalization error:**

$$E_g^* = \frac{\lambda^2 \rho}{2(1+\lambda)^2} + \frac{\tau \sigma_\epsilon^2}{4(1+\lambda) - 2\tau(1+\lambda)^2}$$

---

## 7. ReLU Activation: Analytical Treatment

### 7.1 ReLU Properties

For $\sigma(x) = \max(0, x)$:
- $\sigma'(x) = \mathbf{1}_{x > 0}$ (Heaviside step function)
- $\sigma(x) \cdot \sigma'(x) = \sigma(x)$

### 7.2 Useful Gaussian Integrals

For $(u, v)^\top \sim \mathcal{N}(\mathbf{0}, \begin{pmatrix} a & c \\ c & b \end{pmatrix})$:

$$\mathbb{E}[\sigma(u)\sigma(v)] = \frac{1}{2\pi}\left(\sqrt{ab - c^2} + c \arcsin\left(\frac{c}{\sqrt{ab}}\right)\right)$$

$$\mathbb{E}[\sigma(u)v] = \frac{c}{2}$$

$$\mathbb{E}[\sigma'(u)\sigma'(v)] = \frac{1}{2\pi}\arccos\left(-\frac{c}{\sqrt{ab}}\right)$$

---

## 8. Implementation in StatPhys-ML

### 8.1 Linear Regression Example

```python
from statphys.theory.online import ODESolver, GaussianLinearMseEquations

# Define ODE equations
equations = GaussianLinearMseEquations(
    rho=1.0,        # Teacher norm ||w*||^2/d
    eta_noise=0.1,  # Noise variance
    lr=0.5,         # Learning rate
    reg_param=0.01, # Regularization
)

# Create solver
solver = ODESolver(
    equations=equations,
    order_params=["m", "q"],
)

# Solve ODE
result = solver.solve(
    t_span=(0, 10),
    init_values=(0.0, 0.1),  # Initial (m0, q0)
    n_points=100,
)

# Extract learning curve
t = result.t_values
m = result.order_params["m"]
q = result.order_params["q"]
eg = 0.5 * (equations.rho - 2*m + q)

import matplotlib.pyplot as plt
plt.plot(t, eg)
plt.xlabel("Time t")
plt.ylabel("Generalization Error")
plt.title("Online Learning Dynamics")
plt.show()
```

### 8.2 Custom ODE Equations

```python
from statphys.theory.online.scenario.base import OnlineEquations
import numpy as np

class TwoLayerReLUEquations(OnlineEquations):
    def __init__(self, teacher_overlap_P, a_student, a_teacher, lr=0.1, reg_param=0.0):
        self.P = teacher_overlap_P
        self.a = a_student
        self.a_star = a_teacher
        self.lr = lr
        self.reg = reg_param
        
        self.k = len(a_student)
        self.k_star = len(a_teacher)
        
    def __call__(self, t, state):
        # Unpack state: Q (k*k), M (k*k*)
        Q = state[:self.k**2].reshape(self.k, self.k)
        M = state[self.k**2:].reshape(self.k, self.k_star)
        
        # Compute expectations (using Gaussian integrals)
        dQ = self._compute_dQ(Q, M)
        dM = self._compute_dM(Q, M)
        
        return np.concatenate([dQ.flatten(), dM.flatten()])
    
    def _compute_dQ(self, Q, M):
        # Implement ODE for Q using numerical integration
        # dQ/dt = -tau E[Gamma (a*sigma')Y^T + ...] + tau^2 E[...] - 2*tau*lambda*Q
        pass
        
    def _compute_dM(self, Q, M):
        # Implement ODE for M
        # dM/dt = -tau E[Gamma (a*sigma')(Y*)^T] - tau*lambda*M
        pass
```

### 8.3 Running Simulations with Theory Comparison

```python
from statphys.simulation import OnlineSimulation, SimulationConfig
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import MSELoss
from statphys.vis import ComparisonPlotter

# Setup
dataset = GaussianDataset(d=1000, rho=1.0, eta=0.1)
config = SimulationConfig.for_online(
    t_max=10.0,
    lr=0.1,
    n_seeds=5,
    use_theory=True,
)

# Theory solver
from statphys.theory.online import ODESolver, GaussianLinearMseEquations
equations = GaussianLinearMseEquations(rho=1.0, eta_noise=0.1, lr=0.1)
theory_solver = ODESolver(equations=equations, order_params=["m", "q"])

# Run simulation
sim = OnlineSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=MSELoss(),
    theory_solver=theory_solver,
)

# Plot comparison
plotter = ComparisonPlotter()
plotter.plot_theory_vs_experiment(results)
```

---

## 9. Key Insights and Phenomena

### 9.1 Dimension-Free Dynamics

In the limit $d \to \infty$:
- Learning dynamics are described by ODEs with $O(k^2 + kk^*)$ variables
- Independent of the ambient dimension $d$
- Enables analysis of arbitrarily high-dimensional problems

### 9.2 Convergence Rate

The stochastic process concentrates to the ODE at rate $O(1/\sqrt{d})$:
- Finite-$d$ simulations converge to theory as $d$ increases
- Provides quantitative prediction for practical systems

### 9.3 Phase Transitions

**Critical learning rate:**
$$\tau_c = \frac{2}{1 + \lambda}$$

- $\tau < \tau_c$: Convergent regime
- $\tau > \tau_c$: Divergent regime

### 9.4 Noise Amplification

The $\tau^2 V$ term in the $dQ/dt$ equation captures **gradient noise**:
- Larger learning rate amplifies noise
- Trade-off between convergence speed and steady-state error

### 9.5 Specialization Dynamics

For multi-hidden-unit networks:
- **Symmetric phase:** All hidden units have similar weights
- **Symmetry breaking:** Units specialize to different features
- **Convergence phase:** Approach to optimal solution

---

## 10. Extensions

### 10.1 Batch SGD

For mini-batch size $B$:

$$\frac{dQ}{dt} = -\tau \mathbb{E}[\Gamma \cdot (\mathbf{a} \odot \sigma')\boldsymbol{\Upsilon}^\top + \cdots] + \frac{\tau^2}{B} \mathbb{E}[\Gamma^2 \cdots]$$

The noise term scales as $1/B$.

### 10.2 Learning Rate Schedules

For time-varying $\tau(t)$:

$$\frac{dQ}{dt} = -\tau(t) \mathbb{E}[\cdots] + \tau(t)^2 \mathbb{E}[\cdots]$$

Optimal schedules can be derived analytically.

### 10.3 Momentum SGD

Additional order parameters for momentum:

$$\mathbf{v}^{t+1} = \beta \mathbf{v}^t + \nabla_\theta r$$
$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \tau \mathbf{v}^{t+1}$$

---

## References

1. Saad, D. & Solla, S.A. (1995). "On-line learning in soft committee machines." *Phys. Rev. E*.
2. Biehl, M. & Schwarze, H. (1995). "Learning by on-line gradient descent." *J. Phys. A*.
3. Werfel, J., Xie, X., & Seung, H.S. (2005). "Learning curves for stochastic gradient descent." *Neural Computation*.
4. Goldt, S., Mezard, M., Krzakala, F., & Zdeborova, L. (2020). "Modeling the influence of data structure on learning in neural networks." *Phys. Rev. X*.
5. Engel, A. & Van den Broeck, C. (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
