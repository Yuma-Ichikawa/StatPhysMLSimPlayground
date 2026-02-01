# Replica Method: Static Asymptotic Analysis

This note provides a comprehensive overview of the **replica method** for analyzing machine learning in the high-dimensional limit.

---

## 1. Introduction

Statistical mechanics, pioneered by Boltzmann, Maxwell, and Gibbs, aims to deduce macroscopic thermodynamic properties from microscopic behavior. In machine learning, the replica method enables **typical-case analysis** of learning algorithms in the proportional limit:

$$
n \to \infty, \quad d \to \infty, \quad \alpha = \frac{n}{d} = \Theta(1)
$$

Unlike worst-case PAC bounds, this approach provides **sharp characterization** of generalization error for specific data distributions.

---

## 2. General Framework

### 2.1 Empirical Risk Minimization

Given a training dataset $\mathcal{D} = \{(\mathbf{x}^\mu, y^\mu)\}_{\mu=1}^{n}$ drawn from distribution $p(\mathbf{x}, y)$, learning is performed via:

$$
\hat{\theta} \in \underset{\theta}{\mathrm{argmin}} \left\{ \mathcal{R}(\theta; \mathcal{D}) \right\}
$$

$$
\mathcal{R}(\theta; \mathcal{D}) = \sum_{\mu=1}^{n} \ell(y^\mu, f_\theta(\mathbf{x}^\mu)) + \lambda g(\theta)
$$

where $\ell(\cdot, \cdot)$ is the loss function and $g(\cdot)$ is the regularization term.

### 2.2 Boltzmann Distribution

To analyze the statistical properties of learned parameters, we define the **Boltzmann distribution**:

$$
p(\theta | \mathcal{D}; \beta) = \frac{1}{Z(\beta)} e^{-\beta \mathcal{R}(\theta; \mathcal{D})}
$$

$$
Z(\beta) = \int d\theta \, e^{-\beta \mathcal{R}(\theta; \mathcal{D})}
$$

**Key observations:**
- When $\beta = 1$: Boltzmann distribution equals the **Bayesian posterior** $p(\theta|\mathcal{D})$
- When $\beta \to \infty$: Converges to a **uniform distribution over ERM solutions**:

$$
\lim_{\beta \to \infty} p(\theta|\mathcal{D}; \beta) = \frac{1}{A} \sum_{a=1}^{A} \delta(\theta - \hat{\theta}_a)
$$

### 2.3 Free Energy Density

The **free energy density** serves as a cumulant-generating function:

$$
f(\mathcal{D}) = -\lim_{\beta \to \infty} \frac{1}{\beta d} \log Z(\beta)
$$

This allows evaluation of statistical properties of post-training parameters:

$$
\mathbb{E}_{(\mathbf{x}, y)}[\Phi(\hat{\theta}(\mathcal{D}))] = \lim_{\beta \to \infty} \int d\theta \, \Phi(\theta) \, p(\theta | \mathcal{D}; \beta)
$$

### 2.4 Self-Averaging Property

In the proportional limit, the free energy exhibits **self-averaging**:

$$
\forall \epsilon > 0, \quad \mathbb{P}\left[|f(\mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(\mathcal{D})]| > \epsilon\right] \xrightarrow{d \to \infty} 0
$$

This allows replacing dataset-specific analysis with expected free energy $\mathbb{E}_{\mathcal{D}}[f(\mathcal{D})]$.

### 2.5 The Replica Trick

Using the identity $\log Z = \lim_{r \to 0} \partial_r Z^r$:

$$
\mathbb{E}_{\mathcal{D}}[f(\mathcal{D})] = -\lim_{r \to 0} \frac{1}{r} \lim_{\beta \to \infty} \lim_{d \to \infty} \frac{\partial}{\partial r} \mathbb{E}_{\mathcal{D}}[Z^r(\mathcal{D})]
$$

**Procedure:**
1. Treat $r$ as an integer to evaluate $\mathbb{E}_{\mathcal{D}}[Z^r(\mathcal{D})]$
2. Analytically continue to $r \to 0$
3. Exchange limits (replica assumption)

---

## 3. Data Generation: Low-Dimensional Manifold Model

### 3.1 Definition

The **Low-Dimensional Manifold Model (LMM)** generates data through:

$$
\mathbf{x} \sim p_{\theta^*}(\mathbf{x}|\mathbf{c}), \quad y \sim p_{\phi^*}(y|\mathbf{c})
$$

where $\mathbf{c} \in \mathbb{R}^{k^*}$ is a latent variable with $k^* = \Theta(1)$.

### 3.2 Linear LMM (LLMM)

The simplest case uses a linear transformation:

$$
\mathbf{x} = \frac{1}{\sqrt{d}} W^* (\sqrt{\boldsymbol{\rho}} \odot \mathbf{c}) + \sqrt{\eta} \mathbf{n}
$$

$$
y \sim p_{\phi^*}(y|\mathbf{c})
$$

where:
- $W^* = (\mathbf{w}_m^*)_{m=1}^{k^*} \in \mathbb{R}^{d \times k^*}$: feature matrix
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0}_d, I_d)$: noise vector
- $\boldsymbol{\rho} \in \mathbb{R}^{k^*}$: signal strength
- $\eta$: noise intensity

---

## 4. Case Study: Narrow Two-Layer Neural Networks

### 4.1 Model Definition

A **narrow two-layer NN** with $k = \Theta(1)$ hidden units:

$$
\hat{y}(\mathbf{x}; \mathbf{a}, W, \mathbf{b}) = \mathbf{a}^\top \sigma\left(\frac{1}{\sqrt{d}} W^\top \mathbf{x} + \mathbf{b}\right)
$$

where:
- $\mathbf{a} \in \mathbb{R}^k$: second-layer weights
- $W \in \mathbb{R}^{d \times k}$: first-layer weights (columns $\mathbf{w}_l$)
- $\mathbf{b} \in \mathbb{R}^k$: bias terms
- $\sigma: \mathbb{R}^k \to \mathbb{R}^k$: element-wise activation

### 4.2 Training Objective

$$
\mathcal{R}(\mathbf{a}, W, \mathbf{b}; \mathcal{D}) = \sum_{\mu=1}^{n} \ell(y^\mu, \hat{y}(\mathbf{x}^\mu; \mathbf{a}, W, \mathbf{b})) + \lambda g(W)
$$

### 4.3 Replicated Partition Function

The replicated partition function is:

$$
\mathbb{E}_{\mathcal{D}}[Z^r(\mathcal{D})] = \mathbb{E}_{\mathcal{D}} \left[\prod_{a=1}^{r} \int d\boldsymbol{\theta}^a \, e^{-\beta \lambda \sum_a g(W^a) - \beta \sum_{a,\mu} \ell(y^\mu, \hat{y}(\mathbf{x}^\mu; \boldsymbol{\theta}^a))}\right]
$$

Using i.i.d. samples:

$$
= \prod_a \int d\boldsymbol{\theta}^a \, e^{-\beta \lambda \sum_a g(W^a)} \left(\mathbb{E}_{\mathbf{c}, y, \mathbf{x}}\left[e^{-\beta \sum_a \ell(y, \hat{y}(\mathbf{x}; \boldsymbol{\theta}^a))}\right]\right)^n
$$

### 4.4 Local Fields and Order Parameters

The model output depends on Gaussian variables only through **local fields**:

$$
u_l^a = \frac{1}{\sqrt{d}} (\mathbf{w}_l^a)^\top \mathbf{n} \in \mathbb{R}, \quad \forall a \in [r], l \in [k]
$$

These are jointly Gaussian with covariance defined by **order parameters**:

**Student self-overlap:**
$$
q_{ls}^{ab} = \mathbb{E}_{\mathbf{n}}[u_l^a u_s^b] = \frac{1}{d} (\mathbf{w}_l^a)^\top \mathbf{w}_s^b
$$

**Student-teacher overlap:**
$$
m_{lm}^a = \frac{1}{d} (\mathbf{w}_l^a)^\top \mathbf{w}_m^*
$$

### 4.5 Energy-Entropy Decomposition

The replicated partition function decomposes as:

$$
\mathbb{E}_{\mathcal{D}}[Z^r(\mathcal{D})] = \int e^{r\beta d \left(\alpha \mathcal{E}(\mathbf{A}, \mathbf{B}, q, \mathbf{m}) + \mathcal{S}(q, \mathbf{m}, \tilde{q}, \tilde{\mathbf{m}})\right)} d\mathbf{A} \, d\mathbf{B} \, dq \, d\mathbf{m} \, d\tilde{q} \, d\tilde{\mathbf{m}}
$$

where:
- **Energy term** $\mathcal{E}$: average loss for given order parameters
- **Entropy term** $\mathcal{S}$: volume of weight space for given order parameters

### 4.6 Saddle-Point Approximation

Since exponents scale as $d \to \infty$, the integral is evaluated via saddle-point:

$$
\lim_{d \to \infty} \log \mathbb{E}_{\mathcal{D}}[Z^r(\mathcal{D})] = \underset{q, \tilde{q}, \mathbf{m}, \tilde{\mathbf{m}}, \mathbf{A}, \mathbf{B}}{\mathrm{extr}} \left[\alpha \mathcal{E} + \mathcal{S}\right]
$$

---

## 5. Replica Symmetric (RS) Ansatz

### 5.1 Definition

The RS ansatz assumes all replicas have identical statistics:

$$
Q^{ab} = \left(q + \delta_{ab}\right) \frac{\chi}{\beta}, \quad \tilde{Q}^{ab} = \beta \hat{q} \delta_{ab} - \beta^2 \hat{\chi}
$$

$$
m^a = m, \quad \tilde{m}^a = \beta \hat{m}, \quad \mathbf{a}^a = \mathbf{a}, \quad \mathbf{b}^a = \mathbf{b} \quad \forall a
$$

This means two independent samples from the Boltzmann distribution have identical overlap with the teacher.

### 5.2 Decomposable Regularization

We assume the regularization is decomposable:

$$
g(W) = \sum_{l=1}^{k} g(\mathbf{w}_l)
$$

---

## 6. RS Entropy Term

Under the RS ansatz, the entropy term becomes:

$$
\mathcal{S}(q, m, \chi, \hat{q}, \hat{m}, \hat{\chi}) = \mathrm{tr}\left(\frac{1}{2}(\hat{q}q - \hat{\chi}\chi) - \hat{m}m\right) + \hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda)
$$

where:

$$
\hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda) = \frac{1}{\beta} \mathbb{E}_{\boldsymbol{\xi}} \left[\log \int_{\mathbb{R}^k} d\hat{\mathbf{w}} \, e^{\beta \left(-\frac{1}{2} \hat{\mathbf{w}}^\top \hat{q} \hat{\mathbf{w}} + (\hat{\chi}^{1/2} \boldsymbol{\xi} + \hat{m} \hat{\mathbf{w}}^*)^\top \hat{\mathbf{w}} - \lambda g(\hat{\mathbf{w}})\right)}\right]
$$

### 6.1 Moreau Envelope Form ($\beta \to \infty$)

As $\beta \to \infty$, the integral becomes a **Moreau envelope**:

$$
\hat{V} = -\mathbb{E}_{\boldsymbol{\xi}} \min_{\hat{\mathbf{w}} \in \mathbb{R}^k} \left[-\frac{1}{2} \hat{\mathbf{w}}^\top \hat{q} \hat{\mathbf{w}} + (\hat{\chi}^{1/2} \boldsymbol{\xi} + \hat{m} \hat{\mathbf{w}}^*)^\top \hat{\mathbf{w}} - \lambda g(\hat{\mathbf{w}})\right]
$$

**Minimization condition:**
$$
\hat{\mathbf{w}} = \hat{q}^{-1} \left(\hat{\chi}^{1/2} \boldsymbol{\xi} + \hat{m} \mathbf{w}^* - \lambda \nabla_{\hat{\mathbf{w}}} g(\hat{\mathbf{w}})\right)
$$

### 6.2 Ridge Regularization

For ridge regularization $g(\mathbf{w}) = \frac{1}{2}\sum_l \|\mathbf{w}_l\|^2$:

$$
\hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda) = \frac{1}{2} \left(\mathrm{Tr}\left[(\hat{q} + \lambda I_k)^{-1} \hat{\chi}\right] + (\hat{m} \hat{\mathbf{w}}^*)^\top (\hat{q} + \lambda I_k)^{-1} \hat{m} \hat{\mathbf{w}}^*\right)
$$

---

## 7. RS Energy Term

Under the RS ansatz:

$$
\mathcal{E}_{\mathrm{RS}}(q, \chi, m, \mathbf{a}, \mathbf{b}) = \alpha V(q, \chi, m, \mathbf{a}, \mathbf{b})
$$

where:

$$
V = \frac{1}{\beta} \mathbb{E}_{\mathbf{z}, \mathbf{c}, \zeta} \left[\log \int_{\mathbb{R}^k} d\mathbf{x} \, e^{-\beta \left[\frac{1}{2}\|\mathbf{x}\|^2 + \ell\left(y, \mathbf{a}^\top \sigma(\mathbf{p}(q, \chi, m, \mathbf{b}))\right)\right]}\right]
$$

with pre-activation:

$$
\mathbf{p}(q, \chi, m, \mathbf{b}) = \sqrt{\rho} m \mathbf{c} + \sqrt{\eta} \left(\chi^{1/2} \mathbf{x} + q^{1/2} \mathbf{z}\right) + \mathbf{b}
$$

### 7.1 Moreau Envelope Form ($\beta \to \infty$)

$$
V = -\mathbb{E}_{\mathbf{z}, \mathbf{c}} \left[\min_{\mathbf{x} \in \mathbb{R}^k} \left[\frac{1}{2}\|\mathbf{x}\|^2 + \ell\left(y, \mathbf{a}^\top \sigma(\mathbf{p})\right)\right]\right]
$$

**Minimization condition:**
$$
\mathbf{x} = -\sqrt{\eta} \chi^{1/2} \left\{\mathbf{a} \odot \sigma'(\mathbf{p})\right\} \frac{\partial \ell(y, \hat{y})}{\partial \hat{y}}
$$

---

## 8. RS Free Energy and Self-Consistent Equations

### 8.1 RS Free Energy Density

$$
-f = \underset{q, \chi, m, \mathbf{a}, \mathbf{b}, \hat{q}, \hat{\chi}, \hat{m}}{\mathrm{extr}} \left[\mathrm{tr}\left(\frac{1}{2}(\hat{q}q - \hat{\chi}\chi) - \hat{m}m\right) + \hat{V}(\hat{q}, \hat{\chi}, \hat{m}; \lambda) + \alpha V(q, \chi, m, \mathbf{a}, \mathbf{b})\right]
$$

### 8.2 Self-Consistent Equations

From the saddle-point conditions:

$$
\begin{cases}
q = -2 \frac{\partial}{\partial \hat{q}} \hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda) \\[8pt]
\chi = 2 \frac{\partial}{\partial \hat{\chi}} \hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda) \\[8pt]
m = \frac{\partial}{\partial \hat{m}} \hat{V}(\hat{q}, \hat{\chi}, \hat{m}, \lambda) \\[8pt]
\hat{q} = -2\alpha \frac{\partial}{\partial q} V(q, \chi, m, \mathbf{a}, \mathbf{b}) \\[8pt]
\hat{\chi} = 2\alpha \frac{\partial}{\partial \chi} V(q, \chi, m, \mathbf{a}, \mathbf{b}) \\[8pt]
\hat{m} = \alpha \frac{\partial}{\partial m} V(q, \chi, m, \mathbf{a}, \mathbf{b}) \\[8pt]
\mathbf{0}_k = \frac{\partial}{\partial \mathbf{b}} V(q, \chi, m, \mathbf{a}, \mathbf{b}) \\[8pt]
\mathbf{0}_k = \frac{\partial}{\partial \mathbf{a}} V(q, \chi, m, \mathbf{a}, \mathbf{b})
\end{cases}
$$

**Numerical Solution:** Iterate $\phi^{t+1} = (1-\gamma)\phi^t + \gamma f(\phi^t)$ with damping $\gamma \in [0,1]$.

---

## 9. Generalization Error

The generalization error is computed from the order parameters:

$$
\varepsilon_g = \mathbb{E}_{\mathbf{c}, \mathbf{z}} \left[\ell\left(y(\mathbf{c}), \mathbf{a}^\top \sigma\left(\sqrt{\rho} m \mathbf{c} + q^{1/2} \mathbf{z} + \mathbf{b}\right)\right)\right]
$$

where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}_k, I_k)$ and order parameters $(m, q, \mathbf{a}, \mathbf{b})$ are determined by the self-consistent equations.

### 9.1 Special Cases

**Linear Regression (MSE loss):**
$$
E_g = \frac{1}{2}(\rho - 2m + q)
$$

**Binary Classification:**
$$
E_g = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q \cdot \rho}}\right)
$$

---

## 10. Implementation in StatPhys-ML

### 10.1 Ridge Regression Example

```python
from statphys.theory.replica import SaddlePointSolver, GaussianLinearRidgeEquations

# Define saddle-point equations for ridge regression
equations = GaussianLinearRidgeEquations(
    rho=1.0,        # Teacher norm ||w*||²/d
    eta=0.1,        # Noise variance σ²
    reg_param=0.01, # Ridge parameter λ
)

# Create solver with damping
solver = SaddlePointSolver(
    equations=equations,
    order_params=["m", "q"],
    damping=0.5,
    adaptive_damping=True,
    tol=1e-8,
    max_iter=10000,
)

# Solve for range of α values
alpha_values = [0.5, 1.0, 2.0, 3.0, 5.0]
result = solver.solve(alpha_values=alpha_values, use_continuation=True)

# Access results
for i, alpha in enumerate(alpha_values):
    m, q = result.order_params["m"][i], result.order_params["q"][i]
    eg = equations.generalization_error(m, q)
    print(f"α={alpha}: m={m:.4f}, q={q:.4f}, E_g={eg:.4f}")
```

### 10.2 Custom Equations

```python
from statphys.theory.replica.scenario.base import ReplicaEquations

class CustomEquations(ReplicaEquations):
    def __init__(self, rho=1.0, eta=0.0, reg_param=0.01):
        super().__init__(rho=rho, eta=eta, reg_param=reg_param)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
    
    def __call__(self, m, q, alpha, **kwargs):
        # Implement your saddle-point update equations
        new_m = ...  # Fixed-point update for m
        new_q = ...  # Fixed-point update for q
        return new_m, new_q
    
    def generalization_error(self, m, q, **kwargs):
        return 0.5 * (self.rho - 2*m + q)
```

---

## 11. Mathematical Rigor

The replica method involves non-rigorous steps:
1. Exchange of limits ($d \to \infty$, $\beta \to \infty$, $r \to 0$)
2. Analytic continuation from integers to real $r$

**Rigorous alternatives:**
- **Convex problems:** Gaussian min-max theorem (CGMT) provides rigorous results consistent with replica predictions
- **Random matrix theory:** Certain results can be proven rigorously

---

## References

1. Gardner, E. (1988). "The space of interactions in neural network models." *J. Phys. A*.
2. Engel, A. & Van den Broeck, C. (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
3. Mézard, M., Parisi, G., & Virasoro, M.A. (1987). *Spin Glass Theory and Beyond*. World Scientific.
4. Thrampoulidis, C., Oymak, S., & Hassibi, B. (2018). "Precise error analysis of regularized M-estimators." *IEEE Trans. Inf. Theory*.
