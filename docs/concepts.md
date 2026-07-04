# Key Concepts

Background needed to read the theory modules and interpret simulation output. For derivations see [replica_note.md](replica_note.md) and [online_sgd_learning_note.md](online_sgd_learning_note.md); for literature pointers see [THEORY.md](THEORY.md).

## Order Parameters

In the high-dimensional limit ($d \to \infty$), learning is characterized by a few **order parameters**. The specific parameters depend on the model type.

### Linear models

For a linear student $\hat{y} = \mathbf{w}^\top \mathbf{x} / \sqrt{d}$ and linear teacher $y = \mathbf{w}_0^\top \mathbf{x} / \sqrt{d} + \epsilon$:

| Parameter | Definition | Meaning |
|-----------|------------|---------|
| $m$ | $\frac{1}{d} \mathbf{w}^\top \mathbf{w}_0$ | Student-Teacher overlap (generalization) |
| $q$ | $\frac{1}{d} \mathbf{w}^\top \mathbf{w}$ | Student self-overlap (weight norm) |
| $\rho$ | $\frac{1}{d} \mathbf{w}_0^\top \mathbf{w}_0$ | Teacher norm (dataset parameter) |

### Committee machines / two-layer networks

For models with $K$ hidden units and weight matrix $\mathbf{W} \in \mathbb{R}^{K \times d}$:

| Parameter | Definition | Meaning |
|-----------|------------|---------|
| $M_{km}$ | $\frac{1}{d} \mathbf{W}_k^\top \mathbf{W}_{0,m}$ | Student unit $k$ - Teacher unit $m$ overlap |
| $Q_{kl}$ | $\frac{1}{d} \mathbf{W}_k^\top \mathbf{W}_l$ | Student self-overlap matrix |
| $R_{mn}$ | $\frac{1}{d} \mathbf{W}_{0,m}^\top \mathbf{W}_{0,n}$ | Teacher self-overlap matrix |
| $\mathbf{a}$ | Second-layer weights | $O(1)$ scalars (not normalized) |

## Generalization Error $E_g$

**Regression** (MSE loss):

$$E_g = \frac{1}{2}\left(\rho - 2m + q\right) + \frac{\eta}{2}$$

where $\eta$ is the noise variance. For committee machines, $m$ and $q$ are replaced by appropriate averages over the overlap matrices.

**Binary classification** (linear classifier):

$$E_g = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q \cdot \rho}}\right)$$

the misclassification probability given the angle between student and teacher weight vectors. Both formulas are implemented in `statphys.utils.special_functions` as `regression_error_linear` and `classification_error_linear`.

## Thermodynamic Limits

### Replica method: sample ratio $\alpha = n/d$

For **batch learning** in the limit $n, d \to \infty$ with $\alpha = n/d$ fixed:

- $\alpha < 1$: underdetermined (interpolation regime)
- $\alpha = 1$: transition point
- $\alpha > 1$: overdetermined

### Online learning: normalized time $t = \tau / d$

For **online SGD** in the limit $d \to \infty$, order parameters evolve as functions of normalized time $t$:

$$\frac{dm}{dt} = f_m(m, q; \eta, \lambda), \quad \frac{dq}{dt} = f_q(m, q; \eta, \lambda)$$

where $\eta$ is the learning rate and $\lambda$ the regularization parameter. With the package's $1/\sqrt{d}$ input scaling, the optimizer's `lr` maps **directly** onto the ODE's $\eta$ (no extra $d$ factor).

### Theory types

1. **Replica method**: saddle-point equations for equilibrium order parameters vs $\alpha$
2. **Online learning**: ODE system for order-parameter dynamics vs $t$
3. **DMFT** (coming soon): structured data and non-i.i.d. settings

## Loss Function Scaling

Loss functions use different scaling conventions for replica and online simulations:

| Simulation | Loss Formula | Scaling |
|------------|--------------|---------|
| **Replica** | $\mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \lambda \|\mathbf{w}\|^2$ | $O(d)$ |
| **Online** | $\mathcal{L} = \frac{1}{d}\ell(y, \hat{y}) + \frac{\lambda}{d}\|\mathbf{w}\|^2$ | $O(1)$ |

**Why this matters:**

- **Replica** ($n = O(d)$): the data term sums over $n$ samples → $O(d)$, and $\lambda\|\mathbf{w}\|^2 = \lambda d q$ is also $O(d)$, so both terms compete at the same order.
- **Online**: the single-sample loss scaled by $1/d$ makes gradient components $O(1/\sqrt{d})$, matching the ODE derivation.

The simulations select the right convention automatically:

```python
from statphys.loss import RidgeLoss

loss_fn = RidgeLoss(reg_param=0.1)
loss = loss_fn.for_replica(y_pred, y_true, model)       # used by ReplicaSimulation
loss = loss_fn.for_online(y_pred, y_true, model, d=d)   # used by OnlineSimulation
```

## Automatic Order Parameter Calculation

`statphys.utils.order_params` extracts all relevant order parameters for a trained model, detecting the model type (linear / committee / two-layer / deep / transformer) and task type (regression / binary / multiclass) automatically.

```python
from statphys.utils.order_params import OrderParameterCalculator, auto_calc_order_params

# Quick: returns [m, q, eg] for linear models
params = auto_calc_order_params(dataset, trained_model)

# Detailed
calculator = OrderParameterCalculator(
    return_format="object",         # "list", "dict", or "object"
    include_matrices=True,          # full overlap matrices
    include_teacher_overlaps=True,  # R = W0^T W0 / d
    verbose=True,
)
params = calculator(dataset, trained_model)
print(params.summary())
```

### Output formats

```python
# List (simulation-compatible)
auto_calc_order_params(dataset, model, return_format="list")
# Linear: [m, q, eg]
# Committee: [m_avg, q_diag_avg, q_offdiag_avg, eg]
# TwoLayer: [m_avg, q_diag_avg, q_offdiag_avg, a_norm, eg]

# Dict
auto_calc_order_params(dataset, model, return_format="dict")
# {'M_w_W0': 0.8, 'Q_w_w': 0.9, 'eg': 0.05, ...}

# Object (full access)
params = auto_calc_order_params(dataset, model, return_format="object")
params.student_teacher_overlaps   # all M values
params.student_self_overlaps      # all Q values
params.generalization_error       # E_g
```

### Use in simulations

```python
# Recommended: let the simulation pick the calculator automatically
config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0), alpha_steps=20, n_seeds=5,
    auto_order_params=True,
)
# Works for online too:
config = SimulationConfig.for_online(t_max=10.0, t_steps=100, auto_order_params=True)

# Or pass a custom calculator
calculator = OrderParameterCalculator(return_format="list")
results = sim.run(dataset=dataset, model_class=LinearRegression,
                  loss_fn=loss_fn, calc_order_params=calculator)
```
