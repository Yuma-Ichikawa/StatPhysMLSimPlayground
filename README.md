<p align="center">
  <img src="assets/logo.png" alt="StatPhys-ML Logo" width="400">
</p>

<h1 align="center">StatPhys-ML</h1>

<p align="center">
  <strong>Statistical Mechanics Simulation Package for Machine Learning</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/license-BSD--3--Clause-green.svg" alt="License"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  A powerful Python package for <strong>Teacher-Student model</strong> analysis using <strong>statistical mechanics</strong> methods.
</p>

## Features

- **Dataset Generation**: 22 customizable datasets including Gaussian, sparse, structured, GLM, ICL tasks, sequences/tokens, attention-indexed, fairness, and noisy labels
- **Learning Models**: 19 models including linear, committee machines, two-layer networks, deep linear, random features, softmax, transformers, and sequence models (LSA, SSM, RNN, Hopfield)
- **Loss Functions**: MSE, Ridge, LASSO, Huber, Hinge, Logistic, Probit, Softmax cross-entropy, and more
- **Theory Solvers**:
  - **Replica Method**: Saddle-point equation solver with damping and continuation
  - **Online Learning**: ODE solver for learning dynamics with adaptive stepping
  - **DMFT**: Coming soon
- **Simulation Framework**: Unified interface for experiments with automatic theory comparison
- **Visualization**: Publication-quality plots for theory vs experiment comparison
- **Utility Functions**: Special functions, Gaussian integrals (Gauss-Hermite/numerical quadrature), proximal operators

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/yuma-ichikawa/statphys-ml.git
cd statphys-ml
pip install -e ".[dev]"
```

### Using uv (faster)

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Example: Ridge Regression with Replica Theory

```python
import statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, SimulationConfig
from statphys.vis import ComparisonPlotter

# Fix random seed
statphys.fix_seed(42)

# Create dataset with linear teacher
dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)

# Configure simulation
config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0),  # Sample ratio range
    alpha_steps=20,
    n_seeds=5,
    reg_param=0.01,
)

# Run simulation
sim = ReplicaSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
)

# Visualize results
plotter = ComparisonPlotter()
plotter.plot_theory_vs_experiment(results)
```

### Example: Online SGD Dynamics

```python
from statphys.simulation import OnlineSimulation, SimulationConfig
from statphys.theory.online import ODESolver, GaussianLinearMseEquations

# Configure online simulation
config = SimulationConfig.for_online(
    t_max=10.0,  # Maximum time (t = n/d)
    t_steps=100,
    n_seeds=5,
)

# Create theory solver
# GaussianLinearMseEquations: ODE for online SGD with MSE loss
theory_solver = ODESolver(
    equations=GaussianLinearMseEquations(rho=1.0, lr=0.1),
    order_params=["m", "q"],
)

# Run simulation with theory comparison
sim = OnlineSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
    theory_solver=theory_solver,
)
```

## Supported Components

### Datasets (22 types)

| Category | Class | Description |
|----------|-------|-------------|
| **Gaussian** | `GaussianDataset` | Standard i.i.d. Gaussian input with linear teacher |
| | `GaussianClassificationDataset` | Sign teacher for binary classification |
| | `GaussianMultiOutputDataset` | Multi-output teacher (committee-style) |
| **Sparse** | `SparseDataset` | Sparse input distribution |
| | `BernoulliGaussianDataset` | Bernoulli-Gaussian mixture input |
| **Structured** | `StructuredDataset` | Arbitrary covariance matrix |
| | `CorrelatedGaussianDataset` | Exponentially correlated input |
| | `SpikedCovarianceDataset` | Spiked covariance model |
| **GLM Teachers** | `LogisticTeacherDataset` | Logistic teacher: $P(y=1 \mid u) = \sigma(u)$ |
| | `ProbitTeacherDataset` | Probit teacher: $P(y=1 \mid u) = \Phi(u)$ |
| **Gaussian Mixture** | `GaussianMixtureDataset` | Binary GMM (for DMFT analysis) |
| | `MulticlassGaussianMixtureDataset` | Multi-class GMM |
| **ICL Tasks** | `ICLLinearRegressionDataset` | ICL task with linear teacher (for LSA analysis) |
| | `ICLNonlinearRegressionDataset` | ICL task with nonlinear (2-layer) teacher |
| **Sequence/Token** | `MarkovChainDataset` | Markov chain sequences (for induction head) |
| | `CopyTaskDataset` | Copy/trigger task (induction head emergence) |
| | `GeneralizedPottsDataset` | Language-like Potts sequences (Phys. Rev. 2024) |
| | `TiedLowRankAttentionDataset` | Position-semantics phase transition (NeurIPS 2024) |
| | `MixedGaussianSequenceDataset` | Correlated token sequences with latent clusters |
| **Attention** | `AttentionIndexedModelDataset` | AIM for Bayes-optimal attention (arXiv 2025) |
| **Fairness** | `TeacherMixtureFairnessDataset` | Fairness/bias with group teachers (ICML 2024) |
| **Noisy Labels** | `NoisyGMMSelfDistillationDataset` | Label noise for self-distillation (2025) |

<p align="center">
  <img src="assets/dataset_diagram.png" alt="Dataset Generation Framework" width="800">
</p>
<p align="center"><em>Teacher-Student framework for data generation</em></p>

### Models (19 types)

| Category | Class | Description |
|----------|-------|-------------|
| **Linear** | `LinearRegression` | Linear regression with $1/\sqrt{d}$ scaling |
| | `LinearClassifier` | Linear classifier (sign/logit/prob output) |
| | `RidgeRegression` | Ridge regression wrapper |
| **Committee** | `CommitteeMachine` | Hard committee (sign activation) |
| | `SoftCommitteeMachine` | Soft committee (erf/tanh/relu) |
| **MLP** | `TwoLayerNetwork` | Two-layer network with various activations |
| | `TwoLayerNetworkReLU` | Two-layer ReLU network |
| | `DeepNetwork` | Multi-layer network |
| **Deep Linear** | `DeepLinearNetwork` | Deep linear network (identity activation) |
| **Random Features** | `RandomFeaturesModel` | Random features / kernel approximation |
| | `KernelRidgeModel` | Kernel ridge regression wrapper |
| **Softmax** | `SoftmaxRegression` | Multi-class softmax regression |
| | `SoftmaxRegressionWithBias` | Softmax with bias terms |
| **Transformer** | `SingleLayerAttention` | Single attention layer |
| | `SingleLayerTransformer` | Full single-layer transformer |
| **Sequence Models** | `LinearSelfAttention` | Linear self-attention (LSA) for ICL theory |
| | `StateSpaceModel` | State space model (SSM) for sequences |
| | `LinearRNN` | Linear recurrent neural network |
| **Energy-Based** | `ModernHopfieldNetwork` | Modern Hopfield network (attention ≈ energy min) |

<p align="center">
  <img src="assets/model_diagram.png" alt="Model Architectures" width="800">
</p>
<p align="center"><em>Supported model architectures with unified scaling convention</em></p>

### Loss Functions (16 types)

| Category | Class | Formula |
|----------|-------|---------|
| **Regression** | `MSELoss` | $\frac{1}{2}(y - \hat{y})^2$ |
| | `RidgeLoss` | $\text{MSE} + \lambda \|\mathbf{w}\|_2^2$ |
| | `LassoLoss` | $\text{MSE} + \lambda \|\mathbf{w}\|_1$ |
| | `ElasticNetLoss` | $\text{MSE} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$ |
| | `HuberLoss` | Smooth robust loss |
| | `PseudoHuberLoss` | Differentiable Huber |
| **Binary Classification** | `CrossEntropyLoss` | Binary cross-entropy |
| | `LogisticLoss` | $\log(1 + e^{-y \hat{y}})$ |
| | `HingeLoss` | $\max(0, 1 - y\hat{y})$ |
| | `SquaredHingeLoss` | $\max(0, 1 - y\hat{y})^2$ |
| | `PerceptronLoss` | $\max(0, -y\hat{y})$ |
| | `ExponentialLoss` | $e^{-y\hat{y}}$ |
| | `RampLoss` | Bounded hinge loss |
| | `ProbitLoss` | $-\log \Phi(y\hat{y})$ |
| **Multi-class** | `SoftmaxCrossEntropyLoss` | Softmax + cross-entropy |
| | `MultiMarginLoss` | Multi-class hinge (Crammer-Singer) |

### Theory Equations

#### Replica Method (6 scenarios)

| Full Class Name | Short Alias | Problem |
|-----------------|-------------|---------|
| `GaussianLinearRidgeEquations` | `RidgeRegressionEquations` | Ridge regression saddle-point equations |
| `GaussianLinearLassoEquations` | `LassoEquations` | LASSO with soft-thresholding |
| `GaussianLinearLogisticEquations` | `LogisticRegressionEquations` | Logistic regression |
| `GaussianLinearHingeEquations` | `PerceptronEquations` | Perceptron/SVM (Gardner volume) |
| `GaussianLinearProbitEquations` | `ProbitEquations` | Probit classification |
| `GaussianCommitteeMseEquations` | `CommitteeMachineEquations` | Committee machine (symmetric ansatz) |

#### Online Learning (6 scenarios)

| Full Class Name | Short Alias | Problem |
|-----------------|-------------|---------|
| `GaussianLinearMseEquations` | `OnlineSGDEquations` | Online SGD for linear regression |
| `GaussianLinearRidgeEquations` | `OnlineRidgeEquations` | Online ridge regression |
| `GaussianLinearPerceptronEquations` | `OnlinePerceptronEquations` | Online perceptron learning |
| `GaussianLinearLogisticEquations` | `OnlineLogisticEquations` | Online logistic regression |
| `GaussianLinearHingeEquations` | `OnlineHingeEquations` | Online SVM/hinge loss |
| `GaussianCommitteeMseEquations` | `OnlineCommitteeEquations` | Online committee machine (erf) |

### Utility Functions

#### Special Functions (`statphys.utils.special_functions`)

| Function | Description |
|----------|-------------|
| `gaussian_pdf`, `gaussian_cdf`, `gaussian_tail` | Gaussian distribution functions |
| `Phi`, `H`, `phi` | Standard notation aliases |
| `erf_activation`, `erf_derivative` | Error function activation |
| `sigmoid`, `sigmoid_derivative` | Sigmoid and derivative |
| `I2`, `I3`, `I4` | Committee machine correlation functions ($I_2, I_3, I_4$) |
| `soft_threshold`, `firm_threshold` | Proximal operators |
| `classification_error_linear`, `regression_error_linear` | Generalization error formulas |

#### Numerical Integration (`statphys.utils.integration`)

| Function | Description |
|----------|-------------|
| `gaussian_integral_1d` | Univariate Gaussian integral |
| `gaussian_integral_2d` | Bivariate Gaussian integral |
| `gaussian_integral_nd` | Multivariate Gaussian integral |
| `teacher_student_integral` | Joint $(u, z)$ integral for teacher-student |
| `conditional_expectation` | $\mathbb{E}[f(z) \mid u]$ or $\mathbb{E}[f(u) \mid z]$ |

**Integration Methods:**
- `hermite`: Gauss-Hermite quadrature (efficient, recommended)
- `quad`: Scipy adaptive quadrature (for difficult integrands)
- `monte_carlo`: Monte Carlo sampling (for high dimensions)

#### Order Parameter Utilities (`statphys.utils.order_params`)

| Class/Function | Description |
|----------------|-------------|
| `OrderParameterCalculator` | Comprehensive automatic order parameter calculator |
| `auto_calc_order_params` | Convenience function for quick calculations |
| `OrderParameters` | Dataclass container for all order parameters |
| `ModelType` | Enum for model type detection (LINEAR, COMMITTEE, TWO_LAYER, DEEP, TRANSFORMER) |
| `TaskType` | Enum for task type detection (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION) |

## Package Structure

```
src/statphys/
├── dataset/          # Data generation
│   ├── base.py       # BaseDataset abstract class
│   ├── gaussian.py   # Gaussian, Classification, MultiOutput
│   ├── sparse.py     # Sparse, BernoulliGaussian
│   ├── structured.py # Structured, Correlated, Spiked
│   ├── glm.py        # Logistic, Probit, GaussianMixture teachers
│   ├── icl.py        # ICL linear/nonlinear regression tasks
│   ├── sequence.py   # Markov, Copy, Potts, TiedAttention, MixedSequence
│   ├── attention.py  # AttentionIndexedModel (AIM)
│   ├── fairness.py   # TeacherMixtureFairness
│   └── noisy.py      # NoisyGMMSelfDistillation
├── model/            # Learning models
│   ├── base.py       # BaseModel abstract class
│   ├── linear.py     # LinearRegression, Classifier, Ridge
│   ├── committee.py  # CommitteeMachine, SoftCommittee
│   ├── mlp.py        # TwoLayerNetwork, DeepNetwork
│   ├── random_features.py # RandomFeatures, KernelRidge, DeepLinear
│   ├── softmax.py    # SoftmaxRegression
│   ├── transformer.py # Attention, Transformer
│   └── sequence.py   # LSA, SSM, RNN, Hopfield
├── loss/             # Loss functions
│   ├── base.py       # BaseLoss abstract class
│   ├── regression.py # MSE, Ridge, LASSO, ElasticNet, Huber
│   └── classification.py # Hinge, Logistic, Probit, Softmax, etc.
├── theory/           # Theoretical calculations
│   ├── replica/      # Replica method (6 scenarios)
│   │   ├── solver.py # SaddlePointSolver
│   │   └── scenario/ # Saddle-point equations by scenario
│   │       ├── base.py                    # ReplicaEquations base class
│   │       ├── gaussian_linear_ridge.py   # Ridge regression
│   │       ├── gaussian_linear_lasso.py   # LASSO regression
│   │       ├── gaussian_linear_logistic.py # Logistic regression
│   │       ├── gaussian_linear_hinge.py   # Perceptron/SVM
│   │       ├── gaussian_linear_probit.py  # Probit regression
│   │       └── gaussian_committee_mse.py  # Committee machine
│   ├── online/       # Online learning (6 scenarios)
│   │   ├── solver.py # ODESolver, AdaptiveODESolver
│   │   └── scenario/ # ODE equations by scenario
│   │       ├── base.py                    # OnlineEquations base class
│   │       ├── gaussian_linear_mse.py     # Online SGD (MSE)
│   │       ├── gaussian_linear_ridge.py   # Online ridge
│   │       ├── gaussian_linear_perceptron.py # Online perceptron
│   │       ├── gaussian_linear_logistic.py # Online logistic
│   │       ├── gaussian_linear_hinge.py   # Online SVM/hinge
│   │       └── gaussian_committee_mse.py  # Online committee (erf)
│   └── dmft/         # DMFT (coming soon)
├── simulation/       # Numerical experiments
│   ├── base.py       # BaseSimulation
│   ├── config.py     # SimulationConfig
│   ├── replica_sim.py # ReplicaSimulation
│   ├── online_sim.py  # OnlineSimulation
│   └── runner.py     # SimulationRunner
├── vis/              # Visualization
│   ├── comparison.py # ComparisonPlotter
│   ├── phase_diagram.py # PhaseDiagramPlotter
│   ├── order_params.py # OrderParamPlotter
│   └── default_plots.py # Publication-quality default plots
└── utils/            # Utilities
    ├── special_functions.py # Special functions (Gaussian, erf, etc.)
    ├── integration.py # Gaussian integrals (Hermite/quad/MC)
    ├── order_params.py # Order parameter calculation
    ├── math.py        # Basic math utilities
    ├── seed.py        # Random seed management
    └── io.py          # Results I/O
```

## Key Concepts

### Order Parameters

In the high-dimensional limit ($d \to \infty$), learning can be characterized by a few **order parameters**. The specific parameters depend on the **model type**:

#### Linear Models

For a linear student $\hat{y} = \mathbf{w}^\top \mathbf{x} / \sqrt{d}$ and linear teacher $y = \mathbf{w}_0^\top \mathbf{x} / \sqrt{d} + \epsilon$:

| Parameter | Definition | Meaning |
|-----------|------------|---------|
| $m$ | $\frac{1}{d} \mathbf{w}^\top \mathbf{w}_0$ | Student-Teacher overlap (generalization) |
| $q$ | $\frac{1}{d} \mathbf{w}^\top \mathbf{w}$ | Student self-overlap (weight norm) |
| $\rho$ | $\frac{1}{d} \mathbf{w}_0^\top \mathbf{w}_0$ | Teacher norm (dataset parameter) |

#### Committee Machines / Two-Layer Networks

For models with $K$ hidden units and weight matrix $\mathbf{W} \in \mathbb{R}^{K \times d}$:

| Parameter | Definition | Meaning |
|-----------|------------|---------|
| $M_{km}$ | $\frac{1}{d} \mathbf{W}_k^\top \mathbf{W}_{0,m}$ | Student unit $k$ - Teacher unit $m$ overlap |
| $Q_{kl}$ | $\frac{1}{d} \mathbf{W}_k^\top \mathbf{W}_l$ | Student self-overlap matrix |
| $R_{mn}$ | $\frac{1}{d} \mathbf{W}_{0,m}^\top \mathbf{W}_{0,n}$ | Teacher self-overlap matrix |
| $\mathbf{a}$ | Second-layer weights | $O(1)$ scalars (not normalized) |

### Generalization Error $E_g$

The generalization error formula depends on the **task type**:

**Regression** (MSE loss):

$$E_g = \frac{1}{2}\left(\rho - 2m + q\right) + \frac{\eta}{2}$$

where $\eta$ is the noise variance. For committee machines, $m$ and $q$ are replaced by appropriate averages over the overlap matrices.

**Binary Classification** (linear classifier):

$$E_g = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q \cdot \rho}}\right)$$

This gives the probability of misclassification based on the angle between student and teacher weight vectors.

### Thermodynamic Limits

#### Replica Method: Sample Ratio $\alpha = n/d$

For **batch learning** in the limit $n, d \to \infty$ with $\alpha = n/d$ fixed:
- $\alpha < 1$: Underdetermined (interpolation regime)
- $\alpha = 1$: Transition point
- $\alpha > 1$: Overdetermined

#### Online Learning: Normalized Time $t = \tau / d$

For **online SGD** in the limit $d \to \infty$, order parameters evolve as functions of normalized time $t$:

$$\frac{dm}{dt} = f_m(m, q; \eta, \lambda), \quad \frac{dq}{dt} = f_q(m, q; \eta, \lambda)$$

where $\eta$ is the learning rate and $\lambda$ is the regularization parameter.

### Theory Types

1. **Replica Method**: Saddle-point equations for equilibrium order parameters as a function of $\alpha$
2. **Online Learning**: ODE system for order parameter dynamics as a function of $t$
3. **DMFT** (coming soon): For structured data and non-i.i.d. settings

### Loss Function Scaling (Important)

Loss functions use different scaling conventions for Replica and Online simulations:

| Simulation | Loss Formula | Scaling |
|------------|--------------|---------|
| **Replica** | $\mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \lambda \|\mathbf{w}\|^2$ | $O(d)$ |
| **Online** | $\mathcal{L} = \frac{1}{d}\ell(y, \hat{y}) + \frac{\lambda}{d}\|\mathbf{w}\|^2$ | $O(1)$ |

**Why this matters:**
- **Replica** ($n = O(d)$): Data term sums over $n$ samples → $O(d)$. Regularization $\lambda\|\mathbf{w}\|^2 = \lambda d q \to O(d)$.
- **Online**: Single-sample loss scaled by $1/d$ ensures gradient components are $O(1/\sqrt{d})$, matching ODE theory.

**Usage:**
```python
from statphys.loss import RidgeLoss

loss_fn = RidgeLoss(reg_param=0.1)

# For Replica simulation (automatically used by ReplicaSimulation)
loss = loss_fn.for_replica(y_pred, y_true, model)  # O(d)

# For Online simulation (automatically used by OnlineSimulation)
loss = loss_fn.for_online(y_pred, y_true, model, d=d)  # O(1/d)
```

### Automatic Order Parameter Calculation

The package provides **automatic extraction** of all relevant order parameters for various model types:

```python
from statphys.utils.order_params import OrderParameterCalculator, auto_calc_order_params

# Method 1: Quick calculation with auto_calc_order_params
params = auto_calc_order_params(dataset, trained_model)
# Returns [m, q, eg] for linear models

# Method 2: Detailed calculation with OrderParameterCalculator
calculator = OrderParameterCalculator(
    return_format="object",    # "list", "dict", or "object"
    include_matrices=True,     # Include full overlap matrices
    include_teacher_overlaps=True,  # Compute R = W0^T @ W0 / d
    verbose=True,
)
params = calculator(dataset, trained_model)
print(params.summary())
```

**Key Features:**

| Feature | Description |
|---------|-------------|
| **Model Type Detection** | Automatically detects Linear, Committee, TwoLayer, Deep, Transformer models |
| **Task Type Detection** | Automatically identifies Regression, Binary/Multiclass Classification |
| **Student-Teacher Overlaps** | All $M_{ij} = \frac{1}{d} \mathbf{W}_i^\top \mathbf{W}_0^{(j)}$ overlaps |
| **Student Self-Overlaps** | All $Q_{ij} = \frac{1}{d} \mathbf{W}_i^\top \mathbf{W}_j$ (includes $Q$ matrix for committee machines) |
| **Teacher Self-Overlaps** | $R$ matrix from dataset parameters |
| **$O(1)$ Scalars** | Bias terms, second-layer weights ($\mathbf{a}$), and other non-normalized quantities |
| **Generalization Error** | Regression: $E_g = \frac{1}{2}(\rho - 2m + q) + \frac{\eta}{2}$, Classification: $E_g = \frac{1}{\pi}\arccos(m/\sqrt{q\rho})$ |

**Output Formats:**

```python
# List format (for simulation compatibility)
params = auto_calc_order_params(dataset, model, return_format="list")
# Linear: [m, q, eg]
# Committee: [m_avg, q_diag_avg, q_offdiag_avg, eg]
# TwoLayer: [m_avg, q_diag_avg, q_offdiag_avg, a_norm, eg]

# Dict format
params = auto_calc_order_params(dataset, model, return_format="dict")
# {'M_w_W0': 0.8, 'Q_w_w': 0.9, 'eg': 0.05, ...}

# Object format (full access)
params = auto_calc_order_params(dataset, model, return_format="object")
print(params.student_teacher_overlaps)  # All M values
print(params.student_self_overlaps)     # All Q values
print(params.generalization_error)      # E_g
```

**Use in Simulations:**

```python
# Method 1: Enable auto_order_params in SimulationConfig (Recommended)
# This automatically detects model type and prints computed parameters

config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0),
    alpha_steps=20,
    n_seeds=5,
    auto_order_params=True,  # Enable automatic order parameter calculation
)

# For online simulations:
config_online = SimulationConfig.for_online(
    t_max=10.0,
    t_steps=100,
    auto_order_params=True,  # Works for online simulations too
)

# When simulation starts, it will print:
# ============================================================
# 【AUTO ORDER PARAMETER CALCULATION ENABLED】
# ============================================================
#   Model Type: linear
#   Task Type:  regression
#
#   Order Parameters to be computed:
#     [0] m: Student-Teacher overlap (m = w^T w_0 / d)
#     [1] q: Student self-overlap (q = w^T w / d)
#     [2] eg: Generalization error (E_g)
# ============================================================

# Method 2: Pass custom calculator directly
from statphys.utils.order_params import OrderParameterCalculator

calculator = OrderParameterCalculator(return_format="list")

results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=loss_fn,
    calc_order_params=calculator,  # Custom calculator
)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=statphys

# Run specific test file
pytest tests/test_dataset.py
```

## Tutorial

### Simulation Only (No Theory Comparison)

Run numerical experiments without theoretical analysis:

```python
import statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, SimulationConfig

# Fix random seed for reproducibility
statphys.fix_seed(42)

# Create dataset
dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)

# Configure simulation (simulation only, no theory)
config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0),
    alpha_steps=20,
    n_seeds=5,
    reg_param=0.01,
    use_theory=False,  # Disable theory comparison
)

# Run simulation
sim = ReplicaSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
)

# Plot simulation results
import matplotlib.pyplot as plt
alpha_values = results.experiment_results["alpha_values"]
eg_mean = [op[2] for op in results.experiment_results["order_params_mean"]]  # E_g is index 2
eg_std = [op[2] for op in results.experiment_results["order_params_std"]]

plt.errorbar(alpha_values, eg_mean, yerr=eg_std, fmt='o', label='Simulation')
plt.xlabel(r'$\alpha = n/d$')
plt.ylabel(r'$E_g$')
plt.legend()
plt.show()
```

### Theory vs Simulation Verification (Recommended)

Compare numerical experiments with theoretical predictions:

```python
import statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, SimulationConfig
from statphys.theory.replica import SaddlePointSolver, GaussianLinearRidgeEquations
from statphys.vis import ComparisonPlotter

# Fix random seed
statphys.fix_seed(42)

# Create dataset
dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)

# Configure simulation with theory enabled
config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0),
    alpha_steps=20,
    n_seeds=5,
    reg_param=0.01,
    use_theory=True,  # Enable theory comparison
)

# Create theory solver
equations = GaussianLinearRidgeEquations(rho=1.0, eta=0.1, reg_param=0.01)
theory_solver = SaddlePointSolver(
    equations=equations,
    order_params=["m", "q"],
)

# Run simulation with theory comparison
sim = ReplicaSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
    theory_solver=theory_solver,
)

# Visualize theory vs simulation comparison
plotter = ComparisonPlotter()
plotter.plot_theory_vs_experiment(results)
```

For complete examples, see:
- [`examples/theory_vs_simulation_verification_ja.ipynb`](examples/theory_vs_simulation_verification_ja.ipynb) (日本語)
- [`examples/theory_vs_simulation_verification_en.ipynb`](examples/theory_vs_simulation_verification_en.ipynb) (English)

## Examples

See the `examples/` directory:

| File | Description |
|------|-------------|
| `basic_usage.ipynb` | Comprehensive tutorial covering all features |
| `dataset_gallery.ipynb` | Visualization of all 22 supported datasets |
| `model_gallery.ipynb` | Visualization of all 19 supported models |
| `replica_ridge_regression.py` | Ridge regression with replica theory |
| `online_sgd_learning.py` | Online SGD dynamics |
| `committee_machine.py` | Committee machine analysis |

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7

## Documentation

Theory notes are available in the `docs/` directory:

| File | Description |
|------|-------------|
| [`docs/replica_note.md`](docs/replica_note.md) | Replica method for static analysis (saddle-point equations, order parameters) |
| [`docs/online_sgd_learning_note.md`](docs/online_sgd_learning_note.md) | Online SGD dynamics (ODE derivation, concentration theorems) |

## License

BSD-3-Clause License - see [LICENSE.txt](LICENSE.txt) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ichikawa2025statphysml,
  author       = {Ichikawa, Yuma},
  title        = {{StatPhys-ML}: Statistical Mechanics Simulation Package for Machine Learning},
  year         = {2025},
  month        = feb,
  version      = {0.1.0},
  publisher    = {GitHub},
  url          = {https://github.com/yuma-ichikawa/statphys-ml},
  note         = {Python package for Teacher-Student model analysis using replica method and online learning theory}
}
```

For BibTeX users who prefer `@misc`:

```bibtex
@misc{ichikawa2025statphysml,
  author       = {Ichikawa, Yuma},
  title        = {{StatPhys-ML}: Statistical Mechanics Simulation Package for Machine Learning},
  year         = {2025},
  howpublished = {\url{https://github.com/yuma-ichikawa/statphys-ml}},
  note         = {Version 0.1.0}
}
```

## Author

**Yuma Ichikawa, Ph.D.**

- **Website**: [https://ichikawa-laboratory.com/](https://ichikawa-laboratory.com/)
- **Twitter**: [@yuma_1_or](https://x.com/yuma_1_or)
- **Google Scholar**: [Yuma Ichikawa](https://scholar.google.com/citations?user=yuma-ichikawa)
- **GitHub**: [yuma-ichikawa](https://github.com/yuma-ichikawa)

### Research Topics

- **Statistical Mechanics**: Information Statistical Mechanics, Spin Glass, Phase Transition, Markov Chain Monte Carlo
- **Learning Theory**: High-Dimensional Statistics, Learning Dynamics
- **Combinatorial Optimization**: Learning for Optimization, Heuristics, Simulated Annealing
- **Large Language Model (LLM)**: Architecture, Compression, Quantization, Pruning

### Contact

- **Email**: yuma.ichikawa@a.riken.jp

## Disclaimer

This project is an **independent personal project** developed by Yuma Ichikawa.
It is **not affiliated with, sponsored by, or endorsed by any organization**, including the author's employer.
All views and opinions expressed in this project are solely those of the author.

