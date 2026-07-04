# Getting Started

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

Core requirements: Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7, pandas ≥ 2.0.

## One-liner API

The fastest way to run a complete experiment (simulation + theory + plots):

```python
import statphys

# Online SGD vs exact ODE theory (linear regression)
result = statphys.quick_online(d=400, lr=0.5, t_max=10)

# Ridge regression at several alpha = n/d vs replica theory
result = statphys.quick_replica(d=200, reg_param=0.1)

# Theory-free teacher-student experiment (works for any architecture)
result = statphys.quick_experiment("random_mlp", alphas=[1, 2, 4, 8])
```

Every helper returns the underlying result object, so the quick API also serves as an entry point into the full framework below.

## Full workflow: Replica simulation vs theory

```python
import statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, SimulationConfig
from statphys.theory.replica import SaddlePointSolver, GaussianLinearRidgeEquations
from statphys.vis import ComparisonPlotter

statphys.fix_seed(42)

# Dataset with linear teacher: y = w0.x/sqrt(d) + noise
dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)

# Simulation configuration
config = SimulationConfig.for_replica(
    alpha_range=(0.1, 5.0),   # sample ratio alpha = n/d
    alpha_steps=20,
    n_seeds=5,
    reg_param=0.01,
    use_theory=True,
)

# Theory: RS saddle-point equations for ridge regression
theory_solver = SaddlePointSolver(
    equations=GaussianLinearRidgeEquations(rho=1.0, eta=0.1, reg_param=0.01),
    order_params=["m", "q"],
)

sim = ReplicaSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
    theory_solver=theory_solver,
)

ComparisonPlotter().plot_theory_vs_experiment(results)
```

To run *without* theory, set `use_theory=False` and omit `theory_solver`; the experiment results remain accessible via `results.experiment_results`.

## Full workflow: Online SGD dynamics vs ODE theory

```python
from statphys.simulation import OnlineSimulation, SimulationConfig
from statphys.theory.online import ODESolver, GaussianLinearMseEquations

config = SimulationConfig.for_online(
    t_max=10.0,    # normalized time t = #samples / d
    t_steps=100,
    n_seeds=5,
    lr=0.5,        # equals the ODE learning rate eta (see concepts.md)
)

theory_solver = ODESolver(
    equations=GaussianLinearMseEquations(rho=1.0, lr=0.5),
    order_params=["m", "q"],
)

sim = OnlineSimulation(config)
results = sim.run(
    dataset=dataset,
    model_class=LinearRegression,
    loss_fn=RidgeLoss(0.01),
    theory_solver=theory_solver,
)
```

The theory ODE automatically starts from the experiment's measured initial condition; pass `theory_init_values=...` to `run()` to override.

## Running tests

```bash
pytest tests/              # all tests
pytest tests/ --cov=statphys
pytest tests/test_theory.py
```

## Next steps

- Component catalog (datasets/models/losses/theory): [components.md](components.md)
- Experiments for arbitrary architectures (incl. tiny GPT): [experiments.md](experiments.md)
- Visualization and animations: [visualization.md](visualization.md)
- Cluster execution: [slurm.md](slurm.md)
- The math behind the order parameters: [concepts.md](concepts.md)
