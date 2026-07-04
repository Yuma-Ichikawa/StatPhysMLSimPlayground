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
  A Python package for <strong>Teacher-Student model</strong> analysis using <strong>statistical mechanics</strong> methods — replica theory, online-learning ODEs, and theory-free experiments for arbitrary architectures.
</p>

<p align="center">
  <img src="assets/anim_learning_curve.gif" alt="Online SGD vs ODE theory" width="45%">
  <img src="assets/anim_phase_plane.gif" alt="(m, q) phase plane dynamics" width="42%">
</p>
<p align="center">
  <img src="assets/anim_specialization.gif" alt="Committee machine specialization" width="40%">
</p>
<p align="center"><em>
  Left: online SGD order parameters (solid) converging onto the exact ODE theory (dashed).
  Right: the same dynamics moving through the (m, q) phase plane over the theoretical flow field.
  Bottom: a soft committee machine <strong>specializing</strong> — the student-teacher overlap matrix developing a diagonal during training.
  Reproduce with <code>python scripts/generate_readme_assets.py</code>.
</em></p>

## Features

- **Theory solvers**: replica saddle-point equations (6 scenarios) and online-learning ODEs (6 scenarios), with automatic theory-vs-experiment comparison
- **22 datasets / 19 models / 16 losses**: from Gaussian linear teachers to ICL tasks, sequence models, and attention-indexed models
- **General teacher-student experiments**: theory-free numerical experiments for *any* PyTorch model, with structured teacher weights (sparse, low-rank, spiked, power-law, ...) and configurable input distributions — including **hidden-manifold** inputs for realistic data structure
- **Physics order parameters for any architecture**: function-space magnetization, replica overlap, susceptibility, Binder cumulant, specialization index — locate phase transitions numerically even where no theory exists
- **Numerical phase diagrams**: 2D (parameter × α) sweeps with contour-based boundary estimation, plus finite-size-scaling protocols
- **Architecture zoo**: matched teacher-student pairs for linear / MLP / deep MLP / CNN / LSTM / attention / tiny-GPT
- **Visualization**: publication-quality plots, phase portraits, overlap-matrix heatmaps, order-parameter dashboards, and GIF/MP4 animations
- **Modern phenomenology, ready-made**: grokking (delayed generalization), Gaussian universality, model-wise double descent, and data-scaling exponents as one-command studies
- **Slurm integration**: programmatic sbatch generation and job arrays, no hardcoded cluster paths
- **One-liner API**: `quick_online()`, `quick_replica()`, `quick_experiment()`, `quick_order_parameters()`, `quick_phase_diagram()`
- **CLI**: `statphys list / order-params / phase-diagram / study` — no Python required

## Installation

```bash
git clone https://github.com/yuma-ichikawa/statphys-ml.git
cd statphys-ml
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"
```

Requires Python ≥ 3.10 (PyTorch, NumPy, SciPy, Matplotlib, pandas are installed automatically).

## Quick Start

```python
import statphys

# Online SGD vs exact ODE theory (linear regression), with plots
result = statphys.quick_online(d=400, lr=0.5, t_max=10)

# Ridge regression at several alpha = n/d vs replica theory
result = statphys.quick_replica(d=200, reg_param=0.1)

# Theory-free teacher-student experiment (works for any architecture)
result = statphys.quick_experiment("random_mlp", alphas=[1, 2, 4, 8])

# Physics dashboard for an LLM-style transformer: magnetization, replica
# overlap, susceptibility, Binder cumulant + generalization error vs alpha
result = statphys.quick_order_parameters("tiny_gpt", alphas=[1, 2, 4, 8, 16])

# 2D numerical phase diagram with an estimated phase boundary
result = statphys.quick_phase_diagram("sparse_teacher", "sparsity",
                                      [0.5, 0.8, 0.9, 0.95])
```

### Phase transitions for any architecture

Every observable is defined in *function space* on a shared probe set, so the
same order parameters apply to linear models, MLPs, CNNs, LSTMs, attention,
and tiny GPTs — no analytic theory required:

| Observable | Meaning |
|---|---|
| $\hat m$ | teacher-student overlap (magnetization); noise-independent recovery measure |
| $q_{ab}$ | overlap between independently trained students (replica order parameter) |
| $\epsilon_g$ | generalization error on fresh samples |
| $\chi_m = d\,\mathrm{Var}[\hat m]$ | susceptibility; peaks at the transition |
| Binder $U_4$ | finite-size-scaling estimate of the critical point |

```python
from statphys.experiment import architecture_experiment

exp = architecture_experiment("tiny_gpt", d=256, teacher_init="normal")
result = exp.run_order_parameters(alphas=[2, 4, 8, 16], n_replicas=4)

from statphys.vis import plot_order_parameter_dashboard
plot_order_parameter_dashboard(result, title="tiny GPT")
```

### Command-line interface

Everything is also available without writing Python — the `statphys`
command is installed with the package:

```bash
statphys list                                        # presets / architectures / studies
statphys order-params tiny_gpt --alphas 1 2 4 8      # physics dashboard -> PNG + JSON
statphys phase-diagram sparse_teacher sparsity 0.5 0.8 0.95
statphys study grokking                              # ready-made studies
statphys study all --output-dir phase_results
```

Ready-made studies cover the classic and the modern phenomenology:
committee specialization, sparse-recovery finite-size scaling, 2D phase
diagrams, hidden-manifold data, tiny GPT, **grokking** (delayed
generalization), **Gaussian universality** of learning curves,
**model-wise double descent**, and **data-scaling exponents** across
architectures.

Verify the whole architecture zoo locally or as a Slurm job array:

```bash
python scripts/verify_architectures.py --arch all
python scripts/verify_architectures.py --submit-slurm --partition <name> --gpus 1
```

Run the test suite with `pytest tests/`.

## Documentation

Detailed documentation lives in [`docs/`](docs/README.md):

| Guide | Contents |
|---|---|
| [Getting Started](docs/getting_started.md) | Installation, one-liner API, full replica/online workflows |
| [Component Catalog](docs/components.md) | All datasets, models, losses, theory scenarios, and utilities |
| [General Experiments](docs/experiments.md) | Theory-free teacher-student framework, presets, architecture zoo |
| [Visualization](docs/visualization.md) | Plotters, phase portraits, and GIF/MP4 animations |
| [Slurm Guide](docs/slurm.md) | Cluster execution: single jobs, arrays, verification CLI |
| [Key Concepts](docs/concepts.md) | Order parameters, $E_g$ formulas, scaling conventions |
| [Theory & Literature](docs/THEORY.md) | Feature ↔ paper map; exact vs heuristic status |
| [Package Structure](docs/package_structure.md) | Source-tree layout and design conventions |

## Examples

See the [`examples/`](examples/) directory:

| File | Description |
|------|-------------|
| `basic_usage.ipynb` | Comprehensive tutorial covering all features |
| `dataset_gallery.ipynb` | Visualization of all 22 supported datasets |
| `model_gallery.ipynb` | Visualization of all 19 supported models |
| `theory_vs_simulation_verification_en.ipynb` | Theory vs simulation walkthrough (also [日本語](examples/theory_vs_simulation_verification_ja.ipynb)) |
| `replica_ridge_regression.py` | Ridge regression with replica theory |
| `online_sgd_learning.py` | Online SGD dynamics |
| `committee_machine.py` | Committee machine analysis |
| `general_teacher_student.py` | Theory-free experiments (sparse recovery, attention teacher, custom setups) |

## License

BSD-3-Clause License — see [LICENSE.txt](LICENSE.txt) for details.

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

## Author

**Yuma Ichikawa, Ph.D.**

- **Website**: [https://ichikawa-laboratory.com/](https://ichikawa-laboratory.com/)
- **Twitter**: [@yuma_1_or](https://x.com/yuma_1_or)
- **Google Scholar**: [Yuma Ichikawa](https://scholar.google.com/citations?user=yuma-ichikawa)
- **GitHub**: [yuma-ichikawa](https://github.com/yuma-ichikawa)
- **Email**: yuma.ichikawa@a.riken.jp

## Disclaimer

This project is an **independent personal project** developed by Yuma Ichikawa.
It is **not affiliated with, sponsored by, or endorsed by any organization**, including the author's employer.
All views and opinions expressed in this project are solely those of the author.
