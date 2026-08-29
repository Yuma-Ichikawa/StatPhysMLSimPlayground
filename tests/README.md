# Tests

This directory contains the test suite for `statphys-ml`.

## Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_dataset.py                # Dataset generation tests
├── test_loss.py                   # Loss function tests
├── test_model.py                  # Model architecture tests
├── test_simulation.py             # Simulation runner tests
├── test_theory.py                 # Theory solver tests
├── test_experiment.py             # Theory-free teacher-student experiment tests
├── test_observables.py            # Function-space order parameter tests
├── test_order_params.py           # Automatic order-parameter extraction tests
├── test_online_committee.py       # Exact online committee-machine dynamics tests
├── test_frontier.py               # SFT/RLHF/weak-to-strong/collapse/ICL paradigm tests
├── test_animation.py              # GIF/MP4 animation tests
├── test_cli_dynamics.py           # `statphys` CLI subcommand tests
├── test_zoo_slurm.py              # Architecture zoo + Slurm utility tests
├── test_realistic.py              # Multi-index/mixture/lazy-rich/LoRA setting tests
├── test_realistic_extensions.py   # Extended realistic-setting checks
├── test_paper_contract.py         # Manuscript figure/macro contract tests
├── test_phase_tensor_data.py      # phase_tensor data-pipeline tests
├── test_phase_tensor_reporting.py # phase_tensor reporting/paper tests
├── test_audit_regressions.py      # Regression tests for audited fixes
├── test_fixes.py                  # Regression tests for earlier correctness fixes
├── atlas/                         # Tests for src/statphys/atlas (audited scientific atlas)
├── continuation/                  # Tests for src/statphys/continuation (phase-continuation)
└── predictive/                    # Tests for src/statphys/predictive (predictive pipeline)
```

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run with verbose output

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_dataset.py
```

### Run specific test class or function

```bash
# Run a specific class
pytest tests/test_model.py::TestLinearRegression

# Run a specific test
pytest tests/test_model.py::TestLinearRegression::test_forward
```

### Run with coverage

```bash
pytest tests/ --cov=statphys --cov-report=html
```

This generates an HTML coverage report in `htmlcov/`.

### Run tests matching a pattern

```bash
# Run all tests with "replica" in the name
pytest tests/ -k "replica"
```

## Test Categories

| Module | Description | Key Tests |
|--------|-------------|-----------|
| `test_dataset` | Data generation | Shape validation, reproducibility, device transfer |
| `test_model` | Model architectures | Forward pass, order params, weight retrieval |
| `test_loss` | Loss functions | MSE, Ridge, LASSO, Hinge, Logistic |
| `test_theory` | Theory solvers | Saddle-point convergence, ODE integration |
| `test_simulation` | Simulation runners | Replica & online simulation execution |
| `test_experiment` | Theory-free teacher-student experiments | Presets, sweeps, architecture zoo |
| `test_observables`, `test_order_params` | Function-space order parameters | Magnetization, overlap, susceptibility, Binder, auto-extraction |
| `test_online_committee` | Online committee-machine dynamics | Exact Saad-Solla ODEs, plateau escape time |
| `test_frontier` | Modern paradigms as physics | SFT, RLHF, weak-to-strong, collapse, ICL |
| `test_cli_dynamics` | `statphys` CLI | Guided workflow subcommands end to end |
| `test_zoo_slurm` | Architecture zoo + Slurm | Job/array rendering, no hardcoded paths |
| `test_realistic`, `test_realistic_extensions` | Modern realistic settings | Multi-index, mixture, lazy/rich, LoRA |
| `atlas/`, `continuation/`, `predictive/` | Atlas / phase-continuation / predictive pipelines | Schemas, orchestration, aggregation, reproducibility contract |

## Fixtures

Common fixtures are defined in `conftest.py`:

- `reset_seed`: Auto-reset random seed before each test
- `small_d`, `medium_d`: Standard dimensions
- `gaussian_dataset`: Pre-configured Gaussian dataset
- `linear_model`: Pre-configured linear model
- `ridge_loss`: Pre-configured Ridge loss
- `replica_config`, `online_config`: Minimal simulation configs

## Writing New Tests

1. Create test classes with `Test` prefix
2. Use descriptive method names: `test_<what>_<condition>`
3. Use fixtures from `conftest.py` for common setup
4. Keep tests fast (small dimensions, few iterations)

Example:

```python
class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self, gaussian_dataset):
        """Test that basic functionality works."""
        # Use the fixture
        X, y = gaussian_dataset.generate_dataset(n_samples=10)
        assert X.shape == (10, gaussian_dataset.d)

    def test_edge_case(self):
        """Test edge case behavior."""
        # ...
```

## CI Integration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

GitHub Actions CI is configured in `.github/workflows/ci.yml`.
