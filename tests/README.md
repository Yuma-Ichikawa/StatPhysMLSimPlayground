# Tests

This directory contains the test suite for `statphys-ml`.

## Structure

```
tests/
├── conftest.py          # Pytest fixtures and configuration
├── test_dataset.py      # Dataset generation tests
├── test_loss.py         # Loss function tests
├── test_model.py        # Model architecture tests
├── test_simulation.py   # Simulation runner tests
└── test_theory.py       # Theory solver tests
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
