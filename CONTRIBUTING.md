# Contributing to StatPhys-ML

Thank you for your interest in contributing to StatPhys-ML! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/statphys-ml.git
   cd statphys-ml
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/yuma-ichikawa/statphys-ml.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. (Optional) Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes, following the [code style](#code-style) guidelines

3. Write or update tests as needed

4. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

5. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **isort** for import sorting
- **mypy** for type checking

### Running Formatters

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Style Guidelines

- Use type hints for all function signatures
- Write docstrings for all public functions and classes (Google style)
- Keep functions focused and small
- Use meaningful variable names

Example:

```python
def compute_generalization_error(
    m: float,
    q: float,
    rho: float,
) -> float:
    """Compute the generalization error from order parameters.

    Args:
        m: Student-teacher overlap.
        q: Student self-overlap.
        rho: Teacher weight norm.

    Returns:
        The generalization error E_g = rho + q - 2m.
    """
    return rho + q - 2 * m
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=statphys --cov-report=html

# Run specific test file
pytest tests/test_dataset.py

# Run tests matching a pattern
pytest tests/ -k "replica"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Use descriptive test names: `test_<what>_<condition>`
- Use fixtures from `conftest.py` for common setup

## Submitting Changes

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issues (if any)
   - Test plan

4. Wait for CI checks to pass

5. Address any review feedback

### PR Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new features
- Update CHANGELOG.md for user-facing changes

## Types of Contributions

### Bug Reports

- Use the bug report issue template
- Include steps to reproduce
- Include expected vs actual behavior
- Include environment details

### Feature Requests

- Use the feature request issue template
- Describe the use case
- Explain why it would be useful

### Documentation

- Fix typos or unclear explanations
- Add examples
- Improve docstrings

### Code

- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to StatPhys-ML!
