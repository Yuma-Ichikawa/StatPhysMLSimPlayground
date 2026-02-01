# Scripts

Utility scripts for development, verification, and maintenance.

## Contents

```
scripts/
├── README.md
├── run_verification.py  # Package verification script
└── output/              # Output directory for verification results
```

## Verification Script

### `run_verification.py`

A comprehensive verification script that tests the main functionality of the `statphys-ml` package.

**What it tests:**
1. **Replica Simulation**: Ridge regression with Gaussian data
2. **Online SGD Simulation**: Learning dynamics tracking
3. **Model Comparison**: Parameter counts and outputs for different architectures

**Run:**

```bash
# From project root
python scripts/run_verification.py
```

**Output:**

Results are saved to `scripts/output/`:
- `replica_ridge_regression.png`: Order parameters vs alpha
- `online_sgd_learning.png`: Learning trajectories over time
- `model_comparison.png`: Parameter count comparison

## Usage Notes

- These scripts are for **development and demonstration**, not for automated testing
- For automated tests, use `pytest tests/`
- Output files are ignored by git (see `.gitignore`)
