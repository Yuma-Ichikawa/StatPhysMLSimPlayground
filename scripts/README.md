# Scripts

Utility scripts for development, verification, and maintenance.

## Contents

```
scripts/
├── README.md
├── run_verification.py       # Core package verification (replica/online vs theory)
├── verify_architectures.py   # Teacher-student check across the architecture zoo
└── output/                   # Output directory for verification results
```

## Architecture Verification

### `verify_architectures.py`

Runs a matched teacher-student experiment for every architecture in
`statphys.experiment.zoo` (linear, mlp, deep_mlp, cnn, lstm, attention,
tiny_gpt) and checks that the student learns (test error decreases with
the sample ratio α). Results are written as JSON + PNG per architecture.

```bash
# One architecture
python scripts/verify_architectures.py --arch tiny_gpt

# All architectures, with online SGD dynamics too
python scripts/verify_architectures.py --arch all --online

# Dispatch as a Slurm job array (one task per architecture)
python scripts/verify_architectures.py --submit-slurm \
    --partition debug --gpus 1 --time-limit 01:00:00 \
    --setup "source .venv/bin/activate"
```

Outputs land in `verification_results/` (configurable with `--output-dir`);
Slurm scripts/logs go to `slurm_scripts/` and `slurm_logs/`. All paths are
relative to the working directory — nothing is machine-specific.

## Core Verification Script

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
