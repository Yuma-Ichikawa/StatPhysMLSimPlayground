# Scripts

Utility scripts for development, verification, and maintenance.

## Contents

```
scripts/
├── README.md
├── run_verification.py         # Core package verification (replica/online vs theory)
├── verify_architectures.py     # Teacher-student check across the architecture zoo
├── generate_readme_assets.py   # Regenerate the animated GIFs embedded in README
├── generate_paper_figures.py   # Generate the manuscript's figure suite (paper/generated)
├── run_phase_study.py          # Thin CLI wrapper for statphys.experiment.studies
├── run_gpu_reference.sh        # Docker-based GPU reference run of the tensor validation
├── check_docs.py               # Validate relative doc links and documentation portability
└── phase_tensor/                # Slurm/Spark-portable phase-tensor pipeline scripts
    └── README.md                # Environment variables, manifest expansion, array execution
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

## README Assets

### `generate_readme_assets.py`

Regenerates the animated GIFs embedded at the top of the repository README
(learning-curve animation, phase-plane animation, committee-machine
specialization). Everything is computed from scratch with fixed seeds.

```bash
python scripts/generate_readme_assets.py --out-dir assets --fps 20
```

## Paper figures and studies

### `generate_paper_figures.py`

Generates the manuscript's full conceptual and quantitative figure suite
(`paper/generated/`) from the public strict aggregate and generated TeX
macros. Every figure has the exact physical size 6.4 by 4.8 inches.

```bash
python scripts/generate_paper_figures.py \
    --reference evidence/tensor_reference_validation/aggregate.json \
    --confirmation-macros paper/generated/confirmation_macros.tex \
    --output paper/figures \
    --macros paper/generated/macros.tex
```

### `run_phase_study.py`

A thin CLI wrapper around the ready-made studies in
`statphys.experiment.studies` (`committee`, `fss`, `diagram`, `attention`,
`manifold`, `gpt`, `grokking`, `universality`, `double_descent`, `scaling`, ...).

```bash
python scripts/run_phase_study.py --study all --output-dir phase_results
python scripts/run_phase_study.py --study grokking --quick
```

### `run_gpu_reference.sh`

Runs the registered tensor-reference-validation manifest end to end inside a
pinned PyTorch NGC Docker image on a GPU host (`run-local`, `aggregate`,
`report` via `statphys.continuation.cli`/`statphys.cli`), reading
`STATPHYS_REPO`, `STATPHYS_GPU_IMAGE`, `STATPHYS_MANIFEST`, and
`STATPHYS_OUTPUT` from the environment — nothing is hardcoded.

### `check_docs.py`

Validates every Markdown file under `docs/` for relative-link integrity and
portability (no `localhost`/loopback addresses, `file://` URLs, or absolute
`/home`, `/mnt`, `/tmp`, `/var` paths).

```bash
python scripts/check_docs.py
```

### `phase_tensor/`

The Slurm/Spark-portable phase-tensor workflow: manifest expansion, array
execution, aggregation, figure generation, and TeX macro generation, with no
site-specific paths committed to the repository. See
[`scripts/phase_tensor/README.md`](phase_tensor/README.md) for the required
environment variables and the full pipeline.

## Usage Notes

- These scripts are for **development and demonstration**, not for automated testing
- For automated tests, use `pytest tests/`
- Output files are ignored by git (see `.gitignore`)
