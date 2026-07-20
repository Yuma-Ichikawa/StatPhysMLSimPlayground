#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${STATPHYS_MANIFEST:?set STATPHYS_MANIFEST to the composed manifest}"
: "${STATPHYS_OUTPUT:?set STATPHYS_OUTPUT to the run-artifact root}"
: "${PAPER_DIR:?set PAPER_DIR to the manuscript directory}"
PYTHON="${PYTHON:-python3}"
TAXONOMY_PATH="${TAXONOMY_PATH:-$REPO_ROOT/experiments/phase_continuation/phase_tensor_taxonomy.toml}"
AGGREGATE_PATH="${AGGREGATE_PATH:-$STATPHYS_OUTPUT/aggregate.json}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

"$PYTHON" -m statphys.phase_tensor.cli aggregate "$STATPHYS_MANIFEST" \
  --runs "$STATPHYS_OUTPUT" --output "$AGGREGATE_PATH"
"$PYTHON" -m statphys.phase_tensor.cli plot "$AGGREGATE_PATH" \
  --output "$PAPER_DIR/figures" --taxonomy "$TAXONOMY_PATH"
"$PYTHON" -m statphys.phase_tensor.cli paper "$AGGREGATE_PATH" \
  --output "$PAPER_DIR/generated/phase_tensor_results.tex"
