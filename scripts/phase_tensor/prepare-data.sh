#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${STATPHYS_DATA_ROOT:?set STATPHYS_DATA_ROOT to the prepared-corpus root}"
PYTHON="${PYTHON:-python3}"
MAX_BYTES="${STATPHYS_CORPUS_BYTES:-64000000}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
"$PYTHON" -m statphys.phase_tensor.cli prepare-data \
  tinystories simplestories fineweb_edu dolma \
  --root "$STATPHYS_DATA_ROOT" --max-bytes "$MAX_BYTES"
