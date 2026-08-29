#!/usr/bin/env bash
set -euo pipefail

: "${STATPHYS_MANIFEST:?}"
: "${STATPHYS_OUTPUT:?}"
: "${STATPHYS_DATA_ROOT:?}"
: "${REPO_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
bash "$REPO_ROOT/scripts/phase_tensor/setup-node.sh"
PYTHON="${STATPHYS_VENV_ROOT:-/tmp/${USER}/statphys-phase-tensor-venv}/bin/python"
TASKS_PER_ARRAY="${TASKS_PER_ARRAY:-32}"
start=$((SLURM_ARRAY_TASK_ID * TASKS_PER_ARRAY))
stop=$((start + TASKS_PER_ARRAY))
count=$("$PYTHON" -c 'import json, os; print(len(json.load(open(os.environ["STATPHYS_MANIFEST"]))["tasks"]))')
if [ "$stop" -gt "$count" ]; then
  stop="$count"
fi
cd "$REPO_ROOT"
srun "$PYTHON" -m statphys.continuation.cli run-local "$STATPHYS_MANIFEST" \
  --start "$start" --stop "$stop" --output "$STATPHYS_OUTPUT" --device auto
