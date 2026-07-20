#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${MANIFEST_ROOT:?set MANIFEST_ROOT to a writable artifact directory}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$MANIFEST_ROOT"
cd "$REPO_ROOT"
families=(tensor_mlp tensor_optimizer tensor_objective tensor_scaling tensor_realdata tensor_residual)
for family in "${families[@]}"; do
  "$PYTHON" -m statphys.continuation.cli expand \
    "experiments/phase_continuation/configs/${family}.toml" \
    --manifest "$MANIFEST_ROOT/${family}.json"
done

inputs=()
for family in "${families[@]}"; do
  inputs+=("$MANIFEST_ROOT/${family}.json")
done
"$PYTHON" -m statphys.continuation.cli compose "${inputs[@]}" \
  --study empirical_phase_tensor_o1_v2 \
  --output "$MANIFEST_ROOT/all.json"
