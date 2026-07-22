#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${MANIFEST_ROOT:?set MANIFEST_ROOT to the manifest directory}"
: "${STATPHYS_OUTPUT:?set STATPHYS_OUTPUT to the run-artifact root}"
: "${STATPHYS_DATA_ROOT:?set STATPHYS_DATA_ROOT to the prepared-corpus root}"
: "${ARRAY_SCRIPT:?set ARRAY_SCRIPT to a site-specific Slurm wrapper}"

partition=$(sed -n 's/^#SBATCH[[:space:]]\+--partition=//p' "$ARRAY_SCRIPT" | sed -n '1p')
case "$partition" in
  spark_*) ;;
  *)
    printf 'refusing non-Spark array wrapper: partition=%s script=%s\n' \
      "${partition:-missing}" "$ARRAY_SCRIPT" >&2
    exit 2
    ;;
esac

JOB_IDS_FILE="${JOB_IDS_FILE:-$MANIFEST_ROOT/job_ids.env}"
TASKS_PER_ARRAY="${TASKS_PER_ARRAY:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
REALDATA_DEPENDENCY="${REALDATA_DEPENDENCY:-}"
mkdir -p "$STATPHYS_OUTPUT"
if [ "${RESET_JOB_IDS_FILE:-1}" = "1" ]; then
  : > "$JOB_IDS_FILE"
else
  touch "$JOB_IDS_FILE"
fi

for manifest in "$MANIFEST_ROOT"/tensor_*.json; do
  [ -f "$manifest" ] || continue
  family=$(basename "$manifest" .json)
  count=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$manifest")
  arrays=$(((count + TASKS_PER_ARRAY - 1) / TASKS_PER_ARRAY))
  dependency=()
  if [ "$family" = "tensor_realdata" ] && [ -n "$REALDATA_DEPENDENCY" ]; then
    dependency=(--dependency="afterok:${REALDATA_DEPENDENCY}")
  fi
  job_id=$(sbatch --parsable "${dependency[@]}" --array="0-$((arrays - 1))%${MAX_PARALLEL}" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,STATPHYS_MANIFEST=$manifest,STATPHYS_OUTPUT=$STATPHYS_OUTPUT,STATPHYS_DATA_ROOT=$STATPHYS_DATA_ROOT,TASKS_PER_ARRAY=$TASKS_PER_ARRAY" \
    --job-name="$family" "$ARRAY_SCRIPT")
  printf '%s=%s\n' "$family" "$job_id" | tee -a "$JOB_IDS_FILE"
done
