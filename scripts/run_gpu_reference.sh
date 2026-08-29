#!/usr/bin/env bash
set -euo pipefail

: "${STATPHYS_REPO:?Set STATPHYS_REPO to the checked-out repository directory}"

image="${STATPHYS_GPU_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
manifest="${STATPHYS_MANIFEST:-atlas_manifests/tensor_reference_validation.json}"
output="${STATPHYS_OUTPUT:-results/tensor_reference_gpu}"

cd "$STATPHYS_REPO"
docker pull "$image"
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  --volume "$STATPHYS_REPO:/workspace/statphys" \
  --workdir /workspace/statphys \
  --env PYTHONPATH=/workspace/statphys/src \
  --env STATPHYS_MANIFEST="$manifest" \
  --env STATPHYS_OUTPUT="$output" \
  "$image" bash -lc '
    set -euo pipefail
    python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
    python -m statphys.continuation.cli run-local "$STATPHYS_MANIFEST" --output "$STATPHYS_OUTPUT" --device cuda
    python -m statphys.continuation.cli aggregate "$STATPHYS_MANIFEST" --runs "$STATPHYS_OUTPUT" --output "$STATPHYS_OUTPUT/aggregate"
    python -m statphys.cli report "$STATPHYS_OUTPUT/aggregate/aggregate.json" --output "$STATPHYS_OUTPUT/analysis_report.html"
  '
