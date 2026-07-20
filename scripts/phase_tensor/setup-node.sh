#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
VENV_ROOT="${STATPHYS_VENV_ROOT:-/tmp/${USER}/statphys-phase-tensor-venv}"
LOCK_ROOT="${VENV_ROOT}.lock"

if [ -x "$VENV_ROOT/bin/python" ]; then
  exit 0
fi

until mkdir "$LOCK_ROOT" 2>/dev/null; do
  sleep 5
  if [ -x "$VENV_ROOT/bin/python" ]; then
    exit 0
  fi
done
trap 'rmdir "$LOCK_ROOT"' EXIT

if [ ! -x "$VENV_ROOT/bin/python" ]; then
  python3 -m venv "$VENV_ROOT"
  "$VENV_ROOT/bin/python" -m pip install --upgrade pip
  "$VENV_ROOT/bin/python" -m pip install -r "$REPO_ROOT/requirements/phase-tensor.txt"
  "$VENV_ROOT/bin/python" -m pip freeze > "$VENV_ROOT/requirements.lock"
fi
