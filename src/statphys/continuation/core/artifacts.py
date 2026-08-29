"""Transactional, content-addressed artifacts for independent Slurm workers."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import TaskSpec


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def directory(self, task: TaskSpec) -> Path:
        return self.root / "runs" / task.run_id

    def completed(self, task: TaskSpec) -> bool:
        path = self.directory(task) / "status.json"
        if not path.exists():
            return False
        return json.loads(path.read_text()).get("state") == "completed"

    def _attempt(self, task: TaskSpec) -> int:
        path = self.directory(task) / "status.json"
        if not path.exists():
            return 0
        return int(json.loads(path.read_text()).get("attempt", 0))

    def attempt_directory(self, task: TaskSpec, attempt: int | None = None) -> Path:
        number = self._attempt(task) if attempt is None else int(attempt)
        if number < 1:
            raise ValueError("an execution attempt has not been started")
        return self.directory(task) / "attempts" / f"{number:04d}"

    def begin(self, task: TaskSpec, overwrite: bool = False) -> Path:
        directory = self.directory(task)
        if self.completed(task) and not overwrite:
            raise FileExistsError(f"completed immutable run already exists: {task.run_id}")
        directory.mkdir(parents=True, exist_ok=True)
        attempt = self._attempt(task) + 1
        attempt_directory = self.attempt_directory(task, attempt)
        attempt_directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(directory / "spec.json", task.to_dict())
        _atomic_json(attempt_directory / "spec.json", task.to_dict())
        status = {
            "state": "running",
            "attempt": attempt,
            "scientific_condition_id": task.condition_id,
            "started_unix": time.time(),
        }
        _atomic_json(directory / "status.json", status)
        _atomic_json(attempt_directory / "status.json", status)
        provenance = {
            "platform_system": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "attempt": attempt,
        }
        _atomic_json(
            attempt_directory / "provenance.json",
            provenance,
        )
        _atomic_json(directory / "provenance.json", provenance)
        return attempt_directory

    def complete(
        self,
        task: TaskSpec,
        metrics: Mapping[str, float],
        arrays: Mapping[str, Any],
        *,
        elapsed_seconds: float,
        device: str,
    ) -> Path:
        directory = self.directory(task)
        attempt = self._attempt(task)
        attempt_directory = self.attempt_directory(task, attempt)
        _atomic_json(attempt_directory / "metrics.json", metrics)
        _atomic_npz(attempt_directory / "arrays.npz", arrays)
        _atomic_json(directory / "metrics.json", metrics)
        _atomic_npz(directory / "arrays.npz", arrays)
        status = {
            "state": "completed",
            "attempt": attempt,
            "scientific_condition_id": task.condition_id,
            "elapsed_seconds": float(elapsed_seconds),
            "device": device,
            "finished_unix": time.time(),
        }
        _atomic_json(directory / "status.json", status)
        _atomic_json(attempt_directory / "status.json", status)
        return directory

    def fail(self, task: TaskSpec, error: BaseException) -> None:
        directory = self.directory(task)
        attempt = self._attempt(task)
        status = {
            "state": "failed",
            "attempt": attempt,
            "scientific_condition_id": task.condition_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(error))[-12000:],
            "finished_unix": time.time(),
        }
        _atomic_json(directory / "status.json", status)
        _atomic_json(self.attempt_directory(task, attempt) / "status.json", status)


__all__ = ["RunStore"]
