"""Transactional, content-addressed artifacts for atlas experiments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .schema import RunSpec


TERMINAL_STATES = {"completed", "failed", "cancelled", "preempted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def atomic_json(path: str | Path, value: Any) -> None:
    """Write JSON through a same-filesystem temporary file and atomic rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_npz(path: str | Path, **arrays: Any) -> None:
    """Atomically save compressed NumPy arrays without importing NumPy at startup."""

    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=target.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        # np.savez appends .npz only when the path has no matching suffix.
        generated = Path(temporary)
        os.replace(generated, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_provenance(repo: str | Path | None = None) -> dict[str, Any]:
    """Collect enough environment state to audit a numerical claim."""

    root = Path(repo).resolve() if repo is not None else None
    git_commit = _command_output(["git", "rev-parse", "HEAD"], root)
    git_status = _command_output(["git", "status", "--short"], root)
    slurm = {
        name.lower(): os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_PARTITION",
            "SLURMD_NODENAME",
        )
        if os.environ.get(name) is not None
    }
    accelerator: dict[str, Any] = {}
    try:
        import torch

        accelerator = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "hip_runtime": getattr(torch.version, "hip", None),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except (ImportError, RuntimeError):
        accelerator = {"torch": None, "cuda_available": False}
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "slurm": slurm,
        "accelerator": accelerator,
    }


class RunArtifactStore:
    """Owns one artifact tree and its append-only global manifest.

    Layout::

        root/
          manifest.jsonl
          runs/<content-id>/{spec,provenance,status,summary}.json
                            trajectories.npz
                            checkpoints/
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.manifest = self.root / "manifest.jsonl"

    def run_dir(self, run: RunSpec | str) -> Path:
        run_id = run.run_id if isinstance(run, RunSpec) else run
        if not run_id or any(c not in "0123456789abcdef" for c in run_id.lower()):
            raise ValueError(f"invalid content id: {run_id!r}")
        return self.runs / run_id

    @contextlib.contextmanager
    def _manifest_lock(self) -> Iterator[None]:
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".manifest.lock"
        with lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_event(self, event: Mapping[str, Any]) -> None:
        record = {"timestamp": utc_now(), **dict(event)}
        line = json.dumps(record, sort_keys=True, default=_json_default) + "\n"
        with self._manifest_lock():
            with self.manifest.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def begin(self, run: RunSpec, repo: str | Path | None = None, overwrite: bool = False) -> Path:
        directory = self.run_dir(run)
        status_path = directory / "status.json"
        if status_path.exists() and not overwrite:
            state = json.loads(status_path.read_text()).get("state")
            if state == "completed":
                raise FileExistsError(f"run {run.run_id} is already completed")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "checkpoints").mkdir(exist_ok=True)
        atomic_json(directory / "spec.json", run.to_dict())
        atomic_json(directory / "provenance.json", collect_provenance(repo))
        self.set_status(run.run_id, "running")
        return directory

    def set_status(self, run_id: str, state: str, **details: Any) -> None:
        directory = self.run_dir(run_id)
        previous: dict[str, Any] = {}
        path = directory / "status.json"
        if path.exists():
            previous = json.loads(path.read_text())
        attempts = int(previous.get("attempts", 0))
        if state == "running" and previous.get("state") != "running":
            attempts += 1
        status = {
            "run_id": run_id,
            "state": state,
            "updated_at": utc_now(),
            "attempts": attempts,
            **details,
        }
        atomic_json(path, status)
        self.append_event({"event": "status", **status})

    def save_summary(self, run_id: str, summary: Mapping[str, Any]) -> Path:
        path = self.run_dir(run_id) / "summary.json"
        atomic_json(path, dict(summary))
        self.append_event({"event": "summary", "run_id": run_id, "path": str(path)})
        return path

    def save_arrays(self, run_id: str, name: str = "trajectories", **arrays: Any) -> Path:
        if "/" in name or name.startswith("."):
            raise ValueError("array artifact name must be a simple filename")
        path = self.run_dir(run_id) / f"{name}.npz"
        atomic_npz(path, **arrays)
        checksum = sha256_file(path)
        self.append_event(
            {"event": "artifact", "run_id": run_id, "path": str(path), "sha256": checksum}
        )
        return path

    def pending(self, specs: list[RunSpec], retry_failed: bool = True) -> list[RunSpec]:
        result: list[RunSpec] = []
        for spec in specs:
            path = self.run_dir(spec) / "status.json"
            if not path.exists():
                result.append(spec)
                continue
            state = json.loads(path.read_text()).get("state")
            if state != "completed" and (retry_failed or state not in TERMINAL_STATES):
                result.append(spec)
        return result

