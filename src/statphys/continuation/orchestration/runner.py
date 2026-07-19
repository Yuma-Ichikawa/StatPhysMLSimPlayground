"""Single-task execution with atomic state transitions and lazy family dispatch."""

from __future__ import annotations

import json
from pathlib import Path
import time

from ..core.artifacts import RunStore
from ..core.metrics import resolve_device, validate_metrics
from ..core.registry import resolve_runner
from ..core.schema import TaskSpec, read_manifest


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message)


def run_task(
    task: TaskSpec,
    output_root: str | Path,
    *,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, float]:
    store = RunStore(output_root)
    if store.completed(task) and not overwrite:
        payload = json.loads((store.directory(task) / "metrics.json").read_text(encoding="utf-8"))
        return {str(name): float(value) for name, value in payload.items()}
    store.begin(task, overwrite=overwrite)
    resolved = resolve_device(device)
    started = time.perf_counter()
    try:
        metrics, arrays = resolve_runner(task)(task, resolved)
        validated = validate_metrics(metrics)
        store.complete(
            task,
            validated,
            arrays,
            elapsed_seconds=time.perf_counter() - started,
            device=str(resolved),
        )
        return validated
    except BaseException as error:
        if resolved.type == "cuda" and _is_cuda_oom(error):
            try:
                import torch

                torch.cuda.empty_cache()
            except BaseException:
                pass
            store.begin(task, overwrite=True)
            resolved = resolve_device("cpu")
            try:
                metrics, arrays = resolve_runner(task)(task, resolved)
                validated = validate_metrics(metrics)
                store.complete(
                    task,
                    validated,
                    arrays,
                    elapsed_seconds=time.perf_counter() - started,
                    device=str(resolved),
                )
                return validated
            except BaseException as fallback_error:
                store.fail(task, fallback_error)
                raise
        store.fail(task, error)
        raise


def run_manifest_task(
    manifest_path: str | Path,
    index: int,
    output_root: str | Path,
    *,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, float]:
    manifest = read_manifest(manifest_path)
    if index < 0 or index >= len(manifest.tasks):
        raise IndexError(f"task index {index} outside [0, {len(manifest.tasks)})")
    return run_task(
        manifest.tasks[index],
        output_root,
        device=device,
        overwrite=overwrite,
    )
