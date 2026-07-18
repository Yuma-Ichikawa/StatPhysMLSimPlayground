"""Faithful optimisation protocol for the positional--semantic bridge.

The reference experiment uses a *summed* empirical risk
``sum ||f(x)-y||² / (2 d)``.  It must not be replaced by a framework-default
mean: doing so changes the relative regularisation strength as ``n=alpha*d``
varies and therefore changes the phase diagram.
"""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any, Mapping

from .schema import OptimizerName, Precision, TrainingSpec
from .training import (
    Probe,
    TrainingResult,
    _gradient_norm,
    _parameter_norm,
    _update_norm,
    resolve_device,
    save_checkpoint_atomic,
    seed_torch,
)


def bridge_loss(prediction: Any, target: Any) -> Any:
    """Reference loss ``sum squared error / (2*d)``."""

    if prediction.shape != target.shape or target.ndim < 2:
        raise ValueError("prediction and target must have equal (..., d) shapes")
    return (prediction - target).square().sum() / (2.0 * target.shape[-1])


def _reference_optimizer(model: Any, spec: TrainingSpec) -> Any:
    import torch

    # The reference implementation uses coupled L2/weight_decay for both SGD
    # and Adam. AdamW remains available in the generic training module, but is
    # intentionally not silently substituted here.
    if spec.optimizer == OptimizerName.SGD:
        return torch.optim.SGD(
            model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
        )
    if spec.optimizer == OptimizerName.MOMENTUM:
        return torch.optim.SGD(
            model.parameters(),
            lr=spec.learning_rate,
            momentum=spec.momentum,
            weight_decay=spec.weight_decay,
        )
    if spec.optimizer == OptimizerName.ADAMW:
        # Kept explicit in provenance: this is a robustness intervention, not
        # the reference paper's torch.optim.Adam control.
        return torch.optim.AdamW(
            model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
        )
    raise ValueError(f"unsupported optimizer: {spec.optimizer}")


def train_bridge(
    model: Any,
    inputs: Any,
    targets: Any,
    spec: TrainingSpec,
    *,
    seed: int,
    device: str = "auto",
    probe: Probe | None = None,
    checkpoint_dir: str | Path | None = None,
) -> TrainingResult:
    """Run full-batch reference optimisation with a pre-training step zero."""

    import torch

    selected_device = resolve_device(device)
    seed_torch(seed, spec.deterministic)
    model = model.to(selected_device)
    inputs = inputs.to(selected_device)
    targets = targets.to(selected_device)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count")
    if spec.batch_size not in (None, inputs.shape[0]):
        raise ValueError("the exact bridge protocol is full-batch; set batch_size=None")
    if spec.precision == Precision.FLOAT64:
        model, inputs, targets = model.double(), inputs.double(), targets.double()

    optimizer = _reference_optimizer(model, spec)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    initial_norm = max(_parameter_norm(model), 1e-30)
    use_amp = spec.precision == Precision.BFLOAT16 and selected_device.startswith("cuda")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if use_amp
        else contextlib.nullcontext()
    )
    history: dict[str, list[float]] = {
        "step": [],
        "data_loss": [],
        "objective": [],
        "gradient_norm": [],
        "weight_norm": [],
        "relative_update_norm": [],
    }
    best_loss, final_loss = float("inf"), float("nan")
    best_step = 0
    stale = 0
    converged = False
    started = time.perf_counter()

    for step in range(spec.max_steps + 1):
        gradient_norm = float("nan")
        if step > 0:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                prediction = model(inputs)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                objective = bridge_loss(prediction, targets)
            objective.backward()
            gradient_norm = _gradient_norm(model)
            if spec.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()

        record = step == 0 or step % spec.log_interval == 0 or step == spec.max_steps
        checkpoint = step == 0 or step % spec.checkpoint_interval == 0 or step == spec.max_steps
        if not (record or checkpoint):
            continue
        model.eval()
        with torch.no_grad(), autocast():
            prediction = model(inputs)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            loss = bridge_loss(prediction, targets)
        final_loss = float(loss)
        l2 = 0.5 * spec.weight_decay * sum(
            float(parameter.detach().square().sum()) for parameter in model.parameters()
        )
        if record:
            history["step"].append(float(step))
            history["data_loss"].append(final_loss)
            history["objective"].append(final_loss + l2)
            history["gradient_norm"].append(gradient_norm)
            history["weight_norm"].append(_parameter_norm(model))
            history["relative_update_norm"].append(_update_norm(model, initial) / initial_norm)
        if checkpoint and probe is not None:
            for name, value in probe(model, step).items():
                history.setdefault(name, []).append(float(value))
            history.setdefault("probe_step", []).append(float(step))
        if checkpoint_dir is not None and checkpoint:
            save_checkpoint_atomic(
                Path(checkpoint_dir) / f"step_{step:08d}.pt",
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_spec": spec.to_dict(),
                    "objective_normalization": "sum_squared_error_over_2d",
                    "seed": seed,
                },
            )

        threshold = spec.convergence_rtol * max(1.0, abs(best_loss))
        if final_loss < best_loss - threshold:
            best_loss, best_step, stale = final_loss, step, 0
        elif record and step >= spec.min_steps:
            stale += spec.log_interval
        if spec.patience is not None and step >= spec.min_steps and stale >= spec.patience:
            converged = True
            break

    return TrainingResult(
        final_loss=final_loss,
        best_loss=best_loss,
        best_step=best_step,
        steps=step,
        converged=converged,
        elapsed_seconds=time.perf_counter() - started,
        history=history,
    )

