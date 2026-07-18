"""Instrumented optimisation loop shared by every architecture-ladder stage."""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .schema import OptimizerName, Precision, TrainingSpec


Probe = Callable[[Any, int], Mapping[str, float]]


@dataclass
class TrainingResult:
    """Optimisation outcome plus aligned scalar trajectories."""

    final_loss: float
    best_loss: float
    best_step: int
    steps: int
    converged: bool
    elapsed_seconds: float
    history: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "steps": self.steps,
            "converged": self.converged,
            "elapsed_seconds": self.elapsed_seconds,
        }


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm device was requested but torch.cuda is unavailable")
    return requested


def seed_torch(seed: int, deterministic: bool = True) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def build_optimizer(model: Any, spec: TrainingSpec) -> Any:
    import torch

    kwargs = {"lr": spec.learning_rate, "weight_decay": spec.weight_decay}
    if spec.optimizer == OptimizerName.SGD:
        return torch.optim.SGD(model.parameters(), **kwargs)
    if spec.optimizer == OptimizerName.MOMENTUM:
        return torch.optim.SGD(model.parameters(), momentum=spec.momentum, **kwargs)
    if spec.optimizer == OptimizerName.ADAMW:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"unsupported optimizer: {spec.optimizer}")


def _parameter_norm(model: Any) -> float:
    import torch

    with torch.no_grad():
        squared = sum(float(parameter.detach().float().square().sum()) for parameter in model.parameters())
    return math.sqrt(squared)


def _gradient_norm(model: Any) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def _update_norm(model: Any, initial: list[Any]) -> float:
    import torch

    with torch.no_grad():
        squared = sum(
            float((parameter.detach().float() - reference.float()).square().sum())
            for parameter, reference in zip(model.parameters(), initial)
        )
    return math.sqrt(squared)


def save_checkpoint_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Save a torch checkpoint atomically on the destination filesystem."""

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def train_supervised(
    model: Any,
    inputs: Any,
    targets: Any,
    spec: TrainingSpec,
    *,
    seed: int,
    device: str = "auto",
    l2_coefficient: float = 0.0,
    probe: Probe | None = None,
    checkpoint_dir: str | Path | None = None,
) -> TrainingResult:
    """Train on a fixed teacher data set with reproducible minibatch sampling.

    The reported data loss is ``||prediction-target||²/(2 * number of scalar
    targets)``.  Explicit L2 regularisation is separated from optimizer weight
    decay so the bridge experiment can match its statistical-mechanics energy.
    """

    import torch

    if l2_coefficient < 0:
        raise ValueError("l2_coefficient must be non-negative")
    selected_device = resolve_device(device)
    seed_torch(seed, spec.deterministic)
    model = model.to(selected_device)
    inputs = inputs.to(selected_device)
    targets = targets.to(selected_device)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same number of examples")
    if spec.precision == Precision.FLOAT64:
        model = model.double()
        inputs = inputs.double()
        targets = targets.double()

    optimizer = build_optimizer(model, spec)
    generator_device = selected_device if selected_device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    n_examples = int(inputs.shape[0])
    batch_size = n_examples if spec.batch_size is None else min(spec.batch_size, n_examples)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    initial_norm = max(_parameter_norm(model), 1e-30)

    history: dict[str, list[float]] = {
        "step": [],
        "data_loss": [],
        "objective": [],
        "gradient_norm": [],
        "weight_norm": [],
        "relative_update_norm": [],
    }
    best_loss = float("inf")
    best_step = 0
    stale = 0
    converged = False
    start = time.perf_counter()
    final_loss = float("nan")

    use_amp = spec.precision == Precision.BFLOAT16 and selected_device.startswith("cuda")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if use_amp
        else contextlib.nullcontext()
    )

    for step in range(spec.max_steps + 1):
        should_record = step == 0 or step % spec.log_interval == 0 or step == spec.max_steps
        should_probe = step == 0 or step % max(1, spec.checkpoint_interval) == 0

        if step < spec.max_steps:
            if batch_size == n_examples:
                batch_inputs, batch_targets = inputs, targets
            else:
                indices = torch.randint(
                    n_examples, (batch_size,), generator=generator, device=inputs.device
                )
                batch_inputs, batch_targets = inputs[indices], targets[indices]
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                prediction = model(batch_inputs)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                data_loss = 0.5 * (prediction - batch_targets).square().mean()
                regularizer = sum(parameter.square().sum() for parameter in model.parameters())
                objective = data_loss + 0.5 * l2_coefficient * regularizer
            objective.backward()
            gradient_norm = _gradient_norm(model)
            if spec.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
        else:
            gradient_norm = float("nan")

        if should_record or should_probe:
            model.eval()
            with torch.no_grad(), autocast():
                full_prediction = model(inputs)
                if isinstance(full_prediction, tuple):
                    full_prediction = full_prediction[0]
                full_loss = 0.5 * (full_prediction - targets).square().mean()
                full_regularizer = sum(
                    parameter.square().sum() for parameter in model.parameters()
                )
                full_objective = full_loss + 0.5 * l2_coefficient * full_regularizer
            final_loss = float(full_loss)
            objective_value = float(full_objective)

            if should_record:
                history["step"].append(float(step))
                history["data_loss"].append(final_loss)
                history["objective"].append(objective_value)
                history["gradient_norm"].append(gradient_norm)
                history["weight_norm"].append(_parameter_norm(model))
                history["relative_update_norm"].append(_update_norm(model, initial) / initial_norm)

            if should_probe and probe is not None:
                measured = probe(model, step)
                for name, value in measured.items():
                    history.setdefault(name, []).append(float(value))
                history.setdefault("probe_step", []).append(float(step))

            threshold = spec.convergence_rtol * max(1.0, abs(best_loss))
            if final_loss < best_loss - threshold:
                best_loss = final_loss
                best_step = step
                stale = 0
            elif step >= spec.min_steps:
                stale += spec.log_interval if should_record else 0

            if checkpoint_dir is not None and should_probe:
                save_checkpoint_atomic(
                    Path(checkpoint_dir) / f"step_{step:08d}.pt",
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "training_spec": spec.to_dict(),
                        "seed": seed,
                    },
                )
            model.train()

            if spec.patience is not None and step >= spec.min_steps and stale >= spec.patience:
                converged = True
                break

    elapsed = time.perf_counter() - start
    steps = step
    return TrainingResult(
        final_loss=final_loss,
        best_loss=best_loss,
        best_step=best_step,
        steps=steps,
        converged=converged,
        elapsed_seconds=elapsed,
        history=history,
    )

