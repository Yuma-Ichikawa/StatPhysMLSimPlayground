"""Optimizer classes and a common adapter for fair geometry comparisons."""

from __future__ import annotations

import inspect
from typing import Any, Iterable

import torch


class CompositeOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {str(index): optimizer.state_dict() for index, optimizer in enumerate(self.optimizers)}


def _accepted_kwargs(cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls.__init__).parameters
    return {name: value for name, value in kwargs.items() if name in parameters}


def build_optimizer(
    name: str,
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.95,
    rank: int = 32,
) -> Any:
    normalized = name.lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if normalized == "sgd_m":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )
    if normalized == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
    if normalized == "muon":
        named = list(model.named_parameters())
        matrix = [
            parameter
            for parameter_name, parameter in named
            if parameter.requires_grad
            and parameter.ndim == 2
            and "embedding" not in parameter_name
            and "readout" not in parameter_name
        ]
        matrix_ids = {id(parameter) for parameter in matrix}
        auxiliary = [
            parameter
            for _, parameter in named
            if parameter.requires_grad and id(parameter) not in matrix_ids
        ]
        optimizers: list[torch.optim.Optimizer] = []
        if matrix:
            optimizers.append(
                torch.optim.Muon(
                    matrix,
                    lr=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        if auxiliary:
            optimizers.append(
                torch.optim.AdamW(
                    auxiliary,
                    lr=learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay,
                )
            )
        return CompositeOptimizer(optimizers)
    if normalized in {"soap", "lion", "galore"}:
        try:
            from pytorch_optimizer import load_optimizer
        except ImportError as error:
            raise RuntimeError(
                f"optimizer {normalized} requires the phase-tensor optional dependency"
            ) from error
        cls = load_optimizer(normalized)
        kwargs = _accepted_kwargs(
            cls,
            {
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
                "momentum": momentum,
                "rank": rank,
                "precondition_frequency": 10,
            },
        )
        return cls(parameters, **kwargs)
    raise ValueError(f"unsupported optimizer: {name}")


def set_learning_rate(optimizer: Any, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)
