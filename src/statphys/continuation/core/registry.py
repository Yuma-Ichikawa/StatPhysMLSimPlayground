"""Lazy experiment-family registry; importing it never initializes a GPU."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .schema import Domain, TaskSpec

Runner = Callable[[TaskSpec, Any], tuple[dict[str, float], dict[str, Any]]]

_RUNNER_PATHS: dict[Domain, dict[str, str]] = {
    Domain.TRANSFORMER: {
        "anchor": "..domains.transformer.anchor:run_transformer",
        "architecture": "..domains.transformer.architecture:run_architecture_ladder",
        **{
            family: "..domains.transformer.algorithms:run_transformer_algorithm"
            for family in (
                "heads", "attention_mlp", "icl", "long_context", "lora", "glass",
                "optimizer", "data_bridge", "cot", "generation",
            )
        },
        **{
            family: "..domains.transformer.systems:run_transformer_system"
            for family in ("moe", "retrieval", "multimodal", "compression", "lifecycle", "discovery")
        },
        "learned_decoder": "..domains.transformer.learned:run_learned_transformer",
    },
    Domain.DIFFUSION: {
        "anchor": "..domains.diffusion.anchor:run_diffusion",
        **{
            family: "..domains.diffusion.dynamics:run_diffusion_program"
            for family in ("guidance", "trajectory", "locality", "memorization")
        },
        "learned_score": "..domains.diffusion.learned:run_learned_diffusion",
    },
    Domain.RL: {
        "anchor": "..domains.reinforcement.anchor:run_reinforcement",
        **{
            family: "..domains.reinforcement.mdp:run_reinforcement_program"
            for family in ("entropy_flow", "goodhart", "rollout", "optimizer", "preference")
        },
        "learned_policy": "..domains.reinforcement.learned:run_learned_policy",
    },
    Domain.MULTIAGENT: {
        "anchor": "..domains.multiagent.anchor:run_multiagent",
        **{
            family: "..domains.multiagent.programs:run_multiagent_program"
            for family in ("debate", "minority", "influence", "roles", "scaling")
        },
        "learned_agents": "..domains.multiagent.learned:run_learned_agents",
    },
    Domain.CROSS: {
        **{
            family: "..domains.cross_domain:run_cross_domain"
            for family in (
                "diffusion_language_rl", "diffusion_policy_rl", "multiagent_rl", "moe_multiagent",
            )
        },
        **{
            family: "..domains.continuation:run_continuation_diagnostic"
            for family in ("assumption_pairs", "renormalized_bridge", "critical_window", "outcome_atlas")
        },
    },
}


def supported_families() -> dict[str, tuple[str, ...]]:
    return {domain.value: tuple(families) for domain, families in _RUNNER_PATHS.items()}


def is_supported(domain: Domain | str, family: str) -> bool:
    parsed = Domain.parse(domain)
    return family in _RUNNER_PATHS.get(parsed, {})


def resolve_runner(task: TaskSpec) -> Runner:
    try:
        path = _RUNNER_PATHS[task.domain][task.family]
    except KeyError as error:
        raise KeyError(f"unsupported experiment family: {task.domain.value}/{task.family}") from error
    module_name, attribute = path.split(":", 1)
    return getattr(import_module(module_name, package=__package__), attribute)
