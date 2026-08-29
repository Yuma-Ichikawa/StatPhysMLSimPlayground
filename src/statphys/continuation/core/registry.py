"""Lazy experiment-family registry; importing it never initializes a GPU."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from statphys.core import Fidelity, RunnerContract, TheoryStatus

from .metrics import COMMON_METRICS
from .schema import Domain, TaskSpec

Runner = Callable[[TaskSpec, Any], tuple[dict[str, float], dict[str, Any]]]

_RUNNER_PATHS: dict[Domain, dict[str, str]] = {
    Domain.TRANSFORMER: {
        "anchor": "..domains.transformer.anchor:run_transformer",
        "architecture": "..domains.transformer.architecture:run_architecture_ladder",
        **dict.fromkeys(
            (
                "heads",
                "attention_mlp",
                "icl",
                "long_context",
                "lora",
                "glass",
                "optimizer",
                "data_bridge",
                "cot",
                "generation",
            ),
            "..domains.transformer.algorithms:run_transformer_algorithm",
        ),
        **dict.fromkeys(
            ("moe", "retrieval", "multimodal", "compression", "lifecycle", "discovery"),
            "..domains.transformer.systems:run_transformer_system",
        ),
        "learned_decoder": "..domains.transformer.learned:run_learned_transformer",
        **dict.fromkeys(
            (
                "tensor_mlp",
                "tensor_optimizer",
                "tensor_objective",
                "tensor_scaling",
                "tensor_realdata",
                "tensor_residual",
            ),
            "statphys.phase_tensor.runner:run_phase_tensor",
        ),
    },
    Domain.DIFFUSION: {
        "anchor": "..domains.diffusion.anchor:run_diffusion",
        **dict.fromkeys(
            ("guidance", "trajectory", "locality", "memorization"),
            "..domains.diffusion.dynamics:run_diffusion_program",
        ),
        "learned_score": "..domains.diffusion.learned:run_learned_diffusion",
    },
    Domain.RL: {
        "anchor": "..domains.reinforcement.anchor:run_reinforcement",
        **dict.fromkeys(
            ("entropy_flow", "goodhart", "rollout", "optimizer", "preference"),
            "..domains.reinforcement.mdp:run_reinforcement_program",
        ),
        "learned_policy": "..domains.reinforcement.learned:run_learned_policy",
    },
    Domain.MULTIAGENT: {
        "anchor": "..domains.multiagent.anchor:run_multiagent",
        **dict.fromkeys(
            ("debate", "minority", "influence", "roles", "scaling"),
            "..domains.multiagent.programs:run_multiagent_program",
        ),
        "learned_agents": "..domains.multiagent.learned:run_learned_agents",
    },
    Domain.CROSS: {
        **dict.fromkeys(
            ("diffusion_language_rl", "diffusion_policy_rl", "multiagent_rl", "moe_multiagent"),
            "..domains.cross_domain:run_cross_domain",
        ),
        **dict.fromkeys(
            ("assumption_pairs", "renormalized_bridge", "critical_window", "outcome_atlas"),
            "..domains.continuation:run_continuation_diagnostic",
        ),
    },
}

_TEMPORAL_PHASE_METRICS = frozenset(
    {
        "order_parameter",
        "signed_order_parameter",
        "trajectory_mean_task_order",
        "temporal_order_variance",
        "temporal_kurtosis_proxy",
        "trajectory_sign_entropy",
        "generalization_error",
        "ood_generalization_error",
        "effective_multiplicity",
        "interaction_range",
        "oracle_gap",
        "intervention_response",
    }
)

_DEFAULT_CONTRACT = RunnerContract(
    required_metrics=frozenset(COMMON_METRICS),
    theory_status=TheoryStatus.EMPIRICAL_ONLY,
    fidelity=Fidelity.TRAINABLE_SYNTHETIC,
    phase_ensemble="inner_replica",
)

_PHASE_TENSOR_CONTRACT = RunnerContract(
    required_metrics=_TEMPORAL_PHASE_METRICS,
    required_arrays=frozenset({"history_step", "history_semantic_order"}),
    theory_status=TheoryStatus.EMPIRICAL_ONLY,
    fidelity=Fidelity.TRAINABLE_SYNTHETIC,
    phase_ensemble="outer_seed",
)


def supported_families() -> dict[str, tuple[str, ...]]:
    return {domain.value: tuple(families) for domain, families in _RUNNER_PATHS.items()}


def is_supported(domain: Domain | str, family: str) -> bool:
    parsed = Domain.parse(domain)
    return family in _RUNNER_PATHS.get(parsed, {})


def resolve_runner(task: TaskSpec) -> Runner:
    try:
        path = _RUNNER_PATHS[task.domain][task.family]
    except KeyError as error:
        raise KeyError(
            f"unsupported experiment family: {task.domain.value}/{task.family}"
        ) from error
    module_name, attribute = path.split(":", 1)
    return getattr(import_module(module_name, package=__package__), attribute)


def runner_contract(task: TaskSpec) -> RunnerContract:
    """Return the metric, ensemble, theory, and fidelity contract for a task."""
    if task.domain is Domain.TRANSFORMER and task.family.startswith("tensor_"):
        return _PHASE_TENSOR_CONTRACT
    if task.domain is Domain.CROSS and task.family in {
        "assumption_pairs",
        "renormalized_bridge",
        "critical_window",
        "outcome_atlas",
    }:
        return RunnerContract(
            required_metrics=frozenset(COMMON_METRICS),
            theory_status=TheoryStatus.SYNTHETIC_GROUND_TRUTH,
            fidelity=Fidelity.PHENOMENOLOGICAL_GENERATOR,
            phase_ensemble="inner_replica",
        )
    return _DEFAULT_CONTRACT
