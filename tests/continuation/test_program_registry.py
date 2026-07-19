from __future__ import annotations

import math

import pytest
import torch

from statphys.continuation.core.metrics import COMMON_METRICS
from statphys.continuation.core.registry import resolve_runner
from statphys.continuation.schema import Domain, TaskSpec


CASES = [
    (Domain.TRANSFORMER, "architecture", "m0", 1.0),
    (Domain.TRANSFORMER, "heads", "tied", 1.0),
    (Domain.TRANSFORMER, "attention_mlp", "hybrid", 0.5),
    (Domain.TRANSFORMER, "icl", "ridge", 1.0),
    (Domain.TRANSFORMER, "long_context", "depth2", 0.1),
    (Domain.TRANSFORMER, "lora", "rank2", 0.25),
    (Domain.TRANSFORMER, "glass", "random_init", 1.0),
    (Domain.TRANSFORMER, "optimizer", "sgd", 0.01),
    (Domain.TRANSFORMER, "data_bridge", "d0", 1.0),
    (Domain.TRANSFORMER, "cot", "scratchpad", 0.1),
    (Domain.TRANSFORMER, "generation", "ancestral", 1.0),
    (Domain.TRANSFORMER, "moe", "top1", 0.2),
    (Domain.TRANSFORMER, "retrieval", "hybrid", 0.5),
    (Domain.TRANSFORMER, "multimodal", "hybrid", 0.5),
    (Domain.TRANSFORMER, "compression", "pruning", 0.5),
    (Domain.TRANSFORMER, "lifecycle", "standard", 0.5),
    (Domain.TRANSFORMER, "discovery", "js_fisher", 0.0),
    (Domain.DIFFUSION, "guidance", "cfg", 1.0),
    (Domain.DIFFUSION, "trajectory", "ancestral", 1.0),
    (Domain.DIFFUSION, "locality", "local", 0.1),
    (Domain.DIFFUSION, "memorization", "mixed", 0.5),
    (Domain.RL, "entropy_flow", "annealed", 0.2),
    (Domain.RL, "goodhart", "proxy", 1.0),
    (Domain.RL, "rollout", "finite", 1.0),
    (Domain.RL, "optimizer", "q_learning", 1.0),
    (Domain.RL, "preference", "dpo", 1.0),
    (Domain.MULTIAGENT, "debate", "debate", 1.0),
    (Domain.MULTIAGENT, "minority", "correct", 0.1),
    (Domain.MULTIAGENT, "influence", "hierarchical", 1.0),
    (Domain.MULTIAGENT, "roles", "roles", 1.0),
    (Domain.MULTIAGENT, "scaling", "small_world", 1.0),
    (Domain.CROSS, "diffusion_language_rl", "joint", 0.5),
    (Domain.CROSS, "diffusion_policy_rl", "simplex", 0.5),
    (Domain.CROSS, "multiagent_rl", "debate", 0.5),
    (Domain.CROSS, "moe_multiagent", "routing", 0.5),
    (Domain.TRANSFORMER, "learned_decoder", "linear", 0.2),
    (Domain.DIFFUSION, "learned_score", "mlp", 0.5),
    (Domain.RL, "learned_policy", "linear", 0.2),
    (Domain.MULTIAGENT, "learned_agents", "independent", 0.2),
    (Domain.CROSS, "assumption_pairs", "transformer", 0.4),
    (Domain.CROSS, "renormalized_bridge", "diffusion", 0.5),
    (Domain.CROSS, "critical_window", "reinforcement", 0.5),
    (Domain.CROSS, "outcome_atlas", "stable", 0.5),
]


@pytest.mark.parametrize(("domain", "family", "variant", "control"), CASES)
def test_every_registered_program_has_finite_common_contract(domain, family, variant, control):
    task = TaskSpec(
        study="smoke",
        domain=domain,
        family=family,
        variant=variant,
        stage="test",
        control_name="control",
        control=control,
        size=8,
        seed=11,
        parameters={
            "n_probe": 24,
            "worlds": 24,
            "replicas": 24,
            "steps": 20,
            "tokens": 32,
            "horizon": 16,
            "train_centers": 16,
            "train_examples": 24,
            "sequence_length": 4,
            "image_size": 4,
            "model_width": 8,
            "pair": "data__architecture",
            "transition_kind": "continuous",
        },
    )
    metrics, arrays = resolve_runner(task)(task, torch.device("cpu"))
    assert set(COMMON_METRICS) <= set(metrics)
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert "signed_order_samples" in arrays


def test_m0_is_a_rank_one_exact_anchor():
    task = TaskSpec(
        study="m0",
        domain=Domain.TRANSFORMER,
        family="architecture",
        variant="m0",
        stage="test",
        control_name="sample_exponent",
        control=2.0,
        size=64,
        seed=11,
        parameters={"teacher_rank": 8, "data_stage": "d0"},
    )
    metrics, _ = resolve_runner(task)(task, torch.device("cpu"))
    assert metrics["teacher_rank"] == 1.0
    assert metrics["student_capacity"] == 1.0
