import math

import pytest
import torch

from statphys.continuation.diffusion import run_diffusion
from statphys.continuation.multiagent import run_multiagent
from statphys.continuation.reinforcement import run_reinforcement
from statphys.continuation.schema import Domain, TaskSpec
from statphys.continuation.transformer import run_transformer


@pytest.mark.parametrize(
    ("domain", "variant", "control", "size", "parameters", "runner"),
    [
        (
            Domain.TRANSFORMER,
            "m0",
            0.3,
            8,
            {"steps": 2, "n_probe": 16, "sample_coefficient": 2},
            run_transformer,
        ),
        (
            Domain.DIFFUSION,
            "topk",
            0.8,
            4,
            {"components": 4, "n_probe": 16, "topk": 2},
            run_diffusion,
        ),
        (
            Domain.RL,
            "proxy",
            2.0,
            8,
            {"actions": 4, "action_groups": 2},
            run_reinforcement,
        ),
        (
            Domain.MULTIAGENT,
            "small_world",
            1.0,
            8,
            {"worlds": 16, "steps": 3, "intervention_steps": 2},
            run_multiagent,
        ),
    ],
)
def test_domain_common_contract(domain, variant, control, size, parameters, runner) -> None:
    task = TaskSpec(
        study="test",
        domain=domain,
        variant=variant,
        stage="confirmatory",
        control_name="control",
        control=control,
        size=size,
        seed=11,
        parameters=parameters,
    )
    metrics, arrays = runner(task, torch.device("cpu"))
    for name in (
        "order_parameter",
        "susceptibility",
        "binder_cumulant",
        "generalization_error",
        "ood_generalization_error",
        "effective_multiplicity",
        "interaction_range",
        "macrostate_entropy",
        "oracle_gap",
        "intervention_response",
    ):
        assert name in metrics
        assert math.isfinite(metrics[name])
    assert "signed_order_samples" in arrays


def test_common_coordinates_handles_machine_precision_collapse() -> None:
    import numpy as np

    from statphys.continuation.domains.common import common_coordinates

    metrics, arrays = common_coordinates(
        np.full(1024, 1.0 - np.finfo(np.float64).eps),
        size=2048,
        generalization_error=0.0,
    )
    assert metrics["macrostate_entropy"] == pytest.approx(0.0)
    assert math.isfinite(metrics["binder_cumulant"])
    assert arrays["signed_order_samples"].shape == (1024,)
