from __future__ import annotations

import pytest

from statphys.core import (
    DisorderRole,
    DisorderSpec,
    Estimate,
    EvidenceEngine,
    ExecutionSpec,
    Fidelity,
    ProtocolSpec,
    ScaleSpec,
)


def test_scale_spec_requires_an_explicit_populated_finite_size_coordinate() -> None:
    scale = ScaleSpec(
        width=64,
        depth=4,
        finite_size_coordinate="width",
        scaling_path={"depth": "fixed"},
    )
    assert scale.finite_size_value == 64.0
    assert scale.to_dict()["scaling_path"] == {"depth": "fixed"}
    with pytest.raises(ValueError, match="finite_size_coordinate"):
        ScaleSpec(width=64, finite_size_coordinate="population")


def test_estimate_cannot_hide_outer_seed_count_or_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="must agree"):
        Estimate(
            mean=1.0,
            interval_low=0.5,
            interval_high=1.5,
            interval_level=0.95,
            uncertainty_method="test",
            n_outer=5,
            n_inner=None,
            outer_seed_ids=(1, 2),
            raw_outer_values=(0.9, 1.1),
            units="dimensionless",
        )


def test_protocol_and_execution_identities_separate_pairing_from_hardware() -> None:
    protocol = ProtocolSpec(
        disorders=(
            DisorderSpec(
                DisorderRole.INITIALIZATION,
                "outer",
                paired_across_controls=True,
            ),
        ),
        outer_seed_ids=(11, 13, 17, 19, 23),
        observables=("signed_order_parameter",),
    )
    first = ExecutionSpec(device_type="gpu", precision="float32", attempt=1)
    retry = ExecutionSpec(device_type="cpu", precision="float32", attempt=2)
    assert protocol.to_dict()["disorders"][0]["paired_across_controls"] is True
    assert first.execution_id != retry.execution_id


def test_evidence_is_a_vector_and_does_not_follow_from_size_count_alone() -> None:
    evidence = EvidenceEngine.assess(
        {
            "observable_registered": True,
            "outer_seeds": 5,
            "n_sizes": 12,
            "finite_size_diagnostic": False,
        }
    )
    assert evidence.finite_size == 0
    assert evidence.replication == 1
    assert evidence.grade != "A"
    assert Fidelity.PHENOMENOLOGICAL_GENERATOR != Fidelity.TRAINABLE_SYNTHETIC
