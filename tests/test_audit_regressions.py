"""High-value regression tests for reproducibility and scientific semantics."""

from __future__ import annotations

import json

import pytest
import torch

from statphys.atlas.bridge_training import train_bridge
from statphys.atlas.schema import OptimizerName, TrainingSpec
from statphys.atlas.training import train_supervised
from statphys.core import ObservableSpec, SeedStreams
from statphys.model import (
    LinearSelfAttention,
    RandomFeaturesModel,
    SoftmaxRegression,
    StateSpaceModel,
)
from statphys.predictive.pipeline import audit_aggregate
from statphys.theory import TheoryResult, TheoryStatus, TheoryType
from statphys.ui import render_report, validate_study, write_study_template


def _training_spec(**overrides: object) -> TrainingSpec:
    values: dict[str, object] = {
        "optimizer": OptimizerName.SGD,
        "learning_rate": 0.01,
        "max_steps": 4,
        "min_steps": 0,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "patience": 2,
    }
    values.update(overrides)
    return TrainingSpec(**values)


def test_best_loss_is_finite_after_first_evaluation_and_model_is_restored() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    inputs = torch.ones(8, 1)
    targets = torch.zeros(8, 1)
    result = train_supervised(model, inputs, targets, _training_spec(), seed=7)
    assert result.best_loss < float("inf")
    assert result.best_step >= 0
    assert result.final_loss == pytest.approx(result.best_loss)
    assert result.restored_best_checkpoint
    assert result.stop_reason in {"patience", "budget_exhausted", "converged"}


def test_bridge_best_loss_is_finite_after_step_zero() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    inputs = torch.zeros(4, 2)
    targets = torch.zeros(4, 2)
    result = train_bridge(model, inputs, targets, _training_spec(), seed=3)
    assert result.best_loss == pytest.approx(0.0)
    assert result.final_loss == pytest.approx(0.0)


def test_explicit_regularizer_cannot_be_combined_with_weight_decay() -> None:
    with pytest.raises(ValueError, match="regularization twice"):
        train_supervised(
            torch.nn.Linear(1, 1),
            torch.ones(2, 1),
            torch.zeros(2, 1),
            _training_spec(weight_decay=0.1),
            seed=0,
            l2_coefficient=0.1,
        )


def test_icl_prediction_changes_when_context_labels_are_permuted() -> None:
    model = LinearSelfAttention(d=2, d_model=2, init_scale=1.0)
    with torch.no_grad():
        model.W_q.copy_(torch.eye(2))
        model.W_k.copy_(torch.eye(2))
        model.W_v.copy_(torch.eye(2))
        model.W_o.copy_(torch.eye(2))
        model.context_label_embedding.copy_(torch.tensor([1.0, 0.5]))
        model.query_marker_embedding.copy_(torch.tensor([0.0, 1.0]))
    context_x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    query_x = torch.tensor([[1.0, 1.0]])
    first = model.forward_icl(context_x, torch.tensor([[1.0, -1.0]]), query_x)
    second = model.forward_icl(context_x, torch.tensor([[-1.0, 1.0]]), query_x)
    assert not torch.allclose(first, second)


def test_forward_variance_is_stable_across_widths() -> None:
    torch.manual_seed(1)
    logits = []
    for dimension in (32, 128):
        model = SoftmaxRegression(dimension, n_classes=3)
        logits.append(float(model(torch.randn(2048, dimension), return_logits=True).std()))
    assert min(logits) > 0.25
    assert max(logits) / min(logits) < 2.0

    outputs = []
    for width in (64, 256):
        model = RandomFeaturesModel(32, p=width)
        outputs.append(float(model(torch.randn(2048, 32)).std()))
    assert min(outputs) > 0.05
    assert max(outputs) / min(outputs) < 2.0


def test_parallel_ssm_matches_recurrent_ssm() -> None:
    torch.manual_seed(2)
    model = StateSpaceModel(d=3, state_dim=4, diagonal_A=False)
    inputs = torch.randn(5, 7, 3)
    torch.testing.assert_close(model(inputs), model.forward_parallel(inputs), rtol=1e-5, atol=1e-6)


def test_named_rng_streams_are_independent_of_evaluation() -> None:
    streams = SeedStreams(17)
    first_train = torch.randn(12, generator=streams.torch("training"))
    evaluation = torch.randn(1000, generator=streams.torch("evaluation"))
    second_train = torch.randn(12, generator=streams.torch("training"))
    assert evaluation.numel() == 1000
    torch.testing.assert_close(first_train, second_train)


def test_observable_contract_rejects_trajectory_phase_statistics() -> None:
    with pytest.raises(ValueError, match="independent replicas"):
        ObservableSpec(
            name="binder_cumulant",
            units="dimensionless",
            ensemble="trajectory",
            interpretation="phase evidence",
        )


def test_theory_status_round_trips_with_validity_metadata() -> None:
    result = TheoryResult(
        theory_type=TheoryType.REPLICA,
        order_params={"m": [0.1]},
        param_values=[1.0],
        converged=[True],
        iterations=[4],
        status=TheoryStatus.ASYMPTOTIC,
        validity={"data": "iid Gaussian", "limit": "d -> infinity"},
        residual=[1e-9],
    )
    restored = TheoryResult.from_dict(result.to_dict())
    assert restored.status is TheoryStatus.ASYMPTOTIC
    assert restored.validity["data"] == "iid Gaussian"


def test_predictive_audit_uses_conditions_and_report_omits_local_paths(tmp_path) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "conditions": [
                    {
                        "domain": "transformer",
                        "variant": "anchor",
                        "size": 32,
                        "control": 1.0,
                        "metrics": {"semantic_order": {"mean": 0.4, "ci95": 0.1}},
                    },
                    {
                        "domain": "transformer",
                        "variant": "anchor",
                        "size": 32,
                        "control": 2.0,
                        "metrics": {"semantic_order": {"mean": 0.7, "ci95": 0.12}},
                    },
                ],
            }
        )
    )
    audit = audit_aggregate(aggregate, tmp_path / "audit.json")
    assert audit["conditions"] == 2
    assert "source" not in audit
    report = render_report(aggregate, tmp_path / "report.html")
    content = report.read_text(encoding="utf-8")
    assert "file://" not in content
    assert "localhost" not in content


def test_report_normalizes_seed_records_to_an_uncertainty_aware_curve(tmp_path) -> None:
    artifact = tmp_path / "seed_records.json"
    artifact.write_text(
        json.dumps(
            {
                "result": {
                    "x_values": [0.5, 1.0],
                    "records": {"m_hat": [[0.2, 0.6], [0.4, 0.8]]},
                }
            }
        )
    )
    report = render_report(artifact, tmp_path / "seed_records.html")
    content = report.read_text(encoding="utf-8")
    assert "<polyline" in content
    assert "m_hat" in content


def test_guided_study_template_validates(tmp_path) -> None:
    study = write_study_template(tmp_path / "study.toml")
    validation = validate_study(study)
    assert validation["valid"]
    assert validation["allowed_wording"] == "response shift"
