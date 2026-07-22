"""Focused audit tests for phase-tensor reporting and figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pytest
from scipy.stats import t as student_t

from statphys.phase_tensor.observables import causal_contributions
from statphys.phase_tensor.paper import _select_optimizer_learning_rate, write_phase_tensor_results
from statphys.phase_tensor.plotting import (
    FIGURE_SIZE,
    _causal_effect_figure,
    _coverage_figure,
    _line_figure,
    _phase_splitting_figure,
    _scaling_figure,
    _style,
)
from statphys.phase_tensor.report import (
    _boundary_summary,
    _canonical_causal_metrics,
    _condition_metric_summaries,
    _summary,
    _threshold_estimate,
)
from statphys.phase_tensor.runner import _training_example_budget


def _estimate(mean: float, ci95: float = 0.01) -> dict[str, float]:
    return {"mean": mean, "ci95": ci95}


def _condition(
    family: str,
    variant: str,
    control: float,
    metrics: dict[str, dict[str, float]],
    *,
    size: int = 64,
    **parameters: object,
) -> dict[str, object]:
    return {
        "family": family,
        "variant": variant,
        "control": control,
        "size": size,
        "parameters": parameters,
        "metrics": metrics,
    }


def test_phase_tensor_array_rejects_non_spark_partition(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        ["bash", str(repository / "scripts" / "phase_tensor" / "run-array.sh")],
        env={
            **os.environ,
            "STATPHYS_MANIFEST": str(tmp_path / "manifest.json"),
            "STATPHYS_OUTPUT": str(tmp_path / "output"),
            "STATPHYS_DATA_ROOT": str(tmp_path / "data"),
            "REPO_ROOT": str(repository),
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_PARTITION": "gpu_shared",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing non-Spark execution" in result.stderr


def test_confirmation_paper_macros_cover_every_registered_family(tmp_path: Path) -> None:
    conditions = []
    for variant in ("linear", "gelu", "geglu", "swiglu"):
        conditions.append({
            **_condition(
                "tensor_mlp",
                variant,
                8.0,
                {
                    "semantic_order": _estimate(0.5),
                    "mlp_causal_effect": _estimate(0.1),
                    "mlp_risk_delta": _estimate(0.2),
                    "mlp_ablated_risk": _estimate(0.8),
                },
                size=128,
            ),
            "stage": "heldout_confirmation",
        })
    for variant, rate in (("sgd_m", 0.01), ("adamw", 3e-4), ("muon", 3e-4), ("soap", 3e-4)):
        conditions.append({
            **_condition(
                "tensor_optimizer",
                variant,
                8.0,
                {
                    "normalized_test_risk": _estimate(0.4),
                    "gradient_block_gini": _estimate(0.2),
                },
                size=128,
                normalization="pre_rmsnorm",
                learning_rate=rate,
            ),
            "stage": "heldout_confirmation",
        })
    for exponent in (1.0, 1.5, 2.0):
        conditions.append({
            **_condition(
                "tensor_scaling",
                "data",
                2.0,
                {
                    "normalized_generalization_error": _estimate(0.3),
                    "train_cap_hit": _estimate(0.0, 0.0),
                },
                size=128,
                sample_exponent=exponent,
            ),
            "stage": "heldout_confirmation",
        })
    for corpus in ("tinystories", "simplestories", "fineweb_edu", "dolma"):
        conditions.append({
            **_condition(
                "tensor_realdata",
                corpus,
                2.0,
                {
                    "bits_per_byte": _estimate(2.5),
                    "normalized_generalization_gap": _estimate(0.05),
                    "corpus_split_disjoint": _estimate(1.0, 0.0),
                },
                size=128,
            ),
            "stage": "external_confirmation",
        })
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({
        "tasks": 180,
        "seeds": list(range(12)),
        "boundaries": [],
        "conditions": conditions,
    }))
    output = write_phase_tensor_results(aggregate, tmp_path / "results.tex")
    rendered = output.read_text()
    assert "\\PhaseConfirmSeeds}{12}" in rendered
    assert "\\ConfirmMLPSwiGLUCausalEffect" in rendered
    assert "\\ConfirmSOAPError" in rendered
    assert "\\ConfirmQuadraticDataScalingError" in rendered
    assert "\\ConfirmDolmaBitsPerByte" in rendered


def test_causal_metrics_store_raw_risks_deltas_and_bounded_effects() -> None:
    metrics = causal_contributions(0.999999, 1.4, 0.7, 1.1)
    assert metrics["mlp_ablated_risk"] == pytest.approx(0.7)
    assert metrics["mlp_risk_delta"] == pytest.approx(-0.299999)
    assert metrics["attention_risk_delta"] == pytest.approx(0.400001)
    assert all(
        -1.0 <= value <= 1.0
        for name, value in metrics.items()
        if name.endswith("_effect")
    )
    expected = metrics["attention_risk_delta"] / (
        abs(metrics["attention_risk_delta"]) + metrics["full_risk"] + 1e-12
    )
    assert metrics["attention_causal_effect"] == pytest.approx(expected)
    assert "mlp_causal_contribution" not in metrics


def test_report_summary_supports_five_and_twelve_seeds_with_raw_values() -> None:
    for count in (5, 12):
        seeds = tuple(range(100, 100 + count))
        values = np.arange(count, dtype=float)
        summary = _summary(values, seeds)
        expected = student_t.ppf(0.975, count - 1) * values.std(ddof=1) / np.sqrt(count)
        assert summary["n"] == count
        assert summary["seed_ids"] == list(seeds)
        assert summary["raw_values"] == values.tolist()
        assert summary["ci95"] == pytest.approx(expected)


def test_report_fails_instead_of_dropping_missing_or_nonfinite_metric() -> None:
    seeds = (1, 2, 3, 4, 5)
    members = [{"seed": seed, "metrics": {"risk": 0.2}} for seed in seeds]
    members[-1]["metrics"] = {}
    with pytest.raises(RuntimeError, match="missing"):
        _condition_metric_summaries(members, seeds)
    members[-1]["metrics"] = {"risk": float("inf")}
    with pytest.raises(RuntimeError, match="non-finite"):
        _condition_metric_summaries(members, seeds)


def test_legacy_causal_metrics_are_reconstructed_but_markedly_bounded() -> None:
    canonical = _canonical_causal_metrics({
        "normalized_test_risk": 0.999999,
        "attention_contribution": 40000.0,
        "mlp_causal_contribution": -20000.0,
        "attention_mlp_synergy": 10000.0,
    })
    assert abs(canonical["attention_causal_effect"]) <= 1.0
    assert abs(canonical["mlp_causal_effect"]) <= 1.0
    assert canonical["attention_ablated_risk"] == pytest.approx(
        canonical["full_risk"] + canonical["attention_risk_delta"]
    )


def test_threshold_censoring_never_becomes_a_boundary() -> None:
    crossed = _threshold_estimate(np.array([1.0, 2.0]), np.array([0.2, 0.8]))
    left = _threshold_estimate(np.array([1.0, 2.0]), np.array([0.8, 0.9]))
    right = _threshold_estimate(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    assert crossed["status"] == "interpolated"
    assert left == {"value": 1.0, "status": "left_censored"}
    assert right == {"value": 2.0, "status": "right_censored"}
    estimates = [{"seed_id": seed, **crossed} for seed in range(5)]
    assert _boundary_summary(estimates, tuple(range(5))) is not None
    estimates[-1] = {"seed_id": 4, **right}
    assert _boundary_summary(estimates, tuple(range(5))) is None


def test_runner_reports_requested_examples_and_cap_status() -> None:
    requested, actual, cap_hit = _training_example_budget(8.0, 64, 1.5, 1000)
    assert requested > actual
    assert actual == 1000
    assert cap_hit is True
    assert _training_example_budget(1.0, 32, 1.0, 1000) == (32, 32, False)


def test_optimizer_lr_uses_disjoint_deterministic_calibration_slice() -> None:
    candidates = []
    for size, control in ((64, 1.0), (128, 4.0)):
        for rate in (0.01, 0.1):
            calibration_risk = 0.2 if rate == 0.1 else 0.5
            endpoint_risk = 0.1 if rate == 0.01 else 0.8
            candidates.append(_condition(
                "tensor_optimizer", "adamw", control,
                {"normalized_test_risk": _estimate(calibration_risk if size == 64 else endpoint_risk)},
                size=size, learning_rate=rate, normalization="pre_rmsnorm",
            ))
    rate, size, control, status = _select_optimizer_learning_rate(candidates)
    assert rate == pytest.approx(0.1)
    assert (size, control) == (64, 1.0)
    assert "calibrated" in status


def test_plot_contract_bounded_censoring_scaling_and_paired_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def capture(figure: Figure, destination: str | Path, *args: object, **kwargs: object) -> None:
        captured[Path(destination).name] = {
            "size": tuple(figure.get_size_inches()),
            "face": figure.get_facecolor(),
            "axes": list(figure.axes),
        }

    monkeypatch.setattr(Figure, "savefig", capture)
    _style()
    _line_figure(
        {"gelu": (np.array([1.0, 2.0]), np.array([0.4, 0.3]), np.array([0.02, 0.02]))},
        tmp_path / "line.pdf", r"$x$", r"$y$",
    )

    causal_conditions = [
        _condition(
            "tensor_mlp", variant, control,
            {
                "attention_causal_effect": _estimate(0.2),
                "mlp_causal_effect": _estimate(-0.3),
            },
            data_kind="synthetic_retrieval",
        )
        for variant in ("gelu", "swiglu") for control in (1.0, 2.0)
    ]
    _causal_effect_figure(causal_conditions, tmp_path / "causal.pdf")
    legacy = [_condition(
        "tensor_mlp", "gelu", 1.0,
        {"mlp_causal_contribution": _estimate(1e6)},
        data_kind="synthetic_retrieval",
    )]
    with pytest.warns(RuntimeWarning, match="deprecated unstable"):
        assert _causal_effect_figure(legacy, tmp_path / "legacy.pdf") is None

    scaling = []
    xmetrics = {
        "width": "model_width",
        "depth": "model_depth",
        "context": "context_length",
        "data": "train_examples",
    }
    for variant, xmetric in xmetrics.items():
        for x in (32.0, 64.0):
            scaling.append(_condition(
                "tensor_scaling", variant, 4.0,
                {xmetric: _estimate(x, 0.0), "normalized_test_risk": _estimate(1.0 / x)},
                size=int(x), sample_exponent=1.0,
            ))
    _scaling_figure(scaling, tmp_path / "scaling.pdf")

    phase = causal_conditions[:2]
    aggregate = {
        "boundaries": [],
        "thresholds": [{
            "family": "tensor_mlp", "size": 64, "parameters": {"data_kind": "synthetic_retrieval"},
            "estimates": [{"seed_id": 0, "value": 2.0, "status": "right_censored"}],
        }],
    }
    for item in phase:
        item["metrics"]["semantic_order"] = _estimate(0.4)
    _phase_splitting_figure(aggregate, phase, tmp_path / "phase.pdf")

    taxonomy = Path(__file__).parents[1] / "experiments" / "phase_continuation" / "phase_tensor_taxonomy.toml"
    _coverage_figure(taxonomy, tmp_path / "coverage.pdf")

    assert captured
    for details in captured.values():
        assert details["size"] == pytest.approx(FIGURE_SIZE)
        assert details["face"] == pytest.approx((1.0, 1.0, 1.0, 1.0))
        for ax in details["axes"]:
            if not ax.get_visible() or not hasattr(ax, "get_xgridlines"):
                continue
            visible_grid = [line for line in ax.get_xgridlines() + ax.get_ygridlines() if line.get_visible()]
            assert visible_grid
            assert all(line.get_linestyle() == "--" for line in visible_grid)

    causal_axes = captured["causal.pdf"]["axes"]
    assert all(ax.get_ylim() == pytest.approx((-1.02, 1.02)) for ax in causal_axes)
    scaling_labels = {ax.get_xlabel() for ax in captured["scaling.pdf"]["axes"]}
    assert scaling_labels == {
        r"model width $d_{\rm model}$", r"depth $L$",
        r"context length $T_{\rm ctx}$", r"training examples $N_{\rm train}$",
    }
    assert all("N_{\\rm path}" not in label for label in scaling_labels)
    phase_labels = captured["phase.pdf"]["axes"][0].get_legend_handles_labels()[1]
    assert "right-censored threshold" in phase_labels
    assert "identified threshold" not in phase_labels
    coverage_titles = {ax.get_title() for ax in captured["coverage.pdf"]["axes"] if ax.get_title()}
    assert coverage_titles == {"Field frontier", "This paper"}
