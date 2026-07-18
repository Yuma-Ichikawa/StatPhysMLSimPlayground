"""Numerical contract tests for atlas phase-analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest

from statphys.atlas.analysis import (
    adjacent_histogram_js,
    adjacent_js_divergence,
    binder_crossing,
    binder_cumulant,
    binder_summary,
    bootstrap_finite_size_scaling,
    change_point_candidates,
    classify_phase,
    classify_transition_evidence,
    data_collapse_score,
    estimate_transition_boundary,
    finite_difference_response,
    fit_finite_size_scaling_grid,
    hierarchical_bootstrap,
    nested_estimator,
    order_parameter_histogram,
    rate_function,
    susceptibility,
)


def test_nested_estimator_respects_replica_hierarchy() -> None:
    values = np.arange(8.0).reshape(2, 2, 2)
    result = nested_estimator(values)
    assert result["estimate"] == pytest.approx(3.5)
    np.testing.assert_allclose(result["teacher_means"], [1.5, 5.5])
    assert result["standard_error"] == pytest.approx(2.0)
    assert result["counts"] == {
        "teacher": 2,
        "dataset_per_teacher": 2,
        "optimizer_per_dataset": 2,
    }


def test_hierarchical_bootstrap_is_seeded_and_contains_point_estimate() -> None:
    values = np.arange(24.0).reshape(3, 2, 4)
    first = hierarchical_bootstrap(values, n_bootstrap=150, seed=17, confidence=0.9)
    second = hierarchical_bootstrap(values, n_bootstrap=150, seed=17, confidence=0.9)
    np.testing.assert_allclose(first["distribution"], second["distribution"])
    assert first["estimate"] == pytest.approx(values.mean())
    lower, upper = first["confidence_interval"]
    assert lower < first["estimate"] < upper


def test_susceptibility_requires_explicit_effective_size() -> None:
    assert susceptibility(np.array([-1.0, 1.0]), n_eff=10.0) == pytest.approx(10.0)
    with pytest.raises(TypeError):
        susceptibility(np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        susceptibility(np.array([-1.0, 1.0]), n_eff=0.0)


def test_raw_centered_and_vector_binder_conventions() -> None:
    spins = np.tile([-1.0, 1.0], 100)
    assert binder_cumulant(spins) == pytest.approx(2.0 / 3.0)
    summary = binder_summary(spins)
    assert summary["raw"] == pytest.approx(2.0 / 3.0)
    assert summary["centered"] == pytest.approx(2.0 / 3.0)

    angles = np.linspace(0.0, 2.0 * np.pi, 1000, endpoint=False)
    fixed_radius = np.column_stack((np.cos(angles), np.sin(angles)))
    assert binder_cumulant(fixed_radius, vector_axis=1) == pytest.approx(0.5)

    gaussian = np.random.default_rng(3).normal(size=100_000)
    assert binder_cumulant(gaussian) == pytest.approx(0.0, abs=0.025)


def test_histogram_and_rate_function_have_consistent_minimum() -> None:
    histogram = order_parameter_histogram(np.array([-1.0, -0.9, 0.9, 1.0]), bins=2)
    assert histogram["counts"].sum() == 4
    rate = rate_function(histogram["density"], n_eff=4)
    assert rate.min() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-negative"):
        rate_function(np.array([0.5, -0.5]))


def test_binder_crossing_response_and_transition_boundary() -> None:
    control = np.array([0.0, 1.0, 2.0])
    crossing = binder_crossing(
        control,
        np.array([16.0, 32.0]),
        np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
    )
    assert crossing["crossing_estimate"] == pytest.approx(1.0)
    assert crossing["n_crossings"] == 1

    dense_control = np.linspace(0.0, 2.0, 101)
    signal = np.tanh(8.0 * (dense_control - 1.0))
    response = finite_difference_response(dense_control, signal)
    assert response["peak_control"] == pytest.approx(1.0, abs=0.03)
    gradient_boundary = estimate_transition_boundary(dense_control, signal)
    assert gradient_boundary["boundary"] == pytest.approx(1.0, abs=0.03)
    threshold_boundary = estimate_transition_boundary(
        dense_control, signal, method="threshold", threshold=0.0
    )
    assert threshold_boundary["boundary"] == pytest.approx(1.0)


def test_phase_and_transition_classifiers_are_conservative() -> None:
    assert classify_phase(0.8, 0.1)["label"] == "positional"
    assert classify_phase(0.5, 0.45)["label"] == "coexistence"
    assert classify_phase(0.1, 0.1)["label"] == "disordered_or_unresolved"

    continuous = classify_transition_evidence(
        n_sizes=5,
        susceptibility_peak_growth=0.7,
        binder_crossing_spread=0.02,
        data_collapse_score=0.03,
    )
    assert continuous["label"] == "continuous_transition_candidate"
    first_order = classify_transition_evidence(
        n_sizes=3, bimodal=True, barrier_growth=True, hysteresis=False
    )
    assert first_order["label"] == "first_order_candidate"
    assert classify_transition_evidence(n_sizes=2)["label"] == "insufficient_evidence"


def _synthetic_scaling_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    control = np.linspace(-0.5, 0.5, 41)
    sizes = np.array([4.0, 9.0, 16.0])
    observable_exponent = 0.5
    inverse_nu = 0.5
    scaled = control[None, :] * sizes[:, None] ** inverse_nu
    curves = sizes[:, None] ** (-observable_exponent) * (1.0 + scaled)
    return control, sizes, curves


def test_data_collapse_and_grid_fit_recover_synthetic_exponents() -> None:
    control, sizes, curves = _synthetic_scaling_curves()
    collapse = data_collapse_score(
        control,
        sizes,
        curves,
        critical_control=0.0,
        observable_exponent=0.5,
        inverse_nu=0.5,
        n_grid=64,
    )
    assert collapse["score"] == pytest.approx(0.0, abs=1e-25)
    fit = fit_finite_size_scaling_grid(
        control,
        sizes,
        curves,
        critical_control_grid=np.array([-0.1, 0.0, 0.1]),
        observable_exponent_grid=np.array([0.0, 0.5, 1.0]),
        inverse_nu_grid=np.array([0.0, 0.5, 1.0]),
        n_grid=64,
    )
    assert fit["critical_control"] == pytest.approx(0.0)
    assert fit["observable_exponent"] == pytest.approx(0.5)
    assert fit["inverse_nu"] == pytest.approx(0.5)


def test_finite_size_scaling_bootstrap_returns_parameter_intervals() -> None:
    control, sizes, curves = _synthetic_scaling_curves()
    replicas = np.repeat(curves[..., None], 4, axis=-1)
    result = bootstrap_finite_size_scaling(
        control,
        sizes,
        replicas,
        critical_control_grid=np.array([-0.05, 0.0, 0.05]),
        observable_exponent_grid=np.array([0.25, 0.5, 0.75]),
        inverse_nu_grid=np.array([0.25, 0.5, 0.75]),
        n_bootstrap=5,
        seed=8,
        n_grid=32,
    )
    assert result["distribution"].shape == (5, 4)
    assert result["confidence_intervals"]["critical_control"] == pytest.approx((0.0, 0.0))
    assert result["fit"]["inverse_nu"] == pytest.approx(0.5)


def test_adjacent_js_histograms_and_change_point_candidates() -> None:
    distributions = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    divergence = adjacent_js_divergence(distributions)
    np.testing.assert_allclose(divergence, [0.0, np.log(2.0)])

    histograms = adjacent_histogram_js(
        [np.array([-1.0, -0.9]), np.array([-1.0, -0.8]), np.array([0.8, 1.0])], bins=4
    )
    assert histograms["divergence"].shape == (2,)
    assert histograms["divergence"][1] > histograms["divergence"][0]

    control = np.arange(6.0)
    observables = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    candidates = change_point_candidates(control, observables, z_threshold=2.0)
    assert candidates["candidates"][0]["control_midpoint"] == pytest.approx(2.5)

