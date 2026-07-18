"""Numerical contract tests for atlas observables."""

from __future__ import annotations

import numpy as np
import pytest

from statphys.atlas.observables import (
    attention_geometry,
    centered_functional_overlap,
    correlation_length,
    effective_ov_matrix,
    effective_qk_matrix,
    functional_replica_overlaps,
    head_specialization_metrics,
    intervention_loss_deltas,
    latent_overlap_matrix,
    match_heads,
    matrix_spectrum,
    permutation_invariant_head_spectrum,
    position_correlation,
    representation_statistics,
    subspace_principal_angles,
    two_template_decomposition,
)


def test_centered_overlap_is_bias_and_scale_invariant() -> None:
    template = np.array([-2.0, -1.0, 1.0, 2.0])
    assert centered_functional_overlap(7.0 + 3.0 * template, template) == pytest.approx(1.0)
    assert centered_functional_overlap(7.0 - 3.0 * template, template) == pytest.approx(-1.0)
    assert np.isnan(centered_functional_overlap(np.ones(4), template))
    assert centered_functional_overlap(np.ones(4), template, zero_variance="zero") == 0.0


def test_two_template_decomposition_recovers_coexistence() -> None:
    positional = np.array([-1.0, 1.0, -1.0, 1.0])
    semantic = np.array([-1.0, -1.0, 1.0, 1.0])
    output = 4.0 + 0.6 * positional / np.linalg.norm(positional)
    output += 0.8 * semantic / np.linalg.norm(semantic)
    result = two_template_decomposition(output, positional, semantic)
    assert result["m_pos"] == pytest.approx(0.6)
    assert result["m_sem"] == pytest.approx(0.8)
    assert result["coexistence_strength"] == pytest.approx(0.6)
    assert result["coexistence_balance"] == pytest.approx(6.0 / 7.0)
    assert result["r2"] == pytest.approx(1.0)
    assert result["residual_fraction"] == pytest.approx(0.0, abs=1e-14)
    assert result["rank"] == 2


def test_two_template_decomposition_reports_degeneracy_and_collinearity() -> None:
    positional = np.array([-1.0, 0.0, 1.0, 2.0])
    semantic = -2.0 * positional + 3.0
    result = two_template_decomposition(np.ones(4), positional, semantic)
    assert result["degenerate_output"]
    assert result["rank"] == 1
    assert np.isinf(result["condition_number"])
    with pytest.raises(ValueError, match="template is constant"):
        two_template_decomposition(positional, np.ones(4), semantic)


def test_head_matching_and_permutation_invariant_spectrum() -> None:
    heads = np.eye(3)
    latents = np.eye(3)[[2, 0, 1]]
    overlap = latent_overlap_matrix(heads, latents)
    assignment = match_heads(overlap, method="exact")
    assert assignment["method"] == "exact_permutation"
    assert assignment["is_exact"]
    np.testing.assert_allclose(assignment["matched_scores"], 1.0)

    spectrum = permutation_invariant_head_spectrum(overlap)
    permuted = permutation_invariant_head_spectrum(overlap[[1, 2, 0]][:, [2, 0, 1]])
    np.testing.assert_allclose(spectrum["singular_values"], permuted["singular_values"])
    assert spectrum["effective_rank"] == pytest.approx(3.0)


def test_greedy_fallback_is_explicit_and_specialization_flags_heads() -> None:
    overlap = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    greedy = match_heads(overlap, method="greedy")
    assert greedy["method"] == "greedy_fallback"
    assert not greedy["is_exact"]
    metrics = head_specialization_metrics(overlap, assignment_method="exact")
    np.testing.assert_array_equal(metrics["dead_heads"], [False, False, False, True])
    np.testing.assert_array_equal(metrics["redundant_heads"], [True, True, False, False])
    assert metrics["dead_head_fraction"] == pytest.approx(0.25)
    assert metrics["specialization_strength"] > 0
    assert metrics["normalized_specialization_entropy"] == pytest.approx(0.0, abs=1e-10)


def test_uniform_and_local_attention_geometry() -> None:
    uniform = attention_geometry(np.ones((2, 4, 4)))
    assert uniform["entropy"] == pytest.approx(np.log(4.0))
    assert uniform["normalized_entropy"] == pytest.approx(1.0)
    assert uniform["effective_support"] == pytest.approx(4.0)
    assert uniform["sink_mass"] == pytest.approx(0.25)
    assert uniform["diagonal_mass"] == pytest.approx(0.25)
    assert uniform["previous_token_mass"] == pytest.approx(3.0 / 16.0)

    previous = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    local = attention_geometry(previous)
    assert local["entropy"] == pytest.approx(0.0)
    assert local["sink_mass"] == pytest.approx(2.0 / 3.0)
    assert local["diagonal_mass"] == pytest.approx(1.0 / 3.0)
    assert local["previous_token_mass"] == pytest.approx(2.0 / 3.0)
    assert local["span"] == pytest.approx(2.0 / 3.0)


def test_attention_geometry_excludes_empty_rows_and_rejects_negative_weights() -> None:
    result = attention_geometry(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert result["valid_query_fraction"] == pytest.approx(0.5)
    assert result["diagonal_mass"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        attention_geometry(np.array([[1.0, -0.1]]))


def test_matrix_and_effective_operator_spectra() -> None:
    result = matrix_spectrum(np.diag([3.0, 1.0]), explained_rank=1)
    np.testing.assert_allclose(result["singular_values"], [3.0, 1.0])
    assert result["stable_rank"] == pytest.approx(10.0 / 9.0)
    assert result["outlier_gap"] == pytest.approx(2.0)
    assert result["explained_fraction"] == pytest.approx(0.9)

    query = np.array([[1.0, 2.0], [0.0, 1.0]])
    key = np.array([[2.0, 0.0], [1.0, 3.0]])
    np.testing.assert_allclose(effective_qk_matrix(query, key), query.T @ key)
    value = np.array([[1.0, 2.0], [3.0, 4.0]])
    output = np.array([[2.0, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(effective_ov_matrix(value, output), output @ value)


def test_teacher_subspace_principal_angles() -> None:
    student = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    teacher = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = subspace_principal_angles(student, teacher)
    np.testing.assert_allclose(result["cosines"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result["angles_degrees"], [0.0, 90.0], atol=1e-6)
    assert result["mean_squared_cosine"] == pytest.approx(0.5)


def test_representation_rank_anisotropy_and_correlation_length() -> None:
    isotropic = np.vstack((np.eye(4), -np.eye(4)))
    stats = representation_statistics(isotropic)
    assert stats["participation_ratio"] == pytest.approx(4.0)
    assert stats["effective_rank"] == pytest.approx(4.0)
    assert stats["anisotropy"] == pytest.approx(1.0)

    values = np.linspace(-2.0, 2.0, 9)
    rank_one = values[:, None] * np.array([[1.0, 2.0, -1.0]])
    rank_one_stats = representation_statistics(rank_one)
    assert rank_one_stats["participation_ratio"] == pytest.approx(1.0)
    assert rank_one_stats["anisotropy"] == pytest.approx(3.0)

    correlation = position_correlation(np.array([[1.0], [0.5], [0.25]]), connected=False)
    assert correlation[0] == pytest.approx(1.0)
    exact_curve = np.exp(-np.arange(8) / 2.0)
    assert correlation_length(exact_curve, method="exponential") == pytest.approx(2.0)
    assert correlation_length(exact_curve, method="integral") > 1.0


def test_intervention_deltas_and_functional_replica_distribution() -> None:
    deltas = intervention_loss_deltas(2.0, {"head_0": 2.5, "head_1": 1.5})
    assert deltas["interventions"]["head_0"]["delta"] == pytest.approx(0.5)
    assert deltas["interventions"]["head_0"]["relative_delta"] == pytest.approx(0.25)
    assert deltas["interventions"]["head_1"]["delta"] == pytest.approx(-0.5)

    base = np.array([-1.0, 0.0, 1.0])
    replicas = functional_replica_overlaps(np.stack((base, base + 9.0, -base)), bins=4)
    np.testing.assert_allclose(replicas["overlap_matrix"], [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    np.testing.assert_allclose(np.sort(replicas["pair_overlaps"]), [-1.0, -1.0, 1.0])
    assert replicas["histogram"]["counts"].sum() == 3

