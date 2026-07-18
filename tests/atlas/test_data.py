"""Tests for the positional--semantic teacher ensembles."""

# ruff: noqa: D103

from dataclasses import replace

import pytest
import torch

from statphys.atlas.data import (
    PositionalSemanticDataConfig,
    PositionalSemanticDataset,
)


def test_exact_paper_bridge_parameters_and_formula():
    config = PositionalSemanticDataConfig.exact_paper_bridge(
        d_model=12,
        omega=0.37,
        teacher_seed=11,
        data_seed=12,
        init_seed=13,
    )
    dataset = PositionalSemanticDataset(config)
    batch = dataset.sample(7, seed=19)

    assert config.seq_len == 2
    assert config.signal_rank == 1
    assert config.sigma == 0.5
    assert torch.equal(
        dataset.positional_matrix,
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
    )
    expected_shift = torch.ones(12) / 12**0.5
    assert torch.allclose(dataset.positional_encoding[0], expected_shift)
    assert torch.allclose(dataset.positional_encoding[1], -expected_shift)
    assert torch.allclose(batch.inputs - batch.raw_tokens, dataset.positional_encoding)

    projection = torch.einsum("btd,rd->brt", batch.raw_tokens, dataset.teacher_vectors)
    logits = torch.einsum("brt,brs->bts", projection, projection) / config.d_model
    expected_semantic = torch.softmax(logits, dim=-1)
    expected_mixed = 0.63 * expected_semantic + 0.37 * dataset.positional_matrix
    expected_targets = torch.bmm(expected_mixed, batch.raw_tokens)

    assert torch.allclose(batch.semantic_attention, expected_semantic)
    assert torch.allclose(batch.mixed_attention, expected_mixed)
    assert torch.allclose(batch.targets, expected_targets)


@pytest.mark.parametrize(
    "ensemble",
    ["d0", "d1", "d2", "d3", "d4", "d5"],
)
def test_all_ensembles_share_instrumented_batch_api(ensemble):
    config = PositionalSemanticDataConfig(
        d_model=10,
        seq_len=4,
        signal_rank=2,
        ensemble=ensemble,
        teacher_seed=3,
        data_seed=5,
        init_seed=7,
    )
    dataset = PositionalSemanticDataset(config)
    batch = dataset.sample(9, seed=23)

    assert batch.inputs.shape == (9, 4, 10)
    assert batch.targets.shape == (9, 4, 10)
    assert batch.raw_tokens.shape == (9, 4, 10)
    assert batch.positional_attention.shape == (9, 4, 4)
    assert batch.semantic_attention.shape == (9, 4, 4)
    assert batch.semantic_component_attentions.shape == (9, 2, 4, 4)
    assert batch.mixed_attention.shape == (9, 4, 4)
    assert torch.isfinite(batch.inputs).all()
    assert torch.isfinite(batch.targets).all()
    assert torch.allclose(batch.semantic_attention.sum(dim=-1), torch.ones(9, 4))
    assert torch.allclose(batch.mixed_attention.sum(dim=-1), torch.ones(9, 4))
    assert (batch.hidden_states is not None) == (ensemble in {"d4", "d5"})
    assert (batch.nonterminal_states is not None) == (ensemble == "d5")


def test_rank_one_component_attention_is_total_semantic_attention():
    dataset = PositionalSemanticDataset.exact_paper_bridge(d_model=9)
    batch = dataset.sample(5, seed=4)
    assert torch.allclose(
        batch.semantic_component_attentions[:, 0],
        batch.semantic_attention,
    )


def test_student_t_is_corrected_to_unit_variance_before_sigma():
    config = PositionalSemanticDataConfig(
        d_model=8,
        seq_len=2,
        ensemble="student_t",
        sigma=0.7,
        student_t_df=6.0,
    )
    raw = PositionalSemanticDataset(config).sample(20_000, seed=17).raw_tokens
    empirical_unit_variance = raw.var(unbiased=False) / config.sigma**2
    assert empirical_unit_variance.item() == pytest.approx(1.0, abs=0.06)


def test_elliptical_covariance_has_unit_average_variance_and_requested_condition():
    config = PositionalSemanticDataConfig(
        d_model=16,
        ensemble="elliptical",
        elliptical_condition_number=12.0,
    )
    dataset = PositionalSemanticDataset(config)
    assert dataset.elliptical_covariance is not None
    eigenvalues = torch.linalg.eigvalsh(dataset.elliptical_covariance)
    assert eigenvalues.mean().item() == pytest.approx(1.0, rel=1e-5)
    assert (eigenvalues[-1] / eigenvalues[0]).item() == pytest.approx(12.0, rel=1e-4)


def test_codebook_is_quenched_and_discrete():
    config = PositionalSemanticDataConfig(
        d_model=6,
        seq_len=3,
        ensemble="codebook",
        codebook_size=7,
        sigma=0.5,
    )
    dataset = PositionalSemanticDataset(config)
    batch = dataset.sample(100, seed=31)
    assert dataset.codebook is not None
    observed = batch.raw_tokens.reshape(-1, config.d_model) / config.sigma
    distances = torch.cdist(observed, dataset.codebook)
    assert torch.allclose(distances.min(dim=-1).values, torch.zeros(observed.shape[0]))


def test_hmm_hidden_states_and_persistence_are_exposed():
    config = PositionalSemanticDataConfig(
        d_model=4,
        seq_len=10,
        ensemble="hmm",
        hmm_states=3,
        hmm_persistence=0.92,
    )
    dataset = PositionalSemanticDataset(config)
    batch = dataset.sample(4_000, seed=37)
    assert batch.hidden_states is not None
    assert batch.hidden_states.min() >= 0
    assert batch.hidden_states.max() < config.hmm_states
    empirical_stay = (batch.hidden_states[:, 1:] == batch.hidden_states[:, :-1]).float().mean()
    expected_stay = config.hmm_persistence + (1 - config.hmm_persistence) / config.hmm_states
    assert empirical_stay.item() == pytest.approx(expected_stay, abs=0.015)


def test_d5_binary_pcfg_exposes_quenched_rules_and_nonterminal_states():
    config = PositionalSemanticDataConfig(
        d_model=6,
        seq_len=9,
        ensemble="grammar",
        grammar_nonterminals=5,
        grammar_rule_concentration=0.7,
        grammar_emission_noise=0.1,
        data_seed=43,
    )
    dataset = PositionalSemanticDataset(config)
    batch = dataset.sample(128, seed=47)
    assert dataset.grammar_rule_probabilities is not None
    assert dataset.grammar_rule_probabilities.shape == (5, 5, 5)
    assert torch.allclose(dataset.grammar_rule_probabilities.sum(dim=(1, 2)), torch.ones(5))
    assert batch.nonterminal_states is not None
    assert torch.equal(batch.nonterminal_states, batch.hidden_states)
    assert batch.nonterminal_states.shape == (128, 9)
    assert batch.nonterminal_states.min() >= 0
    assert batch.nonterminal_states.max() < 5


def test_teacher_data_and_initialization_seeds_are_separate():
    base = PositionalSemanticDataConfig(
        d_model=8,
        seq_len=3,
        ensemble="d1",
        teacher_seed=10,
        data_seed=20,
        init_seed=30,
    )
    teacher_changed = replace(base, teacher_seed=11)
    init_changed = replace(base, init_seed=31)
    data_changed = replace(base, data_seed=21)

    original_dataset = PositionalSemanticDataset(base)
    original = original_dataset.sample(6, seed=99)
    changed_teacher_dataset = PositionalSemanticDataset(teacher_changed)
    changed_teacher = changed_teacher_dataset.sample(6, seed=99)
    changed_init = PositionalSemanticDataset(init_changed).sample(6, seed=99)
    changed_data_dataset = PositionalSemanticDataset(data_changed)

    assert torch.equal(original.raw_tokens, changed_teacher.raw_tokens)
    assert not torch.equal(
        original_dataset.teacher_vectors, changed_teacher_dataset.teacher_vectors
    )
    assert torch.equal(original.inputs, changed_init.inputs)
    assert torch.equal(original.targets, changed_init.targets)
    assert not torch.equal(
        original_dataset.elliptical_covariance,
        changed_data_dataset.elliptical_covariance,
    )


def test_sample_seed_is_stateless_and_batch_can_move_dtype():
    dataset = PositionalSemanticDataset.exact_paper_bridge(d_model=5)
    first = dataset.sample(3, seed=8)
    second = dataset.sample(3, seed=8)
    different = dataset.sample(3, seed=9)
    assert torch.equal(first.inputs, second.inputs)
    assert not torch.equal(first.inputs, different.inputs)

    moved = first.to("cpu", torch.float64)
    assert moved.inputs.dtype == torch.float64
    assert moved.semantic_attention.dtype == torch.float64
    assert moved.metadata == first.metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"d_model": 4, "omega": -0.1},
        {"d_model": 4, "student_t_df": 2.0},
        {"d_model": 4, "hmm_emission_noise": 1.0},
        {"d_model": 4, "ensemble": "not-an-ensemble"},
        {"d_model": 4, "positional_matrix": [[1.0, 0.0], [0.2, 0.2]]},
    ],
)
def test_invalid_data_configuration_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PositionalSemanticDataConfig(**kwargs)
