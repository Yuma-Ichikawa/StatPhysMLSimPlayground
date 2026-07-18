"""Tests for the instrumented M0--M8 attention ladder."""

# ruff: noqa: D103

import pytest
import torch

from statphys.atlas.data import PositionalSemanticDataset
from statphys.atlas.models import (
    AttentionLadderConfig,
    InstrumentedAttentionModel,
    build_attention_ladder,
)


def _config_for_stage(stage: str) -> AttentionLadderConfig:
    common = {"stage": stage, "d_model": 8, "seq_len": 5, "init_seed": 17}
    if stage in {"m3", "m4", "m5", "m6"}:
        common.update(n_heads=2, head_dim=4)
    if stage in {"m7", "m8"}:
        common.update(n_heads=2, head_dim=4, n_layers=3)
    return AttentionLadderConfig(**common)


@pytest.mark.parametrize("stage", [f"m{i}" for i in range(9)])
def test_every_ladder_stage_has_unified_forward_and_diagnostics(stage):
    config = _config_for_stage(stage)
    model = build_attention_ladder(config)
    inputs = torch.randn(3, 5, 8)
    output, diagnostics = model(inputs, return_diagnostics=True)

    layers = 3 if stage in {"m7", "m8"} else 1
    heads = 2 if stage in {"m3", "m4", "m5", "m6", "m7", "m8"} else 1
    assert output.shape == (3, 5, 8)
    assert diagnostics["attention_maps"].shape == (3, layers, heads, 5, 5)
    assert diagnostics["representations"].shape == (3, layers + 1, 5, 8)
    assert diagnostics["attention_outputs"].shape == (3, layers, 5, 8)
    assert diagnostics["mlp_outputs"].shape == (3, layers, 5, 8)
    assert diagnostics["effective_qk"].shape == (layers, heads, 8, 8)
    assert diagnostics["effective_ov"].shape == (layers, heads, 8, 8)
    assert torch.allclose(
        diagnostics["attention_maps"].sum(dim=-1),
        torch.ones(3, layers, heads, 5),
    )


def test_m0_semantic_initialization_matches_reference_student_formula():
    dataset = PositionalSemanticDataset.exact_paper_bridge(
        d_model=10,
        omega=0.0,
        teacher_seed=29,
    )
    batch = dataset.sample(6, seed=31)
    model = InstrumentedAttentionModel(AttentionLadderConfig(stage="m0", d_model=10, seq_len=2))
    model.initialize_from_directions(
        dataset.teacher_vectors,
        dataset.positional_encoding,
        strategy="semantic",
    )
    output, diagnostics = model(batch.inputs, return_diagnostics=True)

    projection = torch.einsum("btd,rd->brt", batch.inputs, dataset.teacher_vectors)
    logits = torch.einsum("brt,brs->bts", projection, projection) / 10
    expected_student_attention = torch.softmax(logits, dim=-1)
    expected_student_output = torch.bmm(expected_student_attention, batch.inputs)
    assert torch.allclose(
        diagnostics["attention_maps"][:, 0, 0], expected_student_attention, atol=1e-6
    )
    assert torch.allclose(output, expected_student_output, atol=1e-6)
    # The exact teacher intentionally sees raw noise while the student sees
    # position-shifted inputs, matching the reference implementation.
    assert not torch.equal(batch.inputs, batch.raw_tokens)


def test_m0_is_tied_and_m1_is_structurally_untied():
    m0 = build_attention_ladder(stage="m0", d_model=7, seq_len=2, signal_rank=2)
    m1 = build_attention_ladder(stage="m1", d_model=7, seq_len=2, signal_rank=2)
    m0_names = dict(m0.named_parameters())
    m1_names = dict(m1.named_parameters())
    assert "blocks.0.low_rank_q" in m0_names
    assert "blocks.0.low_rank_k" not in m0_names
    assert "blocks.0.low_rank_q" in m1_names
    assert "blocks.0.low_rank_k" in m1_names
    assert m0.effective_qk_matrices().shape == (1, 1, 7, 7)
    assert torch.allclose(
        m0.effective_qk_matrices(),
        m0.effective_qk_matrices().transpose(-1, -2),
    )


def test_positional_and_mixed_initialization_and_noise_are_deterministic():
    dataset = PositionalSemanticDataset.exact_paper_bridge(d_model=8)
    config = AttentionLadderConfig(stage="m1", d_model=8, seq_len=2, init_seed=41)
    first = build_attention_ladder(config).initialize_from_directions(
        dataset.teacher_vectors,
        dataset.positional_encoding,
        strategy="mixed",
        noise_scale=0.05,
    )
    second = build_attention_ladder(config).initialize_from_directions(
        dataset.teacher_vectors,
        dataset.positional_encoding,
        strategy="mixed",
        noise_scale=0.05,
    )
    assert torch.equal(
        dict(first.named_parameters())["blocks.0.low_rank_q"],
        dict(second.named_parameters())["blocks.0.low_rank_q"],
    )
    assert torch.equal(
        dict(first.named_parameters())["blocks.0.low_rank_k"],
        dict(second.named_parameters())["blocks.0.low_rank_k"],
    )

    positional = build_attention_ladder(config).initialize_from_directions(
        positional_directions=dataset.positional_encoding,
        strategy="positional",
    )
    factor = dict(positional.named_parameters())["blocks.0.low_rank_q"][0, :, 0]
    cosine = torch.nn.functional.cosine_similarity(factor, dataset.positional_encoding[0], dim=0)
    assert cosine.abs().item() == pytest.approx(1.0, abs=1e-6)


def test_directional_initialization_is_rejected_for_full_projection_models():
    model = build_attention_ladder(stage="m2", d_model=8, seq_len=2)
    with pytest.raises(ValueError, match="only for m0 and m1"):
        model.initialize_from_directions(torch.ones(8), strategy="semantic")
    model.initialize_from_directions(strategy="random")


def test_effective_low_rank_ov_is_identity():
    model = build_attention_ladder(stage="m1", d_model=6, seq_len=2)
    expected = torch.eye(6).view(1, 1, 6, 6)
    assert torch.equal(model.effective_ov_matrices(), expected)


def test_attention_and_mlp_branches_can_be_ablated():
    model = build_attention_ladder(
        stage="m5",
        d_model=8,
        seq_len=4,
        n_heads=2,
        head_dim=4,
        norm="pre",
    )
    inputs = torch.randn(2, 4, 8)
    intact = model(inputs)
    no_attention = model(inputs, ablate_attention=True)
    no_mlp = model(inputs, ablate_mlp=True)
    no_branches = model(inputs, ablate_attention=True, ablate_mlp=True)

    assert not torch.allclose(intact, no_attention)
    assert not torch.allclose(intact, no_mlp)
    assert torch.allclose(no_branches, inputs)


@pytest.mark.parametrize("stage", ["m6", "m7", "m8"])
def test_causal_rope_stages_have_zero_future_attention(stage):
    kwargs = {
        "stage": stage,
        "d_model": 8,
        "seq_len": 6,
        "n_heads": 2,
        "head_dim": 4,
    }
    if stage in {"m7", "m8"}:
        kwargs["n_layers"] = 2
    model = build_attention_ladder(**kwargs)
    _, diagnostics = model(torch.randn(3, 6, 8), return_diagnostics=True)
    maps = diagnostics["attention_maps"]
    upper = torch.ones(6, 6, dtype=torch.bool).triu(1)
    assert torch.equal(maps[..., upper], torch.zeros_like(maps[..., upper]))


def test_causal_rope_supports_odd_and_scalar_head_dimensions():
    for head_dim in (1, 3):
        model = build_attention_ladder(
            stage="m6",
            d_model=6,
            seq_len=4,
            n_heads=1,
            head_dim=head_dim,
        )
        output, diagnostics = model(torch.randn(2, 4, 6), return_diagnostics=True)
        assert output.shape == (2, 4, 6)
        upper = torch.ones(4, 4, dtype=torch.bool).triu(1)
        maps = diagnostics["attention_maps"]
        assert torch.equal(maps[..., upper], torch.zeros_like(maps[..., upper]))


def test_m8_readout_is_causal_and_shift_helper_aligns_next_tokens():
    config = AttentionLadderConfig(
        stage="m8",
        d_model=8,
        seq_len=7,
        n_heads=2,
        head_dim=4,
        n_layers=2,
    )
    assert config.is_autoregressive
    assert config.causal_target_shift == 1
    model = build_attention_ladder(config)
    inputs = torch.randn(2, 7, 8)
    changed_future = inputs.clone()
    changed_future[:, 4:] += 100 * torch.randn_like(changed_future[:, 4:])
    first = model(inputs)
    second = model(changed_future)
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6)
    assert model.autoregressive_readout is not None

    shifted_predictions, shifted_targets = model.shifted_autoregressive_pairs(first, inputs)
    assert torch.equal(shifted_predictions, first[:, :-1])
    assert torch.equal(shifted_targets, inputs[:, 1:])


def test_autoregressive_shift_helper_is_m8_only():
    model = build_attention_ladder(
        stage="m7", d_model=8, seq_len=3, n_heads=2, head_dim=4, n_layers=2
    )
    assert not model.config.is_autoregressive
    assert model.config.causal_target_shift == 0
    with pytest.raises(RuntimeError, match="only for m8"):
        model.shifted_autoregressive_pairs(torch.randn(1, 3, 8), torch.randn(1, 3, 8))


def test_pre_and_post_norm_are_both_supported():
    inputs = torch.randn(2, 3, 8)
    for norm in ("pre", "post"):
        model = build_attention_ladder(
            stage="m4",
            d_model=8,
            seq_len=3,
            n_heads=2,
            head_dim=4,
            norm=norm,
        )
        assert model(inputs).shape == inputs.shape


def test_dtype_and_gradients_are_preserved():
    model = build_attention_ladder(
        stage="m5",
        d_model=6,
        seq_len=3,
        n_heads=2,
        head_dim=3,
        dtype=torch.float64,
    )
    inputs = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    output = model(inputs)
    output.square().mean().backward()
    assert output.dtype == torch.float64
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_seed_is_deterministic_and_does_not_change_global_rng():
    config = AttentionLadderConfig(
        stage="m3",
        d_model=8,
        seq_len=3,
        n_heads=2,
        head_dim=4,
        init_seed=101,
    )
    torch.manual_seed(77)
    state_before = torch.random.get_rng_state().clone()
    first = build_attention_ladder(config)
    state_after = torch.random.get_rng_state()
    second = build_attention_ladder(config)

    assert torch.equal(state_before, state_after)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=False
    ):
        assert torch.equal(first_parameter, second_parameter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "m9", "d_model": 8, "seq_len": 2},
        {"stage": "m0", "d_model": 8, "seq_len": 2, "n_heads": 2},
        {"stage": "m5", "d_model": 8, "seq_len": 2, "n_layers": 2},
        {"stage": "m6", "d_model": 8, "seq_len": 2, "attention_temperature": 0.0},
        {"stage": "m4", "d_model": 8, "seq_len": 2, "norm": "invalid"},
    ],
)
def test_invalid_model_configuration_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        AttentionLadderConfig(**kwargs)


def test_input_validation_is_clear():
    model = build_attention_ladder(stage="m0", d_model=4, seq_len=2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 4))
    with pytest.raises(ValueError, match="final dimension"):
        model(torch.randn(2, 2, 5))
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.randn(2, 3, 4))
