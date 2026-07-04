"""Tests for hidden-manifold inputs, realistic presets, and quick APIs."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from statphys.experiment import Teacher, TeacherStudentDataset, get_preset


@pytest.fixture
def small_teacher():
    """Fixed random linear teacher."""
    torch.manual_seed(0)
    return Teacher(nn.Linear(32, 1, bias=False), init="normal")


class TestHiddenManifold:
    """Hidden-manifold input distribution."""

    def test_shapes_and_boundedness(self, small_teacher):
        ds = TeacherStudentDataset(
            small_teacher,
            d=32,
            input_dist="hidden_manifold",
            input_kwargs={"latent_dim": 4, "nonlinearity": "tanh"},
        )
        X, y = ds.sample(100)
        assert X.shape == (100, 32)
        assert y.shape == (100,)
        assert X.abs().max() <= 1.0  # tanh-bounded

    def test_low_rank_structure(self, small_teacher):
        ds = TeacherStudentDataset(
            small_teacher,
            d=32,
            input_dist="hidden_manifold",
            input_kwargs={"latent_dim": 2, "nonlinearity": "identity"},
        )
        X = ds.sample_inputs(500)
        # identity nonlinearity: inputs live exactly in a 2D subspace
        rank = torch.linalg.matrix_rank(X, tol=1e-4).item()
        assert rank == 2

    def test_custom_feature_map(self, small_teacher):
        F = torch.eye(4, 32)
        ds = TeacherStudentDataset(
            small_teacher,
            d=32,
            input_dist="hidden_manifold",
            input_kwargs={"latent_dim": 4, "feature_map": F, "nonlinearity": "identity"},
        )
        X = ds.sample_inputs(50)
        assert torch.allclose(X[:, 4:], torch.zeros(50, 28))

    def test_config_excludes_feature_map(self, small_teacher):
        ds = TeacherStudentDataset(
            small_teacher,
            d=32,
            input_dist="hidden_manifold",
            input_kwargs={"latent_dim": 4, "feature_map": torch.randn(4, 32)},
        )
        cfg = ds.get_config()
        assert "feature_map" not in cfg["input_kwargs"]

    def test_unknown_nonlinearity_raises(self, small_teacher):
        with pytest.raises(ValueError, match="nonlinearity"):
            TeacherStudentDataset(
                small_teacher,
                d=32,
                input_dist="hidden_manifold",
                input_kwargs={"nonlinearity": "swish"},
            )


class TestRealisticPresets:
    """hidden_manifold and tiny_gpt presets."""

    def test_hidden_manifold_preset_runs(self):
        exp = get_preset("hidden_manifold", d=32, latent_dim=4, hidden=4)
        res = exp.run_sample_complexity(
            alphas=[1.0], n_seeds=1, max_epochs=30, verbose=False, n_test=128
        )
        assert "test_error" in res.records

    def test_tiny_gpt_preset_runs(self):
        exp = get_preset("tiny_gpt", d=32, seq_len=4, d_model=8, n_heads=2, n_blocks=1)
        res = exp.run_order_parameters(
            alphas=[1.0],
            n_replicas=2,
            max_epochs=20,
            n_probe=64,
            verbose=False,
        )
        for key in ("m_hat", "q_ab_mean", "test_error"):
            assert key in res.records


class TestQuickAPIs:
    """quick_order_parameters / quick_phase_diagram."""

    def test_quick_order_parameters(self):
        import statphys

        res = statphys.quick_order_parameters(
            "random_mlp",
            alphas=[1.0, 4.0],
            n_replicas=2,
            plot=False,
            preset_kwargs={"d": 24, "hidden": 4},
            max_epochs=50,
            n_probe=128,
        )
        assert len(res.records["m_hat"]) == 2
        assert not np.isnan(res.mean("q_ab_mean")).any()

    def test_quick_phase_diagram(self):
        import statphys

        res = statphys.quick_phase_diagram(
            "sparse_teacher",
            "sparsity",
            [0.5, 0.9],
            alphas=[1.0, 4.0],
            n_replicas=2,
            plot=False,
            preset_kwargs={"d": 24},
            max_epochs=50,
            n_probe=128,
        )
        assert res.grid("m_hat").shape == (2, 2)

    def test_dashboard_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        import statphys
        from statphys.vis import plot_order_parameter_dashboard

        res = statphys.quick_order_parameters(
            "random_mlp",
            alphas=[1.0, 2.0],
            n_replicas=2,
            plot=False,
            preset_kwargs={"d": 16, "hidden": 2},
            max_epochs=20,
            n_probe=64,
        )
        fig, _ = plot_order_parameter_dashboard(res, title="smoke")
        assert fig is not None
