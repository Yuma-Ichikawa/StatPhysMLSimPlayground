"""
Tests for the "realistic settings" extensions:
multi-index models, Gaussian-mixture classification (with an exact Bayes
error check), lazy-vs-rich weight movement, and LoRA-style fine-tuning.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from statphys.experiment import (
    GaussianMixtureDataset,
    Teacher,
    TeacherStudentExperiment,
    bayes_error,
    generalization_error_decomposition,
    subspace_overlap,
    vector_overlap,
)
from statphys.experiment.presets import lora_finetune, mixture_classification, multi_index_model


class TestVectorAndSubspaceOverlap:
    """Basic algebraic properties of the new order parameters."""

    def test_vector_overlap_identical_is_one(self):
        w = torch.randn(50)
        assert vector_overlap(w, w) == pytest.approx(1.0, abs=1e-6)

    def test_vector_overlap_orthogonal_is_zero(self):
        w = torch.tensor([1.0, 0.0, 0.0])
        v = torch.tensor([0.0, 1.0, 0.0])
        assert vector_overlap(w, v) == pytest.approx(0.0, abs=1e-6)

    def test_subspace_overlap_identical_subspace_is_one(self):
        torch.manual_seed(0)
        W = torch.randn(3, 40)
        out = subspace_overlap(W, W)
        assert out["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert out["top_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert len(out["cosines"]) == 3

    def test_subspace_overlap_orthogonal_subspaces_near_zero(self):
        torch.manual_seed(0)
        d = 200
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        Wt = Q[:2].contiguous()
        Ws = Q[2:4].contiguous()
        out = subspace_overlap(Ws, Wt)
        assert out["mean_cosine"] < 0.1

    def test_subspace_overlap_handles_mismatched_widths(self):
        torch.manual_seed(0)
        Wt = torch.randn(2, 30)
        Ws = torch.randn(5, 30)
        out = subspace_overlap(Ws, Wt)
        assert len(out["cosines"]) == 2  # min(K_s, K_t)


class TestGeneralizationErrorDecomposition:
    """The eps_g = 1/2 (rho_f + q_f - 2 m_f) identity must hold exactly."""

    def test_identity_holds(self):
        torch.manual_seed(0)
        teacher = Teacher(nn.Linear(20, 1, bias=False), init="normal", noise_std=0.1)
        student = nn.Linear(20, 1, bias=False)
        X = torch.randn(2048, 20)
        out = generalization_error_decomposition(student, teacher, X)
        assert out["residual"] == pytest.approx(0.0, abs=1e-5)

    def test_perfect_student_has_zero_clean_error(self):
        torch.manual_seed(1)
        teacher = Teacher(nn.Linear(15, 1, bias=False), init="normal")
        X = torch.randn(1024, 15)
        out = generalization_error_decomposition(teacher.model, teacher, X)
        assert out["eps_g_direct"] == pytest.approx(0.0, abs=1e-6)


class TestGaussianMixtureBayesError:
    """Verify the numerically measured 0-1 error against the exact formula."""

    def test_bayes_error_monotonic_in_mu(self):
        errs = [bayes_error(mu) for mu in (0.0, 0.5, 1.0, 2.0, 4.0)]
        assert errs == sorted(errs, reverse=True)
        assert errs[0] == pytest.approx(0.5, abs=1e-6)

    def test_bayes_classifier_matches_analytic_formula(self):
        torch.manual_seed(0)
        d, mu, n = 64, 1.5, 200_000
        ds = GaussianMixtureDataset(d=d, mu=mu)
        X, y = ds.sample(n)
        pred = torch.sign(X @ ds.v)
        empirical_error = (pred != y).float().mean().item()
        assert empirical_error == pytest.approx(bayes_error(mu), abs=0.01)

    def test_suboptimal_direction_matches_formula_with_cosine(self):
        torch.manual_seed(0)
        d, mu, n = 64, 2.0, 200_000
        ds = GaussianMixtureDataset(d=d, mu=mu)
        w = torch.randn(d)
        w = w / w.norm()
        cos = (w @ ds.v).item()
        X, y = ds.sample(n)
        pred = torch.sign(X @ w)
        empirical_error = (pred != y).float().mean().item()
        assert empirical_error == pytest.approx(bayes_error(mu, cos), abs=0.01)

    def test_mixture_classification_preset_runs_and_tracks_overlap(self):
        torch.manual_seed(0)
        d = 48
        exp = mixture_classification(d=d, mu=2.0)
        res = exp.run_order_parameters(
            alphas=[1.0, 8.0], n_replicas=2, lr=5e-2, max_epochs=150, verbose=False
        )
        assert "cluster_overlap" in res.records
        # more data -> better recovery of the cluster axis
        overlap = res.mean("cluster_overlap")
        assert abs(overlap[1]) > abs(overlap[0]) - 0.2


class TestMultiIndexModel:
    """Multi-index model preset and subspace-overlap recovery."""

    def test_matched_width_recovers_subspace(self):
        torch.manual_seed(0)
        d = 96
        exp = multi_index_model(d=d, k_teacher=2, k_student=2, noise_std=0.0)
        res = exp.run_order_parameters(
            alphas=[16.0], n_replicas=2, lr=1e-2, max_epochs=800, verbose=False
        )
        assert res.mean("subspace_overlap")[0] > 0.5

    def test_mismatched_width_is_supported(self):
        torch.manual_seed(0)
        d = 64
        exp = multi_index_model(d=d, k_teacher=2, k_student=5, noise_std=0.0)
        res = exp.run_order_parameters(
            alphas=[8.0], n_replicas=2, lr=1e-2, max_epochs=400, verbose=False
        )
        assert "subspace_overlap" in res.records
        assert np.isfinite(res.mean("subspace_overlap")[0])


class TestWeightMovement:
    """init_scale should suppress relative weight movement (lazy regime)."""

    def test_large_init_scale_reduces_movement(self):
        torch.manual_seed(0)
        d, k = 48, 3
        teacher = Teacher(
            nn.Sequential(nn.Linear(d, k), nn.Tanh(), nn.Linear(k, 1)),
            init="normal",
            noise_std=0.01,
        )

        def student_factory():
            return nn.Sequential(nn.Linear(d, k), nn.Tanh(), nn.Linear(k, 1))

        exp = TeacherStudentExperiment(teacher=teacher, student_factory=student_factory, d=d)
        res_small = exp.run_order_parameters(
            alphas=[4.0], n_replicas=2, lr=5e-3, max_epochs=400, init_scale=1.0, verbose=False
        )
        res_large = exp.run_order_parameters(
            alphas=[4.0], n_replicas=2, lr=5e-3, max_epochs=400, init_scale=50.0, verbose=False
        )
        assert res_large.mean("weight_movement")[0] < res_small.mean("weight_movement")[0]


class TestLoRAFinetune:
    """LoRA-style low-rank adapter recovery."""

    def test_adapter_overlap_improves_with_data(self):
        torch.manual_seed(0)
        d, hidden, rank = 48, 8, 2
        exp = lora_finetune(d=d, hidden=hidden, rank_true=rank, rank_student=rank, noise_std=0.0)
        res = exp.run_order_parameters(
            alphas=[0.5, 16.0], n_replicas=2, lr=2e-2, max_epochs=600, verbose=False
        )
        overlap = res.mean("adapter_overlap")
        assert overlap[1] > overlap[0]

    def test_zero_rank_student_gives_zero_delta(self):
        torch.manual_seed(0)
        d, hidden = 32, 6
        exp = lora_finetune(d=d, hidden=hidden, rank_true=1, rank_student=0, noise_std=0.0)
        res = exp.run_order_parameters(
            alphas=[4.0], n_replicas=1, lr=1e-2, max_epochs=100, verbose=False
        )
        assert res.mean("adapter_overlap")[0] == pytest.approx(0.0, abs=1e-6)


class TestDatasetOverride:
    """TeacherStudentExperiment(dataset=...) plugs in a custom generative model."""

    def test_custom_dataset_is_used(self):
        torch.manual_seed(0)
        d = 20
        ds = GaussianMixtureDataset(d=d, mu=1.0)
        teacher = ds.oracle_teacher()
        exp = TeacherStudentExperiment(
            teacher=teacher,
            student_factory=lambda: nn.Linear(d, 1, bias=False),
            d=d,
            dataset=ds,
        )
        assert exp.dataset is ds
        X, y = exp.dataset.sample(10)
        assert X.shape == (10, d)
        assert set(y.tolist()) <= {-1.0, 1.0}
