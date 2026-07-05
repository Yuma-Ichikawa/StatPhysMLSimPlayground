"""Tests for statphys.frontier (modern-paradigm teacher-student settings)."""

import numpy as np
import pytest
import torch

from statphys.frontier import (
    bon_kl,
    correlated_teacher,
    make_icl_batch,
    mlp,
    model_overlap,
    output_overlap,
    ridge_predictor,
    run_collapse,
    run_finetune,
    run_icl,
    run_overoptimization,
    run_weak_to_strong,
    train_regression,
)
from statphys.experiment import Teacher


class TestCommon:
    """Shared utilities: training loop, overlaps, correlated teachers."""

    def test_train_regression_fits_linear_target(self):
        """The shared loop should fit an easy linear target well."""
        torch.manual_seed(0)
        d, n = 8, 256
        X = torch.randn(n, d)
        w = torch.randn(d) / d**0.5
        y = X @ w
        model = torch.nn.Linear(d, 1)
        train_regression(model, X, y, lr=1e-2, epochs=800)
        pred = model(X).squeeze(-1)
        assert ((pred - y) ** 2).mean().item() < 1e-2

    def test_output_overlap_bounds(self):
        """Overlap of a vector with itself is 1, with its negation -1."""
        v = torch.randn(100)
        assert output_overlap(v, v) == pytest.approx(1.0, abs=1e-6)
        assert output_overlap(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_model_overlap_teacher_with_itself(self):
        """A teacher measured against itself gives m_hat = 1."""
        torch.manual_seed(0)
        t = Teacher(mlp(16, 4), init="normal")
        X = torch.randn(512, 16)
        assert model_overlap(t.clean, t, X) == pytest.approx(1.0, abs=1e-6)

    def test_correlated_teacher_extremes(self):
        """similarity=1 reproduces the task; similarity=0 decorrelates it."""
        torch.manual_seed(0)
        t_a = Teacher(mlp(32, 8), init="normal")
        X = torch.randn(2048, 32)
        t_same = correlated_teacher(t_a, similarity=1.0, seed=1)
        t_indep = correlated_teacher(t_a, similarity=0.0, seed=1)
        m_same = output_overlap(t_a.clean(X), t_same.clean(X))
        m_indep = output_overlap(t_a.clean(X), t_indep.clean(X))
        assert m_same == pytest.approx(1.0, abs=1e-5)
        assert abs(m_indep) < 0.35

    def test_correlated_teacher_validates_similarity(self):
        """Out-of-range similarity raises."""
        t = Teacher(mlp(8, 2), init="normal")
        with pytest.raises(ValueError):
            correlated_teacher(t, similarity=1.5)


class TestSFT:
    """Fine-tuning / forgetting protocol."""

    def test_run_finetune_shapes_and_learning(self):
        """Trajectories have checkpoint length; task B is learned."""
        res = run_finetune(
            d=24,
            hidden=4,
            similarity=0.9,
            alpha_pre=8.0,
            alpha_ft=8.0,
            n_checkpoints=4,
            epochs_per_checkpoint=100,
            pretrain_epochs=600,
            seed=0,
        )
        assert len(res["m_A"]) == len(res["m_B"]) == len(res["epochs"]) == 5
        assert res["m_B"][-1] > res["m_B"][0] - 0.05  # B improves (or was already high)
        assert np.isfinite(res["forgetting"])

    def test_identical_task_no_forgetting(self):
        """similarity=1: fine-tuning on the same task should not hurt m_A."""
        res = run_finetune(
            d=24,
            hidden=4,
            similarity=1.0,
            alpha_pre=8.0,
            alpha_ft=4.0,
            n_checkpoints=3,
            epochs_per_checkpoint=80,
            pretrain_epochs=600,
            seed=0,
            compare_scratch=False,
        )
        assert res["forgetting"] < 0.15


class TestRLHF:
    """Reward-model overoptimization."""

    def test_bon_kl_formula(self):
        """KL(1) = 0 and KL grows with n."""
        kls = bon_kl(np.array([1, 2, 16]))
        assert kls[0] == pytest.approx(0.0)
        assert np.all(np.diff(kls) > 0)

    def test_gold_reward_penalizes_off_distribution(self):
        """The gold reward is lower far off the base distribution."""
        from statphys.frontier import GoldReward

        torch.manual_seed(0)
        gold = GoldReward(Teacher(mlp(8, 2), init="normal"), hack_penalty=5.0)
        x_in = torch.randn(256, 8)
        x_out = 4.0 * torch.randn(256, 8)
        pen_in = gold(x_in) - gold.teacher.clean(x_in)
        pen_out = gold(x_out) - gold.teacher.clean(x_out)
        assert pen_in.mean() > pen_out.mean()

    def test_run_overoptimization_bon(self):
        """BoN mode: curves align with n_values; proxy reward increases."""
        res = run_overoptimization(
            d=16,
            hidden=4,
            alpha_r=8.0,
            policy="bon",
            n_values=[1, 4, 16, 64],
            n_eval=64,
            epochs=300,
            seed=0,
        )
        assert res["gold"].shape == res["proxy"].shape == (4,)
        # more optimization always increases the *proxy* reward
        assert res["proxy"][-1] > res["proxy"][0]
        assert -1.0 <= res["m_RM"] <= 1.0

    def test_run_overoptimization_shift(self):
        """Shift mode: KL grows as the KL penalty decreases."""
        res = run_overoptimization(
            d=16,
            hidden=4,
            alpha_r=8.0,
            policy="shift",
            kl_coefs=[1.0, 0.05],
            policy_steps=150,
            n_eval=128,
            epochs=200,
            seed=0,
        )
        assert res["kl"].shape == (2,)
        assert res["kl"][1] > res["kl"][0]  # weaker penalty -> larger KL

    def test_unknown_policy_raises(self):
        """Invalid policy names are rejected."""
        with pytest.raises(ValueError):
            run_overoptimization(d=8, hidden=2, policy="ppo", epochs=10, seed=0)


class TestWeakToStrong:
    """Weak supervisor -> strong student chain."""

    def test_run_weak_to_strong_ordering(self):
        """Ceiling >= weak supervisor; all overlaps in [-1, 1]."""
        res = run_weak_to_strong(
            d=24,
            hidden_true=4,
            hidden_weak=1,
            hidden_strong=8,
            alpha_weak=4.0,
            alpha_strong=16.0,
            epochs=600,
            seed=0,
        )
        for key in ("m_weak", "m_strong", "m_imit", "m_ceiling"):
            assert -1.0 <= res[key] <= 1.0
        assert res["m_ceiling"] >= res["m_weak"] - 0.1
        # the strong student should at least imitate its supervisor
        assert res["m_imit"] > 0.5


class TestCollapse:
    """Recursive synthetic-data training."""

    def test_run_collapse_trajectories(self):
        """Trajectory lengths match; full real data keeps overlap high."""
        res = run_collapse(
            d=24, hidden=4, alpha=8.0, n_generations=3, p_real=1.0, epochs=500, seed=0
        )
        assert len(res["m"]) == len(res["q_ratio"]) == 4
        assert res["m"][-1] > 0.6  # p_real=1 is ordinary retraining on truth

    def test_synthetic_loop_degrades_more_than_real(self):
        """p_real=0 should end with overlap <= p_real=1 (allow small noise)."""
        kw = {"d": 24, "hidden": 4, "alpha": 6.0, "n_generations": 4, "epochs": 400, "seed": 0}
        res_syn = run_collapse(p_real=0.0, **kw)
        res_real = run_collapse(p_real=1.0, **kw)
        assert res_syn["m"][-1] <= res_real["m"][-1] + 0.1


class TestICL:
    """In-context learning batches, ridge predictor, tiny training run."""

    def test_make_icl_batch_shapes(self):
        """Token layout: (batch, k+1, d+1) with the query label hidden."""
        tokens, y_q, W = make_icl_batch(None, batch=8, d=4, k=6)
        assert tokens.shape == (8, 7, 5)
        assert y_q.shape == (8,)
        assert W.shape == (8, 4)
        assert torch.all(tokens[:, -1, -1] == 0.0)  # query label zeroed

    def test_ridge_predictor_beats_null(self):
        """With enough context, ridge is much better than predicting 0."""
        torch.manual_seed(0)
        tokens, y_q, _ = make_icl_batch(None, batch=256, d=4, k=16)
        pred = ridge_predictor(tokens)
        eps_ridge = ((pred - y_q) ** 2).mean().item()
        eps_null = (y_q**2).mean().item()
        assert eps_ridge < 0.5 * eps_null

    def test_run_icl_smoke(self):
        """A tiny run returns finite scores with the right keys."""
        res = run_icl(n_tasks=4, d=4, k=4, d_model=16, n_layers=1, steps=30, n_eval=64, seed=0)
        for key in ("icl_score", "memo_score", "ridge_alignment", "eps_unseen"):
            assert np.isfinite(res[key])


class TestTeacherTaxonomy:
    """Teacher taxonomy registry and constructors."""

    def test_registry_families(self):
        """Every spec declares a known family."""
        from statphys.frontier import TEACHER_TAXONOMY

        assert len(TEACHER_TAXONOMY) >= 8
        for spec in TEACHER_TAXONOMY.values():
            assert spec.family in ("random", "structured", "trained")

    def test_make_teacher_random(self):
        """A random teacher labels sampler outputs with O(1) statistics."""
        from statphys.frontier import make_teacher

        teacher, sampler, d = make_teacher("random_mlp", d=16, hidden=4, seed=0)
        X = sampler(64)
        assert X.shape == (64, d)
        y = teacher(X)
        assert y.shape == (64,)
        assert torch.isfinite(y).all()

    def test_make_teacher_unknown_raises(self):
        """Unknown taxonomy keys are rejected."""
        from statphys.frontier import make_teacher

        with pytest.raises(ValueError):
            make_teacher("no_such_teacher")

    def test_trained_digits_teacher_learns_parity(self):
        """The trained teacher correlates with digit parity on real images."""
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_digits

        from statphys.frontier import make_teacher

        teacher, sampler, d = make_teacher("trained_digits", hidden=32, seed=0, noise_std=0.0)
        assert d == 64
        X_raw, y_raw = load_digits(return_X_y=True)
        X = torch.as_tensor(X_raw, dtype=torch.float32)
        X = (X - X.mean()) / X.std()
        target = torch.where(torch.as_tensor(y_raw) >= 5, 1.0, -1.0)
        pred = teacher.clean(X)
        acc = ((pred > 0) == (target > 0)).float().mean().item()
        assert acc > 0.8

    def test_taxonomy_table_renders(self):
        """The markdown table lists every teacher."""
        from statphys.frontier.teachers import TEACHER_TAXONOMY, taxonomy_table

        table = taxonomy_table()
        for name in TEACHER_TAXONOMY:
            assert f"`{name}`" in table


class TestTaxonomyCross:
    """Teacher x paradigm cross experiment."""

    def test_run_taxonomy_smoke(self):
        """A 1-teacher, 2-paradigm quick cross run returns finite cells."""
        from statphys.frontier import run_taxonomy, taxonomy_markdown

        res = run_taxonomy(
            teachers=["random_mlp"],
            paradigms=["sft", "collapse"],
            n_seeds=1,
            quick=True,
            verbose=False,
        )
        cell = res["cells"]["random_mlp"]
        assert np.isfinite(cell["sft"]["forgetting"][0])
        assert np.isfinite(cell["collapse"]["collapse_drop"][0])
        md = taxonomy_markdown(res)
        assert "`random_mlp`" in md

    def test_teacher_injection_in_protocols(self):
        """Protocols accept an injected teacher + input sampler."""
        from statphys.frontier import make_teacher, run_collapse

        teacher, sampler, d = make_teacher("sparse_mlp", d=24, hidden=4, seed=0)
        res = run_collapse(
            d=d,
            hidden=4,
            alpha=6.0,
            n_generations=2,
            p_real=1.0,
            epochs=300,
            n_probe=512,
            seed=0,
            teacher=teacher,
            input_sampler=sampler,
        )
        assert len(res["m"]) == 3
        assert np.isfinite(res["m"]).all()


class TestRegistry:
    """Frontier studies are exposed through the shared registry."""

    def test_frontier_studies_registered(self):
        """All six frontier studies appear in the CLI registry."""
        from statphys.experiment.studies import STUDIES

        for name in ("sft", "rlhf", "weak_to_strong", "collapse", "icl", "taxonomy"):
            assert name in STUDIES
