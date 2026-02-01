"""Tests for comprehensive order parameter calculation."""

import numpy as np
import pytest
import torch

from statphys.dataset import GaussianClassificationDataset, GaussianDataset
from statphys.model import CommitteeMachine, LinearRegression, TwoLayerNetwork
from statphys.model.committee import SoftCommitteeMachine
from statphys.utils.order_params import (
    ModelType,
    OrderParameterCalculator,
    OrderParameters,
    TaskType,
    auto_calc_order_params,
)
from statphys.utils.seed import fix_seed


@pytest.fixture
def reset_seed():
    """Reset random seed before each test."""
    fix_seed(42)


class TestOrderParameters:
    """Test OrderParameters dataclass."""

    def test_to_dict(self):
        """Test conversion to flat dictionary."""
        params = OrderParameters(
            student_teacher_overlaps={"w_W0": 0.5},
            student_self_overlaps={"w_w": 0.9},
            scalars={"bias": 0.1},
            generalization_error=0.05,
        )

        d = params.to_dict()
        assert d["M_w_W0"] == 0.5
        assert d["Q_w_w"] == 0.9
        assert d["bias"] == 0.1
        assert d["eg"] == 0.05

    def test_to_list_linear(self):
        """Test conversion to list for linear model."""
        params = OrderParameters(
            student_teacher_overlaps={"w_W0": 0.5},
            student_self_overlaps={"w_w": 0.9},
            generalization_error=0.05,
        )

        lst = params.to_list(ModelType.LINEAR)
        assert len(lst) == 3
        assert lst[0] == 0.5  # m
        assert lst[1] == 0.9  # q
        assert lst[2] == 0.05  # eg

    def test_to_list_committee(self):
        """Test conversion to list for committee machine."""
        params = OrderParameters(
            student_teacher_overlaps={"avg": 0.5},
            student_self_overlaps={"diag_avg": 0.9, "offdiag_avg": 0.1},
            generalization_error=0.05,
        )

        lst = params.to_list(ModelType.COMMITTEE)
        assert len(lst) == 4
        assert lst[0] == 0.5  # m_avg
        assert lst[1] == 0.9  # q_diag
        assert lst[2] == 0.1  # q_offdiag
        assert lst[3] == 0.05  # eg

    def test_summary(self):
        """Test human-readable summary."""
        params = OrderParameters(
            student_teacher_overlaps={"w_W0": 0.5},
            student_self_overlaps={"w_w": 0.9},
            scalars={"bias": 0.1},
            generalization_error=0.05,
        )

        summary = params.summary()
        assert "Student-Teacher Overlaps" in summary
        assert "Student Self-Overlaps" in summary
        assert "Scalars" in summary
        assert "Generalization Error" in summary


class TestOrderParameterCalculator:
    """Test OrderParameterCalculator class."""

    def test_init(self, reset_seed):
        """Test calculator initialization."""
        calc = OrderParameterCalculator()
        assert calc.return_format == "list"
        assert calc.include_matrices is True

        calc_obj = OrderParameterCalculator(return_format="object", include_matrices=False)
        assert calc_obj.return_format == "object"
        assert calc_obj.include_matrices is False

    def test_linear_model_detection(self, reset_seed):
        """Test model type detection for linear models."""
        calc = OrderParameterCalculator()
        model = LinearRegression(d=100)

        model_type = calc._detect_model_type(model)
        assert model_type == ModelType.LINEAR

    def test_committee_model_detection(self, reset_seed):
        """Test model type detection for committee machines."""
        calc = OrderParameterCalculator()
        model = CommitteeMachine(d=100, k=3)

        model_type = calc._detect_model_type(model)
        assert model_type == ModelType.COMMITTEE

    def test_two_layer_model_detection(self, reset_seed):
        """Test model type detection for two-layer networks."""
        calc = OrderParameterCalculator()
        model = TwoLayerNetwork(d=100, k=10)

        model_type = calc._detect_model_type(model)
        assert model_type == ModelType.TWO_LAYER

    def test_task_detection_regression(self, reset_seed):
        """Test task type detection for regression."""
        calc = OrderParameterCalculator()
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.1)

        task_type = calc._detect_task_type(dataset)
        assert task_type == TaskType.REGRESSION

    def test_task_detection_classification(self, reset_seed):
        """Test task type detection for classification."""
        calc = OrderParameterCalculator()
        dataset = GaussianClassificationDataset(d=100, rho=1.0)

        task_type = calc._detect_task_type(dataset)
        assert task_type == TaskType.BINARY_CLASSIFICATION

    def test_linear_full_computation(self, reset_seed):
        """Test full order parameter computation for linear model."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100, init_scale=0.5)

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # Check structure
        assert isinstance(params, OrderParameters)
        assert "w_W0" in params.student_teacher_overlaps
        assert "w_w" in params.student_self_overlaps
        assert "W0_W0" in params.teacher_self_overlaps
        assert params.generalization_error is not None

        # Check values are reasonable
        assert -1 <= params.student_teacher_overlaps["w_W0"] <= 1
        assert params.student_self_overlaps["w_w"] > 0

    def test_committee_full_computation(self, reset_seed):
        """Test full order parameter computation for committee machine."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = SoftCommitteeMachine(d=100, k=3, activation="erf")

        calc = OrderParameterCalculator(return_format="object", include_matrices=True)
        params = calc(dataset, model)

        # Check student-teacher overlaps
        assert "avg" in params.student_teacher_overlaps
        # Should have W_0_W0, W_1_W0, W_2_W0
        assert any("W_0" in k for k in params.student_teacher_overlaps.keys())

        # Check student self-overlaps (Q matrix)
        assert "diag_avg" in params.student_self_overlaps
        assert "offdiag_avg" in params.student_self_overlaps
        if calc.include_matrices:
            assert "matrix" in params.student_self_overlaps

        # Check Q matrix dimensions (K x K)
        if "matrix" in params.student_self_overlaps:
            Q = params.student_self_overlaps["matrix"]
            assert len(Q) == 3
            assert len(Q[0]) == 3

    def test_two_layer_full_computation(self, reset_seed):
        """Test full order parameter computation for two-layer network."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = TwoLayerNetwork(d=100, k=5, activation="relu")

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # Check scalars (second layer weights)
        assert "a" in params.scalars
        assert "a_norm" in params.scalars
        assert "a_mean" in params.scalars
        assert "K" in params.scalars
        assert params.scalars["K"] == 5

    def test_list_format(self, reset_seed):
        """Test list output format."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100, init_scale=0.5)

        calc = OrderParameterCalculator(return_format="list")
        params = calc(dataset, model)

        assert isinstance(params, list)
        assert len(params) == 3  # m, q, eg
        assert all(isinstance(p, float) for p in params)

    def test_dict_format(self, reset_seed):
        """Test dict output format."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100, init_scale=0.5)

        calc = OrderParameterCalculator(return_format="dict")
        params = calc(dataset, model)

        assert isinstance(params, dict)
        assert "M_w_W0" in params
        assert "Q_w_w" in params
        assert "eg" in params

    def test_classification_error(self, reset_seed):
        """Test classification error computation."""
        dataset = GaussianClassificationDataset(d=100, rho=1.0)
        model = LinearRegression(d=100, init_scale=0.5)

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # Classification error should be between 0 and 0.5
        assert 0 <= params.generalization_error <= 0.5

    def test_regression_error(self, reset_seed):
        """Test regression error computation."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100, init_scale=0.5)

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # Generalization error should be non-negative
        assert params.generalization_error >= 0

    def test_teacher_self_overlaps(self, reset_seed):
        """Test teacher self-overlap computation."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100)

        calc = OrderParameterCalculator(
            return_format="object", include_teacher_overlaps=True
        )
        params = calc(dataset, model)

        # Should have teacher self-overlap (rho)
        assert "W0_W0" in params.teacher_self_overlaps
        # rho = W0^T @ W0 / d should be close to 1.0
        assert 0.5 < params.teacher_self_overlaps["W0_W0"] < 1.5

    def test_scalars_extraction(self, reset_seed):
        """Test extraction of scalar quantities."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.1)
        model = TwoLayerNetwork(d=100, k=5)

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # Should have teacher parameters as scalars
        assert "teacher_rho" in params.scalars
        assert "teacher_eta" in params.scalars
        assert params.scalars["teacher_rho"] == 1.0
        assert params.scalars["teacher_eta"] == 0.1


class TestAutoCalcOrderParams:
    """Test auto_calc_order_params convenience function."""

    def test_basic_usage(self, reset_seed):
        """Test basic usage."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100)

        params = auto_calc_order_params(dataset, model)
        assert isinstance(params, list)
        assert len(params) == 3

    def test_dict_format(self, reset_seed):
        """Test dict format option."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100)

        params = auto_calc_order_params(dataset, model, return_format="dict")
        assert isinstance(params, dict)

    def test_object_format(self, reset_seed):
        """Test object format option."""
        dataset = GaussianDataset(d=100, rho=1.0, eta=0.0)
        model = LinearRegression(d=100)

        params = auto_calc_order_params(dataset, model, return_format="object")
        assert isinstance(params, OrderParameters)


class TestParamNames:
    """Test parameter name retrieval."""

    def test_linear_names(self):
        """Test parameter names for linear model."""
        names = OrderParameterCalculator.get_param_names(ModelType.LINEAR)
        assert names == ["m", "q", "eg"]

    def test_committee_names(self):
        """Test parameter names for committee machine."""
        names = OrderParameterCalculator.get_param_names(ModelType.COMMITTEE)
        assert names == ["m_avg", "q_diag_avg", "q_offdiag_avg", "eg"]

    def test_two_layer_names(self):
        """Test parameter names for two-layer network."""
        names = OrderParameterCalculator.get_param_names(ModelType.TWO_LAYER)
        assert names == ["m_avg", "q_diag_avg", "q_offdiag_avg", "a_norm", "eg"]

    def test_string_input(self):
        """Test with string model type."""
        names = OrderParameterCalculator.get_param_names("linear")
        assert names == ["m", "q", "eg"]


class TestComprehensiveOverlaps:
    """Test that all overlaps are computed correctly."""

    def test_all_pairwise_overlaps_linear(self, reset_seed):
        """Test that linear model computes all overlaps."""
        dataset = GaussianDataset(d=50, rho=1.0)
        model = LinearRegression(d=50)

        calc = OrderParameterCalculator(return_format="object")
        params = calc(dataset, model)

        # For linear: should have w-W0 overlap and w-w self overlap
        assert len(params.student_teacher_overlaps) >= 1
        assert len(params.student_self_overlaps) >= 1

    def test_all_pairwise_overlaps_committee(self, reset_seed):
        """Test that committee machine computes all K x K0 overlaps."""
        dataset = GaussianDataset(d=50, rho=1.0)
        model = SoftCommitteeMachine(d=50, k=3)

        calc = OrderParameterCalculator(return_format="object", include_matrices=True)
        params = calc(dataset, model)

        # Should have individual overlaps W_i with W0
        overlap_count = sum(1 for k in params.student_teacher_overlaps.keys()
                          if k.startswith("W_") and "matrix" not in k)
        assert overlap_count >= 3  # At least 3 hidden units

        # Should have Q matrix elements
        self_overlap_count = sum(1 for k in params.student_self_overlaps.keys()
                                if k.startswith("W_") and "matrix" not in k)
        # 3 hidden units -> 3 diagonal + 3 off-diagonal = 6 unique pairs
        assert self_overlap_count >= 6

    def test_matrix_overlaps_included(self, reset_seed):
        """Test that full matrices are included when requested."""
        dataset = GaussianDataset(d=50, rho=1.0)
        model = SoftCommitteeMachine(d=50, k=3)

        calc = OrderParameterCalculator(return_format="object", include_matrices=True)
        params = calc(dataset, model)

        # Q matrix should be included
        assert "matrix" in params.student_self_overlaps
        Q = np.array(params.student_self_overlaps["matrix"])
        assert Q.shape == (3, 3)

        # Q should be symmetric
        assert np.allclose(Q, Q.T)
