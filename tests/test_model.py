"""
Tests for model module.
"""

import pytest
import torch
import numpy as np

from statphys.model import (
    LinearRegression,
    LinearClassifier,
    CommitteeMachine,
    SoftCommitteeMachine,
    TwoLayerNetwork,
    get_model,
)
from statphys.utils import fix_seed


class TestLinearRegression:
    """Tests for LinearRegression."""

    def test_init(self):
        """Test model initialization."""
        d = 100
        model = LinearRegression(d=d)

        assert model.d == d
        assert model.W.shape == (d, 1)

    def test_forward(self):
        """Test forward pass."""
        fix_seed(42)
        d = 100
        batch_size = 32
        model = LinearRegression(d=d)

        x = torch.randn(batch_size, d)
        y = model(x)

        assert y.shape == (batch_size,)

    def test_forward_single(self):
        """Test forward with single sample."""
        fix_seed(42)
        d = 100
        model = LinearRegression(d=d)

        x = torch.randn(d)
        y = model(x)

        assert y.dim() == 0 or y.shape == (1,)

    def test_get_weight_vector(self):
        """Test weight vector retrieval."""
        d = 100
        model = LinearRegression(d=d)

        w = model.get_weight_vector()
        assert w.shape == (d,)

    def test_order_params(self):
        """Test order parameter computation."""
        fix_seed(42)
        d = 100
        model = LinearRegression(d=d)

        W0 = torch.randn(d, 1)
        teacher_params = {"W0": W0, "rho": 1.0, "eta": 0.0}

        order_params = model.compute_order_params(teacher_params)

        assert "m" in order_params
        assert "q" in order_params
        assert "eg" in order_params


class TestLinearClassifier:
    """Tests for LinearClassifier."""

    def test_output_types(self):
        """Test different output types."""
        d = 100
        x = torch.randn(10, d)

        # Sign output
        model_sign = LinearClassifier(d=d, output_type="sign")
        y_sign = model_sign(x)
        assert all(torch.abs(y_sign) == 1)  # All +1 or -1

        # Logit output
        model_logit = LinearClassifier(d=d, output_type="logit")
        y_logit = model_logit(x)
        # Logits can be any real value

        # Probability output
        model_prob = LinearClassifier(d=d, output_type="prob")
        y_prob = model_prob(x)
        assert all((y_prob >= 0) & (y_prob <= 1))


class TestCommitteeMachine:
    """Tests for CommitteeMachine."""

    def test_init(self):
        """Test initialization."""
        d = 100
        k = 5
        model = CommitteeMachine(d=d, k=k)

        assert model.d == d
        assert model.k == k
        assert model.W.shape == (k, d)

    def test_forward(self):
        """Test forward pass."""
        fix_seed(42)
        d = 100
        k = 5
        batch_size = 32
        model = CommitteeMachine(d=d, k=k)

        x = torch.randn(batch_size, d)
        y = model(x)

        assert y.shape == (batch_size,)

    def test_get_weight_vectors(self):
        """Test weight matrix retrieval."""
        d = 100
        k = 5
        model = CommitteeMachine(d=d, k=k)

        W = model.get_weight_vectors()
        assert W.shape == (k, d)


class TestSoftCommitteeMachine:
    """Tests for SoftCommitteeMachine."""

    def test_activations(self):
        """Test different activations."""
        d = 100
        k = 3
        x = torch.randn(10, d)

        for activation in ["erf", "tanh", "relu"]:
            model = SoftCommitteeMachine(d=d, k=k, activation=activation)
            y = model(x)
            assert y.shape == (10,)


class TestTwoLayerNetwork:
    """Tests for TwoLayerNetwork."""

    def test_init(self):
        """Test initialization."""
        d = 100
        k = 50
        model = TwoLayerNetwork(d=d, k=k)

        assert model.d == d
        assert model.k == k

    def test_forward(self):
        """Test forward pass."""
        fix_seed(42)
        d = 100
        k = 50
        batch_size = 32
        model = TwoLayerNetwork(d=d, k=k)

        x = torch.randn(batch_size, d)
        y = model(x)

        assert y.shape == (batch_size,)

    def test_second_layer_fixed(self):
        """Test fixed second layer."""
        d = 100
        k = 50
        model = TwoLayerNetwork(d=d, k=k, second_layer_fixed=True)

        # Second layer should not be a parameter
        assert not model.a.requires_grad


class TestRegistry:
    """Tests for model registry."""

    def test_get_registered_model(self):
        """Test getting model from registry."""
        model = get_model("linear", d=100)
        assert isinstance(model, LinearRegression)

    def test_get_committee(self):
        """Test getting committee machine."""
        model = get_model("committee", d=100, k=5)
        assert isinstance(model, CommitteeMachine)
