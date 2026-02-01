"""Tests for loss module."""

import pytest
import torch
import torch.nn as nn

from statphys.loss import (
    HingeLoss,
    LassoLoss,
    LogisticLoss,
    MSELoss,
    RidgeLoss,
    get_loss,
)


class TestMSELoss:
    """Tests for MSELoss."""

    def test_basic(self):
        """Test basic MSE computation."""
        loss_fn = MSELoss()

        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(y_pred, y_true)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_zero(self):
        """Test non-zero loss."""
        loss_fn = MSELoss()

        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([2.0, 3.0, 4.0])

        loss = loss_fn(y_pred, y_true)
        expected = (1.0**2 + 1.0**2 + 1.0**2) / 3  # Mean of squared errors
        assert loss.item() == pytest.approx(expected, abs=1e-6)


class TestRidgeLoss:
    """Tests for RidgeLoss."""

    def test_with_regularization(self):
        """Test ridge loss with regularization."""
        reg_param = 0.1
        loss_fn = RidgeLoss(reg_param=reg_param)

        # Create simple model
        model = nn.Linear(10, 1, bias=False)
        nn.init.ones_(model.weight)

        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 2.0])

        loss = loss_fn(y_pred, y_true, model=model)

        # Should include regularization term
        reg_param * torch.sum(model.weight**2).item()
        assert loss.item() > 0  # Should be positive due to regularization


class TestLassoLoss:
    """Tests for LassoLoss."""

    def test_l1_regularization(self):
        """Test L1 regularization."""
        reg_param = 0.1
        loss_fn = LassoLoss(reg_param=reg_param)

        model = nn.Linear(10, 1, bias=False)
        nn.init.ones_(model.weight)

        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 2.0])

        loss = loss_fn(y_pred, y_true, model=model)

        # L1 regularization
        expected_reg = reg_param * torch.sum(torch.abs(model.weight)).item()
        assert loss.item() == pytest.approx(expected_reg, abs=1e-5)


class TestHingeLoss:
    """Tests for HingeLoss."""

    def test_correct_classification(self):
        """Test loss for correct classification."""
        loss_fn = HingeLoss()

        # Correct with large margin
        y_pred = torch.tensor([2.0, -2.0])  # Predictions with margin > 1
        y_true = torch.tensor([1.0, -1.0])  # Labels

        loss = loss_fn(y_pred, y_true)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_incorrect_classification(self):
        """Test loss for incorrect classification."""
        loss_fn = HingeLoss(margin=1.0)

        # Incorrect prediction
        y_pred = torch.tensor([-1.0])
        y_true = torch.tensor([1.0])

        loss = loss_fn(y_pred, y_true)
        assert loss.item() == pytest.approx(2.0, abs=1e-6)  # max(0, 1 - (-1)) = 2


class TestLogisticLoss:
    """Tests for LogisticLoss."""

    def test_positive(self):
        """Test that loss is always positive."""
        loss_fn = LogisticLoss()

        y_pred = torch.randn(10)
        y_true = torch.sign(torch.randn(10))

        loss = loss_fn(y_pred, y_true)
        assert loss.item() > 0


class TestRegistry:
    """Tests for loss registry."""

    def test_get_registered_loss(self):
        """Test getting loss from registry."""
        loss = get_loss("ridge", reg_param=0.01)
        assert isinstance(loss, RidgeLoss)
        assert loss.reg_param == 0.01

    def test_aliases(self):
        """Test loss aliases."""
        mse = get_loss("mse")
        l2 = get_loss("l2")
        assert type(mse) is type(l2)
