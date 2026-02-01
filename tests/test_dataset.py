"""Tests for dataset module."""

import pytest
import torch

from statphys.dataset import (
    GaussianClassificationDataset,
    GaussianDataset,
    SparseDataset,
    get_dataset,
)
from statphys.utils import fix_seed


class TestGaussianDataset:
    """Tests for GaussianDataset."""

    def test_init(self):
        """Test dataset initialization."""
        d = 100
        dataset = GaussianDataset(d=d, rho=1.0, eta=0.1)

        assert dataset.d == d
        assert dataset.rho == 1.0
        assert dataset.eta == 0.1
        assert dataset.W0.shape == (d, 1)

    def test_generate_sample(self):
        """Test single sample generation."""
        fix_seed(42)
        d = 100
        dataset = GaussianDataset(d=d, rho=1.0, eta=0.0)

        x, y = dataset.generate_sample()

        assert x.shape == (d,)
        assert y.dim() == 0 or y.shape == ()  # Scalar

    def test_generate_dataset(self):
        """Test batch generation."""
        fix_seed(42)
        d = 100
        n_samples = 50
        dataset = GaussianDataset(d=d)

        X, y = dataset.generate_dataset(n_samples)

        assert X.shape == (n_samples, d)
        assert y.shape == (n_samples,)

    def test_teacher_params(self):
        """Test teacher parameter retrieval."""
        d = 100
        rho = 2.0
        eta = 0.5
        dataset = GaussianDataset(d=d, rho=rho, eta=eta)

        params = dataset.get_teacher_params()

        assert "W0" in params
        assert params["rho"] == rho
        assert params["eta"] == eta

    def test_device_transfer(self):
        """Test device transfer."""
        d = 100
        dataset = GaussianDataset(d=d)

        # Should not raise on CPU
        dataset.to("cpu")
        assert dataset.device == torch.device("cpu")

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        d = 100

        fix_seed(123)
        dataset1 = GaussianDataset(d=d, rho=1.0)
        x1, y1 = dataset1.generate_sample()

        fix_seed(123)
        dataset2 = GaussianDataset(d=d, rho=1.0)
        x2, y2 = dataset2.generate_sample()

        # Teacher should be the same
        assert torch.allclose(dataset1.W0, dataset2.W0)


class TestGaussianClassificationDataset:
    """Tests for GaussianClassificationDataset."""

    def test_labels_are_binary(self):
        """Test that labels are +1 or -1."""
        fix_seed(42)
        d = 100
        dataset = GaussianClassificationDataset(d=d)

        for _ in range(10):
            _, y = dataset.generate_sample()
            assert y.item() in [-1, 1]

    def test_flip_prob(self):
        """Test label flipping probability."""
        fix_seed(42)
        d = 100
        dataset = GaussianClassificationDataset(d=d, flip_prob=0.5)

        # With high flip probability, labels should be roughly balanced
        labels = []
        for _ in range(100):
            _, y = dataset.generate_sample()
            labels.append(y.item())

        # Not deterministic, but should have both labels
        assert len(set(labels)) == 2


class TestSparseDataset:
    """Tests for SparseDataset."""

    def test_sparsity(self):
        """Test that inputs are sparse."""
        fix_seed(42)
        d = 1000
        sparsity = 0.1
        dataset = SparseDataset(d=d, sparsity=sparsity)

        x, _ = dataset.generate_sample()

        # Fraction of non-zero elements should be close to sparsity
        nonzero_frac = (x != 0).float().mean().item()

        # Allow some variance
        assert 0.05 < nonzero_frac < 0.2


class TestRegistry:
    """Tests for dataset registry."""

    def test_get_registered_dataset(self):
        """Test getting dataset from registry."""
        dataset = get_dataset("gaussian", d=100, rho=1.0)
        assert isinstance(dataset, GaussianDataset)

    def test_get_unknown_dataset(self):
        """Test error on unknown dataset."""
        with pytest.raises(KeyError):
            get_dataset("nonexistent", d=100)
