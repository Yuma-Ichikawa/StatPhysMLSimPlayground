"""
Base classes for dataset generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np


class TeacherType(Enum):
    """Enum for different teacher model types."""
    LINEAR = "linear"
    COMMITTEE = "committee"
    MLP = "mlp"
    TRANSFORMER = "transformer"
    CUSTOM = "custom"


@dataclass
class TeacherConfig:
    """Configuration for teacher model in data generation."""
    teacher_type: TeacherType = TeacherType.LINEAR
    hidden_dim: Optional[int] = None  # For committee/MLP
    num_heads: Optional[int] = None  # For transformer
    activation: str = "relu"
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """
    Abstract base class for all datasets in statistical mechanics simulations.

    This class defines the interface for generating training data in the
    teacher-student framework.

    Attributes:
        d: Input dimension.
        device: Device for tensor operations ('cpu' or 'cuda').
        dtype: Data type for tensors.

    Subclasses must implement:
        - generate_sample(): Generate a single (x, y) pair
        - get_teacher_params(): Return teacher model parameters for order param computation
    """

    def __init__(
        self,
        d: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        """
        Initialize the dataset.

        Args:
            d: Input dimension.
            device: Device for tensor operations. Defaults to 'cpu'.
            dtype: Data type for tensors. Defaults to torch.float32.
            **kwargs: Additional keyword arguments for subclasses.
        """
        self.d = d
        self.device = torch.device(device)
        self.dtype = dtype
        self._teacher_params: Dict[str, Any] = {}

    @abstractmethod
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single data sample (x, y).

        Returns:
            Tuple of:
                - x: Input tensor of shape (d,)
                - y: Output tensor (scalar or vector depending on task)
        """
        pass

    def generate_dataset(
        self,
        n_samples: int,
        return_as_batch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of n_samples.

        Args:
            n_samples: Number of samples to generate.
            return_as_batch: If True, return stacked tensors. If False, return lists.

        Returns:
            Tuple of:
                - X: Input tensor of shape (n_samples, d)
                - y: Output tensor of shape (n_samples,) or (n_samples, output_dim)
        """
        X_list = []
        y_list = []

        for _ in range(n_samples):
            x, y = self.generate_sample()
            X_list.append(x)
            y_list.append(y)

        if return_as_batch:
            X = torch.stack(X_list, dim=0)
            y = torch.stack(y_list, dim=0) if y_list[0].dim() > 0 else torch.tensor(
                [yi.item() if yi.dim() == 0 else yi for yi in y_list],
                device=self.device,
                dtype=self.dtype,
            )
            return X, y
        else:
            return X_list, y_list

    @abstractmethod
    def get_teacher_params(self) -> Dict[str, Any]:
        """
        Get teacher model parameters.

        These parameters are used for computing order parameters
        (overlap with teacher, generalization error, etc.)

        Returns:
            Dictionary containing teacher parameters (e.g., W0, rho, eta).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get dataset configuration as a dictionary.

        Returns:
            Dictionary containing dataset configuration.
        """
        return {
            "class": self.__class__.__name__,
            "d": self.d,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self.d

    @property
    def teacher_params(self) -> Dict[str, Any]:
        """Return teacher parameters."""
        return self._teacher_params

    def to(self, device: str) -> "BaseDataset":
        """
        Move dataset to specified device.

        Args:
            device: Target device.

        Returns:
            Self (for method chaining).
        """
        self.device = torch.device(device)
        # Move teacher parameters to new device
        for key, value in self._teacher_params.items():
            if isinstance(value, torch.Tensor):
                self._teacher_params[key] = value.to(self.device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, device={self.device})"
