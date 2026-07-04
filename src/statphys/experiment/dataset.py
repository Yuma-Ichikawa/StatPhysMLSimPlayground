"""
Dataset for general teacher-student experiments.

Combines a Teacher with a configurable input distribution:

- "gaussian":    x ~ N(0, I)
- "correlated":  x ~ N(0, C) with a given covariance (or AR(1) correlation)
- "rademacher":  x_i in {-1, +1}
- "sphere":      x uniform on the sphere of radius sqrt(d)
- callable:      any function n -> (n, d) tensor

Example:
    >>> ds = TeacherStudentDataset(teacher, d=200, input_dist="correlated",
    ...                            input_kwargs={"ar_coeff": 0.6})
    >>> X, y = ds.sample(1000)

"""

from collections.abc import Callable
from typing import Any

import torch

from statphys.experiment.teacher import Teacher


def _ar1_covariance(d: int, ar_coeff: float) -> torch.Tensor:
    """AR(1) covariance C_ij = ar_coeff^|i-j|."""
    idx = torch.arange(d, dtype=torch.float32)
    return ar_coeff ** (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()


class TeacherStudentDataset:
    """
    Data generator pairing an input distribution with a Teacher.

    Args:
        teacher: Teacher instance producing labels.
        d: Input dimension.
        input_dist: "gaussian", "correlated", "rademacher", "sphere",
            or a callable n -> (n, d) tensor.
        input_kwargs: Options for the input distribution:
            - correlated: cov (tensor) or ar_coeff (float)
        device: Device for generated tensors.

    """

    def __init__(
        self,
        teacher: Teacher,
        d: int,
        input_dist: str | Callable[[int], torch.Tensor] = "gaussian",
        input_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
    ):
        self.teacher = teacher
        self.d = d
        self.device = torch.device(device)
        self.input_kwargs = input_kwargs or {}

        if callable(input_dist) and not isinstance(input_dist, str):
            self._sampler = input_dist
            self.input_dist = "custom"
        elif input_dist == "gaussian":
            self._sampler = self._sample_gaussian
            self.input_dist = input_dist
        elif input_dist == "correlated":
            cov = self.input_kwargs.get("cov")
            if cov is None:
                ar = float(self.input_kwargs.get("ar_coeff", 0.5))
                cov = _ar1_covariance(d, ar)
            cov = torch.as_tensor(cov, dtype=torch.float32, device=self.device)
            if cov.shape != (d, d):
                raise ValueError(f"cov must have shape ({d}, {d}), got {tuple(cov.shape)}")
            jitter = 1e-8 * torch.eye(d, device=self.device)
            self._chol = torch.linalg.cholesky(cov + jitter)
            self._sampler = self._sample_correlated
            self.input_dist = input_dist
        elif input_dist == "rademacher":
            self._sampler = self._sample_rademacher
            self.input_dist = input_dist
        elif input_dist == "sphere":
            self._sampler = self._sample_sphere
            self.input_dist = input_dist
        else:
            raise ValueError(f"Unknown input_dist: {input_dist!r}")

    def _sample_gaussian(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.d, device=self.device)

    def _sample_correlated(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.d, device=self.device)
        return z @ self._chol.T

    def _sample_rademacher(self, n: int) -> torch.Tensor:
        return torch.where(
            torch.rand(n, self.d, device=self.device) < 0.5,
            -torch.ones(1, device=self.device),
            torch.ones(1, device=self.device),
        )

    def _sample_sphere(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.d, device=self.device)
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return z * self.d**0.5

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate n labelled samples (X, y)."""
        X = self._sampler(n)
        y = self.teacher(X)
        return X, y

    def get_config(self) -> dict[str, Any]:
        """Return a summary config for logging."""
        return {
            "d": self.d,
            "input_dist": self.input_dist,
            "input_kwargs": {k: v for k, v in self.input_kwargs.items() if k != "cov"},
            "teacher": self.teacher.get_config(),
        }

    def __repr__(self) -> str:
        return (
            f"TeacherStudentDataset(d={self.d}, input={self.input_dist}, "
            f"teacher={self.teacher!r})"
        )
