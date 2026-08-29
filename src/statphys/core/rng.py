"""Named, reproducible random-number streams for scientific experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import torch

RNG_STREAMS = (
    "teacher_or_environment_disorder",
    "data_disorder",
    "initialization",
    "minibatch_order",
    "dropout",
    "diffusion_noise",
    "rollout",
    "evaluation",
    "intervention",
    "bootstrap",
    # Compatibility names used by the original public API.
    "teacher",
    "data",
    "training",
)


@dataclass(frozen=True)
class SeedStreams:
    """Derive stable independent generators from one user-visible seed."""

    root: int

    def __post_init__(self) -> None:
        if not isinstance(self.root, int) or self.root < 0:
            raise ValueError("root seed must be a non-negative integer")

    def seed(self, name: str) -> int:
        if not name or not name.strip():
            raise ValueError("stream name must be non-empty")
        payload = f"statphys-rng-v1:{self.root}:{name}".encode()
        return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") % (2**63 - 1)

    def numpy(self, name: str) -> np.random.Generator:
        """Return a fresh NumPy generator for ``name``."""
        return np.random.default_rng(self.seed(name))

    def torch(self, name: str, *, device: str = "cpu") -> torch.Generator:
        """Return a fresh torch generator for ``name`` and ``device``."""
        return torch.Generator(device=device).manual_seed(self.seed(name))
