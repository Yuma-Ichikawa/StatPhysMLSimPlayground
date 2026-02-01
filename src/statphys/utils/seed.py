"""Random seed utilities for reproducible experiments."""

import random

import numpy as np
import torch


def fix_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility across all random number generators.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generators (CPU and CUDA)
    - CUDA deterministic behavior

    Args:
        seed: The seed value to use. Defaults to 42.

    Example:
        >>> from statphys.utils import fix_seed
        >>> fix_seed(123)
        >>> # Now all random operations are reproducible

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for computation.

    Args:
        prefer_cuda: If True, prefer CUDA if available. Defaults to True.

    Returns:
        torch.device: The selected device (cuda or cpu).

    Example:
        >>> device = get_device()
        >>> model = model.to(device)

    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_seed_list(num_seeds: int, base_seed: int = 100) -> list[int]:
    """
    Generate a list of seeds for multiple trials.

    Args:
        num_seeds: Number of seeds to generate.
        base_seed: Starting seed value. Defaults to 100.

    Returns:
        List of seed values.

    Example:
        >>> seeds = get_seed_list(5)
        >>> # seeds = [100, 101, 102, 103, 104]

    """
    return [base_seed + i for i in range(num_seeds)]
