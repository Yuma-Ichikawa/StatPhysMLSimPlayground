"""
I/O utilities for saving and loading simulation results.
"""

import json
import pickle
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and torch tensors."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "auto",
) -> Path:
    """
    Save simulation results to file.

    Args:
        results: Dictionary containing results.
        filepath: Path to save file.
        format: File format ('json', 'pickle', 'npz', or 'auto' to infer from extension).

    Returns:
        Path to saved file.

    Example:
        >>> results = {"alpha": [0.1, 0.5, 1.0], "error": [0.5, 0.3, 0.1]}
        >>> save_results(results, "results.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        format = filepath.suffix.lstrip(".")
        if not format:
            format = "json"

    if format == "json":
        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        return filepath.with_suffix(".json")

    elif format == "pickle" or format == "pkl":
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(results, f)
        return filepath.with_suffix(".pkl")

    elif format == "npz":
        # Convert to numpy arrays where possible
        np_results = {}
        for k, v in results.items():
            if isinstance(v, (list, np.ndarray)):
                np_results[k] = np.array(v)
            elif isinstance(v, torch.Tensor):
                np_results[k] = v.cpu().numpy()
            else:
                np_results[k] = np.array([v])
        np.savez(filepath.with_suffix(".npz"), **np_results)
        return filepath.with_suffix(".npz")

    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(
    filepath: Union[str, Path],
    format: str = "auto",
) -> Dict[str, Any]:
    """
    Load simulation results from file.

    Args:
        filepath: Path to results file.
        format: File format ('json', 'pickle', 'npz', or 'auto' to infer from extension).

    Returns:
        Dictionary containing loaded results.
    """
    filepath = Path(filepath)

    if format == "auto":
        format = filepath.suffix.lstrip(".")

    if format == "json":
        with open(filepath, "r") as f:
            return json.load(f)

    elif format == "pickle" or format == "pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    elif format == "npz":
        data = np.load(filepath, allow_pickle=True)
        return {k: data[k] for k in data.files}

    else:
        raise ValueError(f"Unknown format: {format}")


class ResultsManager:
    """
    Manager for organizing and storing simulation results.

    Attributes:
        base_dir: Base directory for storing results.
        experiment_name: Name of the experiment.
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = "./results",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize ResultsManager.

        Args:
            base_dir: Base directory for storing results.
            experiment_name: Name of the experiment. If None, uses timestamp.
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Store metadata
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
        }

    def save(
        self,
        name: str,
        data: Dict[str, Any],
        format: str = "json",
    ) -> Path:
        """
        Save data with a given name.

        Args:
            name: Name for the saved data.
            data: Data to save.
            format: File format.

        Returns:
            Path to saved file.
        """
        filepath = self.experiment_dir / name
        return save_results(data, filepath, format)

    def load(self, name: str, format: str = "auto") -> Dict[str, Any]:
        """
        Load data by name.

        Args:
            name: Name of the saved data.
            format: File format.

        Returns:
            Loaded data.
        """
        filepath = self.experiment_dir / name
        return load_results(filepath, format)

    def save_config(self, config: Dict[str, Any]) -> Path:
        """Save experiment configuration."""
        return self.save("config", config, format="json")

    def save_theory_results(self, results: Dict[str, Any]) -> Path:
        """Save theory computation results."""
        return self.save("theory_results", results, format="json")

    def save_experiment_results(self, results: Dict[str, Any]) -> Path:
        """Save numerical experiment results."""
        return self.save("experiment_results", results, format="npz")

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Update and save metadata."""
        self._metadata.update(metadata)
        self._metadata["updated_at"] = datetime.now().isoformat()
        return self.save("metadata", self._metadata, format="json")

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Args:
            results: Results dictionary with array values.

        Returns:
            DataFrame with results.
        """
        return pd.DataFrame(results)

    def list_files(self) -> list[Path]:
        """List all files in the experiment directory."""
        return list(self.experiment_dir.iterdir())

    def __repr__(self) -> str:
        return f"ResultsManager(experiment_dir={self.experiment_dir})"
