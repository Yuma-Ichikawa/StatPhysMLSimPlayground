"""
Numerical phase diagrams for teacher-student experiments.

run_phase_diagram sweeps a 2D grid (control parameter x sample ratio),
runs replica-resolved experiments at every grid point, and collects
order-parameter maps -- a purely numerical analogue of the analytic
phase diagrams of statistical-learning theory, applicable to any
architecture.

Example:
    >>> def factory(snr):
    ...     teacher = Teacher(nn.Linear(d, 1, bias=False),
    ...                       init="spiked", init_kwargs={"snr": snr})
    ...     return TeacherStudentExperiment(teacher, student_factory, d=d)
    >>> result = run_phase_diagram(factory, param_name="snr",
    ...                            param_values=[0.5, 1, 2, 4],
    ...                            alphas=[0.5, 1, 2, 4, 8])
    >>> result.plot("m_hat")

"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from statphys.experiment.protocol import ExperimentResult, TeacherStudentExperiment

__all__ = ["PhaseDiagramResult", "run_phase_diagram"]


@dataclass
class PhaseDiagramResult:
    """
    Container for a 2D numerical phase diagram.

    Attributes:
        alphas: Sample-ratio axis (columns).
        param_name: Name of the control parameter (rows).
        param_values: Control-parameter axis.
        grids: Mapping metric -> (n_param, n_alpha) array of
            replica-averaged values.
        config: Sweep configuration snapshot.
        metadata: Wall time etc.

    """

    alphas: list[float]
    param_name: str
    param_values: list[float]
    grids: dict[str, list[list[float]]]
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def grid(self, metric: str) -> np.ndarray:
        """Return a metric grid as a (n_param, n_alpha) array."""
        return np.asarray(self.grids[metric])

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "alphas": self.alphas,
            "param_name": self.param_name,
            "param_values": self.param_values,
            "grids": self.grids,
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseDiagramResult":
        """Restore from a dictionary."""
        return cls(**data)

    def plot(
        self,
        metric: str = "m_hat",
        ax=None,
        cmap: str = "viridis",
        logx: bool = False,
        contour_level: float | None = None,
        show: bool = False,
    ):
        """
        Heatmap of a metric over the (alpha, parameter) plane.

        Args:
            metric: Which grid to draw.
            ax: Existing matplotlib Axes.
            cmap: Colormap.
            logx: Log-scale the alpha axis.
            contour_level: If given, overlay the contour at this value
                (e.g. 0.5 to trace a numerically estimated phase boundary).
            show: Call plt.show().

        Returns:
            Tuple of (Figure, Axes).

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 5))
        else:
            fig = ax.get_figure()

        Z = self.grid(metric)
        A, P = np.meshgrid(self.alphas, self.param_values)
        pc = ax.pcolormesh(A, P, Z, cmap=cmap, shading="nearest")
        fig.colorbar(pc, ax=ax, label=metric)
        if contour_level is not None:
            try:
                cs = ax.contour(A, P, Z, levels=[contour_level], colors="w", linewidths=2)
                ax.clabel(cs, fmt=f"{metric}=%g")
            except Exception:
                pass
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha = n/d$")
        ax.set_ylabel(self.param_name)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax


def run_phase_diagram(
    experiment_factory: Callable[[float], TeacherStudentExperiment],
    param_name: str,
    param_values: Sequence[float],
    alphas: Sequence[float],
    n_replicas: int = 3,
    base_seed: int = 0,
    verbose: bool = True,
    **train_kwargs: Any,
) -> PhaseDiagramResult:
    """
    Sweep (control parameter x alpha) and collect order-parameter grids.

    For each parameter value, a fresh experiment is built via
    `experiment_factory(value)` and `run_order_parameters` is executed
    over the alpha axis. Replica-averaged per-replica metrics and
    cross-replica aggregates (q_ab_mean, chi_m, binder_m) are stored as
    2D grids.

    Args:
        experiment_factory: Callable value -> TeacherStudentExperiment.
        param_name: Name of the control parameter (for labelling).
        param_values: Row axis of the diagram.
        alphas: Column axis (sample ratios).
        n_replicas: Replicas per grid point.
        base_seed: Base RNG seed (offset per row for independence).
        verbose: Print progress.
        **train_kwargs: Forwarded to run_order_parameters
            (lr, max_epochs, weight_decay, n_probe, ...).

    Returns:
        PhaseDiagramResult with one grid per recorded metric.

    """
    alphas = [float(a) for a in alphas]
    param_values = [float(v) for v in param_values]
    grids: dict[str, list[list[float]]] = {}
    t_start = time.time()
    config_snapshot: dict[str, Any] = {}

    for ip, value in enumerate(param_values):
        if verbose:
            print(f"--- {param_name} = {value:.4g} ({ip + 1}/{len(param_values)}) ---")
        exp = experiment_factory(value)
        res: ExperimentResult = exp.run_order_parameters(
            alphas=alphas,
            n_replicas=n_replicas,
            base_seed=base_seed + 7919 * ip,
            verbose=verbose,
            **train_kwargs,
        )
        if not config_snapshot:
            config_snapshot = res.config
        for metric in res.metrics():
            grids.setdefault(metric, []).append(res.mean(metric).tolist())

    return PhaseDiagramResult(
        alphas=alphas,
        param_name=param_name,
        param_values=param_values,
        grids=grids,
        config={
            "n_replicas": n_replicas,
            "base_seed": base_seed,
            "train_kwargs": {k: v for k, v in train_kwargs.items() if np.isscalar(v)},
            "example_point": config_snapshot,
        },
        metadata={"wall_time_sec": time.time() - t_start},
    )
