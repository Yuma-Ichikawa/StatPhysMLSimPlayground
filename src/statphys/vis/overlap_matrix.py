"""
Overlap-matrix visualization for committee machines and multi-layer models.

Visualizes the order-parameter matrices that characterize learning in
networks with multiple hidden units:

- M (student-teacher overlaps, K x M matrix)
- Q (student-student overlaps, K x K matrix)
- R (teacher-teacher overlaps, M x M matrix)

Also supports snapshots over training time to show specialization
(emergence of a diagonal structure in M).

Example:
    >>> from statphys.utils.order_params import OrderParameterCalculator
    >>> from statphys.vis.overlap_matrix import OverlapMatrixPlotter
    >>> calc = OrderParameterCalculator(return_format="object")
    >>> params = calc(dataset, model)
    >>> plotter = OverlapMatrixPlotter()
    >>> fig, axes = plotter.plot_from_order_params(params)

"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statphys.vis.plotter import PlotStyle, Plotter


class OverlapMatrixPlotter(Plotter):
    """Heatmap plotter for overlap matrices M, Q, R."""

    def __init__(
        self,
        style: PlotStyle | None = None,
        cmap: str = "RdBu_r",
    ):
        super().__init__(style)
        self.cmap = cmap

    def plot_matrix(
        self,
        matrix: np.ndarray,
        ax: Axes | None = None,
        title: str = "",
        xlabel: str = "teacher unit",
        ylabel: str = "student unit",
        vmin: float | None = None,
        vmax: float | None = None,
        annotate: bool | None = None,
        colorbar: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot a single overlap matrix as a heatmap.

        Args:
            matrix: 2D array of overlaps.
            ax: Axes. Creates new if None.
            title: Panel title.
            xlabel, ylabel: Axis labels.
            vmin, vmax: Color scale limits. Defaults to symmetric around 0.
            annotate: Write values in cells. Defaults to True for small matrices.
            colorbar: Add a colorbar.
            **kwargs: Extra imshow arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        if vmin is None and vmax is None:
            vabs = max(np.abs(matrix).max(), 1e-12)
            vmin, vmax = -vabs, vabs

        im = ax.imshow(matrix, cmap=self.cmap, vmin=vmin, vmax=vmax, **kwargs)
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if annotate is None:
            annotate = matrix.size <= 64
        if annotate:
            for (i, j), val in np.ndenumerate(matrix):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=self.style.font_size - 3,
                    color="black",
                )

        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        return fig, ax

    def plot_from_order_params(
        self,
        params: Any,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """
        Plot M, Q, R matrices from an OrderParameters object.

        Args:
            params: OrderParameters with student_teacher_overlaps,
                student_self_overlaps and teacher_self_overlaps dicts
                containing a "matrix" entry.
            figsize: Figure size.

        Returns:
            Tuple of (Figure, array of Axes).

        """
        panels: list[tuple[str, np.ndarray, str, str]] = []

        st = getattr(params, "student_teacher_overlaps", {}) or {}
        ss = getattr(params, "student_self_overlaps", {}) or {}
        tt = getattr(params, "teacher_self_overlaps", {}) or {}

        if "matrix" in st:
            panels.append(
                (r"$M$ (student-teacher)", np.array(st["matrix"]), "teacher unit", "student unit")
            )
        if "matrix" in ss:
            panels.append(
                (r"$Q$ (student-student)", np.array(ss["matrix"]), "student unit", "student unit")
            )
        if "matrix" in tt:
            panels.append(
                (r"$R$ (teacher-teacher)", np.array(tt["matrix"]), "teacher unit", "teacher unit")
            )

        if not panels:
            raise ValueError(
                "No overlap matrices found; the model may be a simple linear model "
                "(scalar m/q) rather than a committee/multi-unit network."
            )

        n = len(panels)
        figsize = figsize or (4.5 * n, 4)
        fig, axes = self.create_figure(1, n, figsize=figsize)
        axes = np.atleast_1d(axes)

        for ax, (title, mat, xl, yl) in zip(axes, panels, strict=False):
            self.plot_matrix(mat, ax=ax, title=title, xlabel=xl, ylabel=yl)

        fig.tight_layout()
        return fig, axes

    def plot_specialization_over_time(
        self,
        matrices: list[np.ndarray],
        t_values: list[float] | np.ndarray | None = None,
        n_snapshots: int = 5,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """
        Plot snapshots of the M matrix over training to show specialization.

        In committee machines, the student units start symmetric
        ("unspecialized plateau") and eventually each aligns with a distinct
        teacher unit — the M matrix becomes (a permutation of) a diagonal.

        Args:
            matrices: List of M matrices sampled over training.
            t_values: Times corresponding to each matrix (optional).
            n_snapshots: Number of evenly spaced snapshots to show.
            figsize: Figure size.

        Returns:
            Tuple of (Figure, array of Axes).

        """
        n_total = len(matrices)
        if n_total == 0:
            raise ValueError("matrices list is empty")
        n_snapshots = min(n_snapshots, n_total)
        indices = np.linspace(0, n_total - 1, n_snapshots).astype(int)

        figsize = figsize or (3.5 * n_snapshots, 3.5)
        fig, axes = self.create_figure(1, n_snapshots, figsize=figsize)
        axes = np.atleast_1d(axes)

        # Common color scale across snapshots
        vabs = max(max(np.abs(np.asarray(m)).max() for m in matrices), 1e-12)

        for ax, idx in zip(axes, indices, strict=False):
            title = (
                f"t = {t_values[idx]:.2f}" if t_values is not None else f"snapshot {idx}"
            )
            self.plot_matrix(
                np.asarray(matrices[idx]),
                ax=ax,
                title=title,
                vmin=-vabs,
                vmax=vabs,
                colorbar=(idx == indices[-1]),
                annotate=False,
            )

        fig.suptitle("Specialization of student units ($M$ matrix)", y=1.02)
        fig.tight_layout()
        return fig, axes
