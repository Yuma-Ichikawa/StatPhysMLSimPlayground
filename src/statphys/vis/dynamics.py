"""
Dynamics visualization for online learning.

Provides phase-portrait style visualizations of the (m, q) order-parameter
plane for online-learning ODE scenarios:

- Vector field (flow) of the ODEs
- Nullclines (dm/dt = 0, dq/dt = 0)
- Theory trajectories from multiple initial conditions
- Per-seed experimental trajectories overlaid on the flow

Example:
    >>> from statphys.theory.online.scenario.gaussian_linear_mse import (
    ...     GaussianLinearMseEquations,
    ... )
    >>> from statphys.vis.dynamics import DynamicsPlotter
    >>> eqs = GaussianLinearMseEquations(rho=1.0, lr=0.5, reg_param=0.01)
    >>> plotter = DynamicsPlotter()
    >>> fig, ax = plotter.plot_flow_field(eqs, m_range=(0, 1.2), q_range=(0, 1.5))

"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

from statphys.simulation.base import SimulationResult
from statphys.vis.plotter import PlotStyle, Plotter


class DynamicsPlotter(Plotter):
    """
    Plotter for online-learning dynamics in the (m, q) plane.

    Works with any `OnlineEquations`-style object with the signature
    ``equations(t, y, params) -> np.ndarray`` where ``y = [m, q]``.
    """

    def __init__(self, style: PlotStyle | None = None):
        super().__init__(style)

    def _eval_field(
        self,
        equations: Any,
        M: np.ndarray,
        Q: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate (dm/dt, dq/dt) on a grid; NaN out failures."""
        dM = np.full_like(M, np.nan, dtype=float)
        dQ = np.full_like(Q, np.nan, dtype=float)
        for idx in np.ndindex(M.shape):
            try:
                deriv = equations(0.0, np.array([M[idx], Q[idx]]), params)
                dM[idx], dQ[idx] = float(deriv[0]), float(deriv[1])
            except Exception:
                pass
        return dM, dQ

    def plot_flow_field(
        self,
        equations: Any,
        m_range: tuple[float, float] = (0.0, 1.2),
        q_range: tuple[float, float] = (0.01, 1.5),
        n_grid: int = 20,
        params: dict[str, Any] | None = None,
        ax: Axes | None = None,
        show_nullclines: bool = True,
        show_speed: bool = True,
        streamlines: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot the ODE vector field in the (m, q) plane.

        Args:
            equations: Online ODE equations, callable as f(t, [m, q], params).
            m_range: Range of m values.
            q_range: Range of q values (avoid q = 0 exactly).
            n_grid: Grid resolution per axis.
            params: Parameter overrides passed to the equations.
            ax: Axes to plot on. Creates new if None.
            show_nullclines: Draw dm/dt = 0 and dq/dt = 0 contours.
            show_speed: Color the flow by speed |dy/dt|.
            streamlines: Use streamplot (True) or quiver (False).
            **kwargs: Extra arguments passed to streamplot/quiver.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        params = params or {}
        m_grid = np.linspace(m_range[0], m_range[1], n_grid)
        q_grid = np.linspace(q_range[0], q_range[1], n_grid)
        M, Q = np.meshgrid(m_grid, q_grid)

        dM, dQ = self._eval_field(equations, M, Q, params)
        speed = np.sqrt(dM**2 + dQ**2)

        if streamlines:
            color = speed if show_speed else self.style.colors[0]
            strm = ax.streamplot(
                M, Q, dM, dQ, color=color, cmap="viridis", density=1.2, **kwargs
            )
            if show_speed:
                cbar = fig.colorbar(strm.lines, ax=ax)
                cbar.set_label(r"flow speed $|\dot{y}|$")
        else:
            ax.quiver(M, Q, dM, dQ, speed if show_speed else None, cmap="viridis", **kwargs)

        if show_nullclines:
            with np.errstate(invalid="ignore"):
                cs_m = ax.contour(
                    M, Q, dM, levels=[0.0], colors=["tab:red"], linewidths=2
                )
                cs_q = ax.contour(
                    M, Q, dQ, levels=[0.0], colors=["tab:orange"], linewidths=2
                )
            # Label nullclines via proxy artists (contour has no label kwarg)
            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], color="tab:red", lw=2, label=r"$\dot{m} = 0$"),
                Line2D([0], [0], color="tab:orange", lw=2, label=r"$\dot{q} = 0$"),
            ]
            ax.legend(handles=handles, loc="best")
            del cs_m, cs_q

        self.set_labels(ax, xlabel=r"$m$", ylabel=r"$q$", title="Order-parameter flow")
        return fig, ax

    def plot_theory_trajectories(
        self,
        equations: Any,
        initial_conditions: list[tuple[float, float]],
        t_max: float = 20.0,
        params: dict[str, Any] | None = None,
        ax: Axes | None = None,
        mark_endpoints: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Integrate and plot theory trajectories in the (m, q) plane.

        Args:
            equations: Online ODE equations f(t, [m, q], params).
            initial_conditions: List of (m0, q0) starting points.
            t_max: Integration time.
            params: Parameter overrides for the equations.
            ax: Axes (e.g. from plot_flow_field) to overlay onto.
            mark_endpoints: Mark start (circle) and end (star) of each trajectory.
            **kwargs: Extra arguments for solve_ivp (e.g. method, rtol).

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        params = params or {}
        t_eval = np.linspace(0.0, t_max, 400)

        for i, (m0, q0) in enumerate(initial_conditions):
            color = self.style.colors[i % len(self.style.colors)]
            sol = solve_ivp(
                lambda t, y: equations(t, y, params),
                (0.0, t_max),
                [m0, q0],
                t_eval=t_eval,
                **kwargs,
            )
            ax.plot(sol.y[0], sol.y[1], color=color, linewidth=self.style.line_width)
            if mark_endpoints:
                ax.plot(m0, q0, "o", color=color, markersize=6)
                ax.plot(sol.y[0][-1], sol.y[1][-1], "*", color=color, markersize=14)

        self.set_labels(ax, xlabel=r"$m$", ylabel=r"$q$")
        self.add_grid(ax)
        return fig, ax

    def plot_experiment_trajectories(
        self,
        result: SimulationResult,
        m_index: int = 0,
        q_index: int = 1,
        ax: Axes | None = None,
        alpha: float = 0.5,
        show_mean: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot per-seed experimental trajectories in the (m, q) plane.

        Args:
            result: SimulationResult from OnlineSimulation.
            m_index: Index of m in the trajectory arrays.
            q_index: Index of q in the trajectory arrays.
            ax: Axes to overlay onto.
            alpha: Transparency for individual seed trajectories.
            show_mean: Also draw the seed-averaged trajectory in black.
            **kwargs: Extra plot arguments.

        Returns:
            Tuple of (Figure, Axes).

        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        all_traj = np.array(result.experiment_results["trajectories_all"])
        # shape: (n_seeds, n_times, n_params)
        for s in range(all_traj.shape[0]):
            ax.plot(
                all_traj[s, :, m_index],
                all_traj[s, :, q_index],
                color=self.style.colors[0],
                alpha=alpha,
                linewidth=1,
                **kwargs,
            )

        if show_mean:
            mean_traj = np.array(result.experiment_results["trajectories_mean"])
            ax.plot(
                mean_traj[:, m_index],
                mean_traj[:, q_index],
                color="black",
                linewidth=self.style.line_width,
                label="seed mean",
            )
            self.add_legend(ax)

        self.set_labels(ax, xlabel=r"$m$", ylabel=r"$q$")
        self.add_grid(ax)
        return fig, ax

    def plot_phase_portrait(
        self,
        equations: Any,
        result: SimulationResult | None = None,
        m_range: tuple[float, float] = (0.0, 1.2),
        q_range: tuple[float, float] = (0.01, 1.5),
        initial_conditions: list[tuple[float, float]] | None = None,
        t_max: float = 20.0,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        One-call phase portrait: flow field + nullclines + trajectories.

        Args:
            equations: Online ODE equations.
            result: Optional experiment result to overlay per-seed trajectories.
            m_range, q_range: Plot ranges.
            initial_conditions: Starting points for theory trajectories.
                Defaults to a small grid of corners.
            t_max: Theory integration time.
            params: Parameter overrides.
            **kwargs: Passed to plot_flow_field.

        Returns:
            Tuple of (Figure, Axes).

        """
        fig, ax = self.plot_flow_field(
            equations, m_range=m_range, q_range=q_range, params=params, **kwargs
        )

        if initial_conditions is None:
            initial_conditions = [
                (m_range[0] + 0.05 * (m_range[1] - m_range[0]), q_range[1] * 0.9),
                (m_range[0] + 0.05 * (m_range[1] - m_range[0]), q_range[0] + 0.05),
                (m_range[1] * 0.9, q_range[1] * 0.9),
            ]
        self.plot_theory_trajectories(
            equations, initial_conditions, t_max=t_max, params=params, ax=ax
        )

        if result is not None:
            self.plot_experiment_trajectories(result, ax=ax)

        return fig, ax
