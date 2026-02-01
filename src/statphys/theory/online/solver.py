"""ODE solver for online learning dynamics."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from statphys.theory.base import BaseTheory, TheoryResult, TheoryType


class ODESolver(BaseTheory):
    """
    ODE solver for online learning dynamics.

    In the high-dimensional limit, online learning dynamics
    are described by ODEs for order parameters.

    Example:
        >>> def ode_rhs(t, y, params):
        ...     m, q = y
        ...     dm_dt = ...
        ...     dq_dt = ...
        ...     return [dm_dt, dq_dt]
        >>>
        >>> solver = ODESolver(equations=ode_rhs, order_params=['m', 'q'])
        >>> result = solver.solve(t_span=(0, 10), t_eval=np.linspace(0, 10, 100))

    """

    def __init__(
        self,
        equations: Callable[[float, np.ndarray, dict], np.ndarray],
        order_params: list[str],
        method: str = "RK45",
        tol: float = 1e-8,
        max_step: float = 0.1,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize ODESolver.

        Args:
            equations: ODE right-hand side function.
                      Signature: equations(t, y, params) -> dy_dt
                      where y and dy_dt are arrays of order parameters.
            order_params: Names of order parameters.
            method: Integration method ('RK45', 'RK23', 'Radau', 'BDF', 'LSODA').
            tol: Relative tolerance.
            max_step: Maximum step size.
            verbose: Whether to print progress.

        """
        super().__init__(tol=tol, verbose=verbose, **kwargs)

        self.equations = equations
        self.order_params = order_params
        self.n_params = len(order_params)
        self.method = method
        self.max_step = max_step

    def solve(
        self,
        t_span: tuple[float, float],
        init_values: tuple[float, ...] | None = None,
        t_eval: np.ndarray | None = None,
        n_points: int = 100,
        **params: Any,
    ) -> TheoryResult:
        """
        Solve the ODE system.

        Args:
            t_span: Time span (t_start, t_end).
            init_values: Initial values for order parameters.
            t_eval: Time points to evaluate at. If None, uses linspace.
            n_points: Number of evaluation points if t_eval is None.
            **params: Parameters passed to equations.

        Returns:
            TheoryResult containing solution trajectories.

        """
        if init_values is None:
            # Default initialization
            init_values = tuple([0.0] * self.n_params)

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Wrapper for scipy
        def ode_func(t, y):
            return self.equations(t, y, params)

        if self.verbose:
            print(f"Solving ODE from t={t_span[0]} to t={t_span[1]}")

        # Solve ODE
        solution = solve_ivp(
            ode_func,
            t_span,
            init_values,
            method=self.method,
            t_eval=t_eval,
            rtol=self.tol,
            atol=self.tol * 1e-3,
            max_step=self.max_step,
        )

        # Check success
        converged = [solution.success] * len(t_eval)
        iterations = [0] * len(t_eval)  # Not applicable for ODE

        # Extract order parameters
        order_params_dict = {}
        for i, name in enumerate(self.order_params):
            order_params_dict[name] = solution.y[i].tolist()

        return TheoryResult(
            theory_type=TheoryType.ONLINE,
            order_params=order_params_dict,
            param_values=t_eval.tolist(),
            converged=converged,
            iterations=iterations,
            metadata={
                "method": self.method,
                "params": params,
                "success": solution.success,
                "message": solution.message,
            },
        )

    def solve_with_generalization_error(
        self,
        t_span: tuple[float, float],
        eg_formula: Callable[..., float],
        init_values: tuple[float, ...] | None = None,
        t_eval: np.ndarray | None = None,
        **params: Any,
    ) -> TheoryResult:
        """
        Solve and compute generalization error trajectory.

        Args:
            t_span: Time span.
            eg_formula: Function to compute E_g from order params.
            init_values: Initial order parameter values.
            t_eval: Evaluation time points.
            **params: Parameters.

        Returns:
            TheoryResult with 'eg' trajectory added.

        """
        result = self.solve(t_span, init_values, t_eval, **params)

        # Compute generalization error at each time
        n_points = len(result.param_values)
        eg_list = []

        for i in range(n_points):
            op_values = {name: result.order_params[name][i] for name in self.order_params}
            eg = eg_formula(**op_values, t=result.param_values[i], **params)
            eg_list.append(eg)

        result.order_params["eg"] = eg_list
        return result

    def solve_multiple_lr(
        self,
        t_span: tuple[float, float],
        learning_rates: list[float],
        init_values: tuple[float, ...] | None = None,
        t_eval: np.ndarray | None = None,
        **params: Any,
    ) -> dict[float, TheoryResult]:
        """
        Solve for multiple learning rates.

        Args:
            t_span: Time span.
            learning_rates: List of learning rates to try.
            init_values: Initial values.
            t_eval: Evaluation times.
            **params: Other parameters.

        Returns:
            Dictionary mapping learning rate to TheoryResult.

        """
        results = {}

        for lr in learning_rates:
            if self.verbose:
                print(f"Solving for lr = {lr}")

            result = self.solve(t_span, init_values, t_eval, lr=lr, **params)
            results[lr] = result

        return results

    def get_theory_type(self) -> TheoryType:
        """Return theory type."""
        return TheoryType.ONLINE

    def get_config(self) -> dict[str, Any]:
        """Get solver configuration."""
        config = super().get_config()
        config.update(
            {
                "order_params": self.order_params,
                "method": self.method,
                "max_step": self.max_step,
            }
        )
        return config


class AdaptiveODESolver(ODESolver):
    """
    ODE solver with adaptive time stepping and event detection.

    Useful for detecting phase transitions and stopping conditions.
    """

    def __init__(
        self,
        equations: Callable[[float, np.ndarray, dict], np.ndarray],
        order_params: list[str],
        events: list[Callable] | None = None,
        method: str = "RK45",
        tol: float = 1e-8,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize AdaptiveODESolver.

        Args:
            equations: ODE right-hand side.
            order_params: Order parameter names.
            events: Event functions for stopping conditions.
            method: Integration method.
            tol: Tolerance.
            verbose: Verbosity.

        """
        super().__init__(
            equations=equations,
            order_params=order_params,
            method=method,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )
        self.events = events

    def solve(
        self,
        t_span: tuple[float, float],
        init_values: tuple[float, ...] | None = None,
        t_eval: np.ndarray | None = None,
        n_points: int = 100,
        **params: Any,
    ) -> TheoryResult:
        """Solve with event detection."""
        if init_values is None:
            init_values = tuple([0.0] * self.n_params)

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def ode_func(t, y):
            return self.equations(t, y, params)

        # Prepare events
        event_funcs = None
        if self.events:
            event_funcs = []
            for event in self.events:

                def wrapped_event(t, y, event=event):
                    return event(t, y, params)

                wrapped_event.terminal = True
                event_funcs.append(wrapped_event)

        solution = solve_ivp(
            ode_func,
            t_span,
            init_values,
            method=self.method,
            t_eval=t_eval,
            rtol=self.tol,
            atol=self.tol * 1e-3,
            events=event_funcs,
            dense_output=True,
        )

        # Handle early termination
        if solution.t_events and any(len(te) > 0 for te in solution.t_events):
            actual_t_eval = solution.t
        else:
            actual_t_eval = t_eval

        converged = [solution.success] * len(actual_t_eval)
        iterations = [0] * len(actual_t_eval)

        order_params_dict = {}
        for i, name in enumerate(self.order_params):
            order_params_dict[name] = solution.y[i].tolist()

        return TheoryResult(
            theory_type=TheoryType.ONLINE,
            order_params=order_params_dict,
            param_values=(
                actual_t_eval.tolist()
                if isinstance(actual_t_eval, np.ndarray)
                else list(actual_t_eval)
            ),
            converged=converged,
            iterations=iterations,
            metadata={
                "method": self.method,
                "params": params,
                "success": solution.success,
                "message": solution.message,
                "t_events": (
                    [te.tolist() for te in solution.t_events] if solution.t_events else []
                ),
            },
        )
