"""Saddle-point equation solver for replica calculations."""

from collections.abc import Callable
from typing import Any

import numpy as np

from statphys.theory.base import BaseTheory, TheoryResult, TheoryType


class SaddlePointSolver(BaseTheory):
    """
    Solver for saddle-point equations in replica calculations.

    This solver handles:
    - Fixed-point iteration with damping
    - Automatic damping adjustment
    - Multiple starting points for robustness
    - Tracking convergence history

    Example:
        >>> def equations(m, q, alpha, **params):
        ...     # Your saddle-point equations
        ...     new_m = ...
        ...     new_q = ...
        ...     return new_m, new_q
        >>>
        >>> solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])
        >>> result = solver.solve(alpha_values=[0.1, 0.5, 1.0, 2.0], rho=1.0, eta=0.0)

    """

    def __init__(
        self,
        equations: Callable[..., tuple[float, ...]],
        order_params: list[str],
        damping: float = 0.5,
        adaptive_damping: bool = True,
        damping_decay: float = 0.9,
        min_damping: float = 0.01,
        tol: float = 1e-8,
        max_iter: int = 10000,
        n_restarts: int = 3,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize SaddlePointSolver.

        Args:
            equations: Callable that takes current order params and returns updated values.
                      Signature: equations(m, q, ..., alpha, **params) -> (new_m, new_q, ...)
            order_params: Names of order parameters (e.g., ['m', 'q']).
            damping: Initial damping factor (0 < damping <= 1).
            adaptive_damping: Whether to adaptively reduce damping.
            damping_decay: Factor to reduce damping when oscillating.
            min_damping: Minimum allowed damping.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            n_restarts: Number of random restarts for robustness.
            verbose: Whether to print progress.

        """
        super().__init__(tol=tol, max_iter=max_iter, verbose=verbose, **kwargs)

        self.equations = equations
        self.order_params = order_params
        self.n_params = len(order_params)
        self.damping = damping
        self.adaptive_damping = adaptive_damping
        self.damping_decay = damping_decay
        self.min_damping = min_damping
        self.n_restarts = n_restarts

    def _fixed_point_iteration(
        self,
        init_values: tuple[float, ...],
        alpha: float,
        params: dict[str, Any],
    ) -> tuple[tuple[float, ...], bool, int]:
        """
        Run fixed-point iteration with damping.

        Args:
            init_values: Initial values for order parameters.
            alpha: Sample ratio parameter.
            params: Additional parameters for equations.

        Returns:
            Tuple of (final_values, converged, iterations).

        """
        values = np.array(init_values)
        damping = self.damping
        prev_error = float("inf")
        oscillation_count = 0

        for iteration in range(self.max_iter):
            # Compute new values
            new_values = np.array(self.equations(*values, alpha=alpha, **params))

            # Damped update
            updated_values = damping * new_values + (1 - damping) * values

            # Check convergence
            error = np.max(np.abs(updated_values - values))

            if error < self.tol:
                if self.verbose:
                    print(f"  α={alpha:.3f}: Converged at iteration {iteration + 1}")
                return tuple(updated_values), True, iteration + 1

            # Adaptive damping: reduce if oscillating
            if self.adaptive_damping and error > prev_error:
                oscillation_count += 1
                if oscillation_count >= 3:
                    damping = max(self.min_damping, damping * self.damping_decay)
                    oscillation_count = 0
                    if self.verbose:
                        print(f"  Reducing damping to {damping:.4f}")
            else:
                oscillation_count = 0

            prev_error = error
            values = updated_values

        if self.verbose:
            print(f"  α={alpha:.3f}: Did not converge (error={error:.2e})")
        return tuple(values), False, self.max_iter

    def _solve_single_alpha(
        self,
        alpha: float,
        init_values: tuple[float, ...] | None = None,
        params: dict[str, Any] = None,
    ) -> tuple[dict[str, float], bool, int]:
        """
        Solve for a single alpha value.

        Args:
            alpha: Sample ratio parameter.
            init_values: Initial values. If None, uses random restarts.
            params: Additional parameters.

        Returns:
            Tuple of (order_params_dict, converged, iterations).

        """
        params = params or {}
        best_result = None
        best_error = float("inf")

        # Generate initial values for restarts
        if init_values is not None:
            init_list = [init_values]
        else:
            init_list = []
            for _ in range(self.n_restarts):
                # Random initialization
                init = tuple(np.random.uniform(0.1, 1.0, self.n_params))
                init_list.append(init)

        for init in init_list:
            values, converged, iterations = self._fixed_point_iteration(init, alpha, params)

            if converged:
                result = dict(zip(self.order_params, values, strict=False))
                return result, True, iterations

            # Track best non-converged result
            error = np.max(
                np.abs(np.array(self.equations(*values, alpha=alpha, **params)) - np.array(values))
            )
            if error < best_error:
                best_error = error
                best_result = values

        # Return best non-converged result
        result = dict(zip(self.order_params, best_result, strict=False))
        return result, False, self.max_iter

    def solve(
        self,
        alpha_values: list[float] | np.ndarray,
        init_values: tuple[float, ...] | None = None,
        use_continuation: bool = True,
        **params: Any,
    ) -> TheoryResult:
        """
        Solve saddle-point equations for a range of alpha values.

        Args:
            alpha_values: List of sample ratio values.
            init_values: Initial values for first alpha.
            use_continuation: Use solution from previous alpha as init for next.
            **params: Additional parameters passed to equations.

        Returns:
            TheoryResult containing solutions for all alpha values.

        """
        alpha_values = np.array(alpha_values)
        n_alphas = len(alpha_values)

        # Initialize result storage
        order_params_list = {name: [] for name in self.order_params}
        converged_list = []
        iterations_list = []

        current_init = init_values

        for i, alpha in enumerate(alpha_values):
            if self.verbose:
                print(f"Solving α = {alpha:.4f} ({i + 1}/{n_alphas})")

            result, converged, iterations = self._solve_single_alpha(
                alpha, init_values=current_init, params=params
            )

            # Store results
            for name in self.order_params:
                order_params_list[name].append(result[name])
            converged_list.append(converged)
            iterations_list.append(iterations)

            # Update initial values for continuation
            if use_continuation:
                current_init = tuple(result[name] for name in self.order_params)

        return TheoryResult(
            theory_type=TheoryType.REPLICA,
            order_params=order_params_list,
            param_values=alpha_values.tolist(),
            converged=converged_list,
            iterations=iterations_list,
            metadata={
                "damping": self.damping,
                "params": params,
            },
        )

    def solve_with_generalization_error(
        self,
        alpha_values: list[float] | np.ndarray,
        eg_formula: Callable[..., float],
        **params: Any,
    ) -> TheoryResult:
        """
        Solve and compute generalization error.

        Args:
            alpha_values: List of alpha values.
            eg_formula: Function to compute E_g from order params.
            **params: Parameters for equations.

        Returns:
            TheoryResult with 'eg' added to order_params.

        """
        result = self.solve(alpha_values, **params)

        # Compute generalization error
        eg_list = []
        for i in range(len(alpha_values)):
            op_values = {name: result.order_params[name][i] for name in self.order_params}
            eg = eg_formula(**op_values, alpha=alpha_values[i], **params)
            eg_list.append(eg)

        result.order_params["eg"] = eg_list
        return result

    def get_theory_type(self) -> TheoryType:
        """Return the theory type."""
        return TheoryType.REPLICA

    def get_config(self) -> dict[str, Any]:
        """Get solver configuration."""
        config = super().get_config()
        config.update(
            {
                "order_params": self.order_params,
                "damping": self.damping,
                "adaptive_damping": self.adaptive_damping,
                "n_restarts": self.n_restarts,
            }
        )
        return config
