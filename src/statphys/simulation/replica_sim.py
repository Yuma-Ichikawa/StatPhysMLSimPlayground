"""Replica simulation for static analysis experiments."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from statphys.simulation.base import BaseSimulation, SimulationResult
from statphys.simulation.config import TheoryType
from statphys.utils.seed import fix_seed
from statphys.utils.order_params import OrderParameterCalculator, ModelType


class ReplicaSimulation(BaseSimulation):
    """
    Simulation for static (replica) analysis.

    Runs gradient descent to convergence for various values of α = n/d,
    and compares to replica theory predictions.

    Example:
        >>> config = SimulationConfig.for_replica(
        ...     alpha_range=(0.1, 5.0),
        ...     alpha_steps=20,
        ...     n_seeds=5
        ... )
        >>> sim = ReplicaSimulation(config)
        >>> results = sim.run(dataset, LinearRegression, RidgeLoss(0.01))

    """

    def run(
        self,
        dataset: Any,
        model_class: type[nn.Module],
        loss_fn: Callable,
        calc_order_params: Callable | None = None,
        theory_solver: Any | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Run replica simulation.

        Args:
            dataset: Dataset instance with generate_dataset method.
            model_class: Model class to instantiate.
            loss_fn: Loss function.
            calc_order_params: Function to compute order parameters.
                              Signature: calc_order_params(dataset, model) -> dict
                              If None and config.auto_order_params=True, uses OrderParameterCalculator.
            theory_solver: Replica theory solver for comparison.
            **kwargs: Additional arguments passed to training.

        Returns:
            SimulationResult with experiment and theory results.

        """
        alpha_values = self.config.get_alpha_values()
        seed_list = self.config.seed_list
        d = dataset.d

        # Storage for results across seeds
        all_order_params = []  # Shape: (n_seeds, n_alphas, n_params)
        all_train_losses = []
        all_converged = []

        # Order params calculator setup
        if calc_order_params is None:
            if self.config.auto_order_params:
                # Use automatic order parameter calculator
                calc_order_params = self._setup_auto_order_params(dataset, model_class)
            else:
                calc_order_params = self._default_calc_order_params

        for seed_idx, seed in enumerate(seed_list):
            self._print_progress(f"【TRIAL {seed_idx + 1}/{len(seed_list)}, SEED {seed}】")

            seed_order_params = []
            seed_train_losses = []
            seed_converged = []

            for alpha in alpha_values:
                n_samples = int(d * alpha)

                # Train model
                result = self._train_single_alpha(
                    dataset=dataset,
                    model_class=model_class,
                    loss_fn=loss_fn,
                    calc_order_params=calc_order_params,
                    n_samples=n_samples,
                    seed=seed,
                    alpha=alpha,
                    **kwargs,
                )

                seed_order_params.append(result["order_params"])
                seed_train_losses.append(result["train_loss"])
                seed_converged.append(result["converged"])

            all_order_params.append(seed_order_params)
            all_train_losses.append(seed_train_losses)
            all_converged.append(seed_converged)

        # Convert to numpy arrays
        all_order_params = np.array(all_order_params)  # (n_seeds, n_alphas, n_params)
        all_train_losses = np.array(all_train_losses)

        # Extract individual order parameters
        experiment_results = {
            "alpha_values": alpha_values.tolist(),
            "order_params_all": all_order_params.tolist(),
            "order_params_mean": np.mean(all_order_params, axis=0).tolist(),
            "order_params_std": np.std(all_order_params, axis=0).tolist(),
            "train_loss_mean": np.mean(all_train_losses, axis=0).tolist(),
            "train_loss_std": np.std(all_train_losses, axis=0).tolist(),
            "converged": all_converged,
        }

        # Run theory if requested
        theory_results = None
        if self.config.use_theory and theory_solver is not None:
            self._print_progress("Computing theory predictions...")
            theory_results = theory_solver.solve(
                alpha_values=alpha_values,
                rho=dataset.get_teacher_params().get("rho", 1.0),
                eta=dataset.get_teacher_params().get("eta", 0.0),
                reg_param=self.config.reg_param,
            )

        return SimulationResult(
            theory_type=TheoryType.REPLICA,
            experiment_results=experiment_results,
            theory_results=theory_results,
            config=self.config,
            metadata={
                "d": d,
                "dataset_config": dataset.get_config(),
            },
        )

    def _train_single_alpha(
        self,
        dataset: Any,
        model_class: type[nn.Module],
        loss_fn: Callable,
        calc_order_params: Callable,
        n_samples: int,
        seed: int,
        alpha: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train model for a single alpha value.

        Returns:
            Dictionary with 'order_params', 'train_loss', 'converged'.

        """
        fix_seed(seed)

        d = dataset.d

        # Generate training data
        X_data, y_data = dataset.generate_dataset(n_samples)
        X_data = X_data.to(self.device)
        y_data = y_data.to(self.device)

        # Initialize model
        model = model_class(d=d).to(self.device)
        optimizer = self._get_optimizer(model, self.config.lr)

        # Training loop
        model.train()
        prev_loss = float("inf")
        no_improvement_count = 0
        converged = False

        for iteration in range(self.config.max_iter):
            optimizer.zero_grad()
            y_pred = model(X_data)
            # Use for_replica for proper scaling: L = (1/n)Σℓ + λ||w||² (O(d))
            loss = loss_fn.for_replica(y_pred, y_data, model)
            loss.backward()
            optimizer.step()

            # Check convergence (loss_value is O(d) due to regularization)
            loss_value = loss.item() / n_samples
            if abs(prev_loss - loss_value) < self.config.tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= self.config.patience:
                converged = True
                self._print_progress(f"  α={alpha:.2f}: Converged at iteration {iteration + 1}")
                break

            prev_loss = loss_value

            # Verbose output
            if (iteration + 1) % self.config.verbose_interval == 0:
                self._print_progress(
                    f"  α={alpha:.2f}, iter={iteration + 1}, loss={loss_value:.6f}"
                )

        # Compute final order parameters
        model.eval()
        with torch.no_grad():
            order_params = calc_order_params(dataset, model)

        # Convert to list if dict
        if isinstance(order_params, dict):
            order_params_list = list(order_params.values())
        else:
            order_params_list = order_params

        return {
            "order_params": order_params_list,
            "train_loss": loss_value,
            "converged": converged,
        }

    def _setup_auto_order_params(
        self,
        dataset: Any,
        model_class: type[nn.Module],
    ) -> Callable:
        """
        Setup automatic order parameter calculator and print info.

        Args:
            dataset: Dataset instance.
            model_class: Model class.

        Returns:
            Callable for order parameter calculation.
        """
        # Create a temporary model to detect type
        temp_model = model_class(d=dataset.d)
        calculator = OrderParameterCalculator(return_format="list", verbose=False)

        # Detect model and task types
        model_type = calculator._detect_model_type(temp_model)
        task_type = calculator._detect_task_type(dataset)

        # Print information about what will be calculated
        self._print_progress("=" * 60)
        self._print_progress("【AUTO ORDER PARAMETER CALCULATION ENABLED】")
        self._print_progress("=" * 60)
        self._print_progress(f"  Model Type: {model_type.value}")
        self._print_progress(f"  Task Type:  {task_type.value}")
        self._print_progress("")
        self._print_progress("  Order Parameters to be computed:")

        param_names = OrderParameterCalculator.get_param_names(model_type)
        param_descriptions = {
            "m": "Student-Teacher overlap (M = W^T @ W0 / d)",
            "q": "Student self-overlap (Q = W^T @ W / d)",
            "eg": "Generalization error (E_g)",
            "m_avg": "Average Student-Teacher overlap",
            "q_diag_avg": "Average diagonal of Q matrix",
            "q_offdiag_avg": "Average off-diagonal of Q matrix",
            "a_norm": "Second-layer weight norm",
        }

        for i, name in enumerate(param_names):
            desc = param_descriptions.get(name, name)
            self._print_progress(f"    [{i}] {name}: {desc}")

        self._print_progress("")
        self._print_progress("  Additional computed quantities:")
        self._print_progress("    - All Student-Teacher overlaps (M matrix)")
        self._print_progress("    - All Student self-overlaps (Q matrix)")
        self._print_progress("    - Teacher self-overlaps (R matrix)")
        self._print_progress("    - O(1) scalars (bias, second-layer weights)")
        self._print_progress("=" * 60)
        self._print_progress("")

        # Clean up temporary model
        del temp_model

        return calculator

    def _default_calc_order_params(
        self,
        dataset: Any,
        model: nn.Module,
    ) -> list[float]:
        """
        Default order parameter calculation for linear models.

        Computes m, q, and E_g.
        """
        teacher_params = dataset.get_teacher_params()
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)
        d = dataset.d

        # Get model weights
        if hasattr(model, "W"):
            w = model.W
        elif hasattr(model, "get_weight_vector"):
            w = model.get_weight_vector().reshape(-1, 1)
        else:
            raise ValueError("Model must have W attribute or get_weight_vector method")

        # Compute overlaps
        m = (w.T @ W0 / d).item() if W0 is not None else 0.0
        q = (w.T @ w / d).item()

        # Generalization error
        eg = 0.5 * (rho - 2 * m + q)

        return [m, q, eg]
