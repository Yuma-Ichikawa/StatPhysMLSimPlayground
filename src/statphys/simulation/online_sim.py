"""
Online learning simulation.
"""

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import copy

from statphys.simulation.base import BaseSimulation, SimulationResult
from statphys.simulation.config import SimulationConfig, TheoryType
from statphys.utils.seed import fix_seed


class OnlineSimulation(BaseSimulation):
    """
    Simulation for online learning dynamics.

    Runs online SGD and tracks order parameters over time,
    comparing to theoretical ODE predictions.

    Example:
        >>> config = SimulationConfig.for_online(t_max=10.0, t_steps=100)
        >>> sim = OnlineSimulation(config)
        >>> results = sim.run(dataset, LinearRegression, RidgeLoss(0.01))
    """

    def run(
        self,
        dataset: Any,
        model_class: Type[nn.Module],
        loss_fn: Callable,
        calc_order_params: Optional[Callable] = None,
        theory_solver: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Run online learning simulation.

        Args:
            dataset: Dataset instance with generate_sample method.
            model_class: Model class to instantiate.
            loss_fn: Loss function.
            calc_order_params: Function to compute order parameters.
            theory_solver: ODE solver for comparison.
            **kwargs: Additional arguments.

        Returns:
            SimulationResult with experiment and theory results.
        """
        d = dataset.d
        t_values = self.config.get_time_values()
        n_epochs = int(d * self.config.t_max)  # Total number of samples
        seed_list = self.config.seed_list

        # Evaluation points (in sample indices)
        eval_indices = (t_values * d).astype(int)
        eval_indices = np.clip(eval_indices, 0, n_epochs)

        # Default order params calculator
        if calc_order_params is None:
            calc_order_params = self._default_calc_order_params

        # Storage for results
        all_trajectories = []  # (n_seeds, n_time_points, n_params)

        for seed_idx, seed in enumerate(seed_list):
            self._print_progress(f"【TRIAL {seed_idx + 1}/{len(seed_list)}, SEED {seed}】")

            trajectory = self._run_single_seed(
                dataset=dataset,
                model_class=model_class,
                loss_fn=loss_fn,
                calc_order_params=calc_order_params,
                n_epochs=n_epochs,
                eval_indices=eval_indices,
                seed=seed,
                **kwargs,
            )
            all_trajectories.append(trajectory)

        # Convert to numpy
        all_trajectories = np.array(all_trajectories)  # (n_seeds, n_times, n_params)

        experiment_results = {
            "t_values": t_values.tolist(),
            "trajectories_all": all_trajectories.tolist(),
            "trajectories_mean": np.mean(all_trajectories, axis=0).tolist(),
            "trajectories_std": np.std(all_trajectories, axis=0).tolist(),
        }

        # Run theory if requested
        theory_results = None
        if self.config.use_theory and theory_solver is not None:
            self._print_progress("Computing theory predictions...")
            teacher_params = dataset.get_teacher_params()

            theory_results = theory_solver.solve(
                t_span=(0, self.config.t_max),
                t_eval=t_values,
                rho=teacher_params.get("rho", 1.0),
                eta_noise=teacher_params.get("eta", 0.0),
                lr=self.config.lr,
                reg_param=self.config.reg_param,
            )

        return SimulationResult(
            theory_type=TheoryType.ONLINE,
            experiment_results=experiment_results,
            theory_results=theory_results,
            config=self.config,
            metadata={
                "d": d,
                "n_epochs": n_epochs,
                "dataset_config": dataset.get_config(),
            },
        )

    def _run_single_seed(
        self,
        dataset: Any,
        model_class: Type[nn.Module],
        loss_fn: Callable,
        calc_order_params: Callable,
        n_epochs: int,
        eval_indices: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Run online learning for a single seed.

        Returns:
            List of order parameters at each evaluation point.
        """
        fix_seed(seed)

        d = dataset.d

        # Initialize model
        model = model_class(d=d).to(self.device)

        # Use SGD for online learning
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)

        # Track order parameters
        trajectory = []
        eval_idx = 0

        # Initial state
        if 0 in eval_indices:
            order_params = calc_order_params(dataset, model)
            if isinstance(order_params, dict):
                order_params = list(order_params.values())
            trajectory.append(order_params)
            eval_idx = 1

        model.train()

        for epoch in range(n_epochs):
            # Generate single sample
            x_sample, y_sample = dataset.generate_sample()
            x_sample = x_sample.unsqueeze(0).to(self.device)
            y_sample = y_sample.unsqueeze(0).to(self.device) if y_sample.dim() == 0 else y_sample.to(self.device)

            # SGD step
            optimizer.zero_grad()
            y_pred = model(x_sample)
            loss = loss_fn(y_pred, y_sample, model, online=True)
            loss.backward()
            optimizer.step()

            # Record if at evaluation point
            if eval_idx < len(eval_indices) and (epoch + 1) >= eval_indices[eval_idx]:
                order_params = calc_order_params(dataset, model)
                if isinstance(order_params, dict):
                    order_params = list(order_params.values())
                trajectory.append(order_params)
                eval_idx += 1

            # Verbose
            if (epoch + 1) % (self.config.verbose_interval * d) == 0:
                t = (epoch + 1) / d
                self._print_progress(f"  t={t:.2f}, loss={loss.item():.6f}")

        return trajectory

    def _default_calc_order_params(
        self,
        dataset: Any,
        model: nn.Module,
    ) -> List[float]:
        """Default order parameter calculation."""
        teacher_params = dataset.get_teacher_params()
        W0 = teacher_params.get("W0")
        rho = teacher_params.get("rho", 1.0)
        d = dataset.d

        if hasattr(model, "W"):
            w = model.W
        elif hasattr(model, "get_weight_vector"):
            w = model.get_weight_vector().reshape(-1, 1)
        else:
            raise ValueError("Model must have W attribute or get_weight_vector method")

        m = (w.T @ W0 / d).item() if W0 is not None else 0.0
        q = (w.T @ w / d).item()
        eg = 0.5 * (rho - 2 * m + q)

        return [m, q, eg]
