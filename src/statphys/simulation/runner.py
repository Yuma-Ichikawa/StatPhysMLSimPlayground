"""Unified simulation runner."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from statphys.simulation.base import SimulationResult
from statphys.simulation.config import SimulationConfig, TheoryType
from statphys.simulation.online_sim import OnlineSimulation
from statphys.simulation.replica_sim import ReplicaSimulation


class SimulationRunner:
    """
    Unified interface for running simulations.

    Automatically selects the appropriate simulation type
    based on configuration.

    Example:
        >>> runner = SimulationRunner()
        >>> result = runner.run(
        ...     config=SimulationConfig.for_replica(),
        ...     dataset=dataset,
        ...     model_class=LinearRegression,
        ...     loss_fn=RidgeLoss(0.01)
        ... )

    """

    def __init__(self):
        """Initialize SimulationRunner."""
        self._simulations = {
            TheoryType.REPLICA: ReplicaSimulation,
            TheoryType.ONLINE: OnlineSimulation,
        }

    def run(
        self,
        config: SimulationConfig,
        dataset: Any,
        model_class: type[nn.Module],
        loss_fn: Callable,
        calc_order_params: Callable | None = None,
        theory_solver: Any | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Run simulation based on configuration.

        Args:
            config: Simulation configuration.
            dataset: Dataset instance.
            model_class: Model class.
            loss_fn: Loss function.
            calc_order_params: Order parameter calculator.
            theory_solver: Theory solver for comparison.
            **kwargs: Additional arguments.

        Returns:
            SimulationResult.

        """
        # Get appropriate simulation class
        sim_class = self._simulations.get(config.theory_type)
        if sim_class is None:
            raise ValueError(f"Unsupported theory type: {config.theory_type}")

        # Create and run simulation
        simulation = sim_class(config)
        return simulation.run(
            dataset=dataset,
            model_class=model_class,
            loss_fn=loss_fn,
            calc_order_params=calc_order_params,
            theory_solver=theory_solver,
            **kwargs,
        )

    def run_comparison(
        self,
        config: SimulationConfig,
        dataset: Any,
        model_classes: dict[str, type[nn.Module]],
        loss_fn: Callable,
        **kwargs: Any,
    ) -> dict[str, SimulationResult]:
        """
        Run simulations for multiple model classes.

        Useful for comparing different architectures.

        Args:
            config: Simulation configuration.
            dataset: Dataset instance.
            model_classes: Dictionary mapping names to model classes.
            loss_fn: Loss function.
            **kwargs: Additional arguments.

        Returns:
            Dictionary mapping model names to results.

        """
        results = {}

        for name, model_class in model_classes.items():
            print(f"\n{'='*50}")
            print(f"Running simulation for: {name}")
            print(f"{'='*50}\n")

            results[name] = self.run(
                config=config,
                dataset=dataset,
                model_class=model_class,
                loss_fn=loss_fn,
                **kwargs,
            )

        return results

    def run_parameter_sweep(
        self,
        base_config: SimulationConfig,
        dataset: Any,
        model_class: type[nn.Module],
        loss_fn: Callable,
        param_name: str,
        param_values: list,
        **kwargs: Any,
    ) -> dict[Any, SimulationResult]:
        """
        Run simulations across parameter sweep.

        Args:
            base_config: Base configuration.
            dataset: Dataset instance.
            model_class: Model class.
            loss_fn: Loss function.
            param_name: Name of parameter to sweep.
            param_values: Values to sweep over.
            **kwargs: Additional arguments.

        Returns:
            Dictionary mapping parameter values to results.

        """
        results = {}

        for value in param_values:
            print(f"\n{'='*50}")
            print(f"Running simulation for {param_name}={value}")
            print(f"{'='*50}\n")

            # Create modified config
            config_dict = base_config.to_dict()
            config_dict[param_name] = value
            config = SimulationConfig.from_dict(config_dict)

            results[value] = self.run(
                config=config,
                dataset=dataset,
                model_class=model_class,
                loss_fn=loss_fn,
                **kwargs,
            )

        return results
