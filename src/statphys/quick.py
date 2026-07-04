"""
One-liner high-level API for common workflows.

These helpers wire together dataset / model / loss / theory / simulation /
visualization so that standard experiments are a single call:

    >>> import statphys
    >>> result = statphys.quick_online(d=400, lr=0.5, t_max=10)   # runs + plots
    >>> result = statphys.quick_replica(d=200, reg_param=0.1)     # runs + plots
    >>> res = statphys.quick_experiment("random_mlp", alphas=[1, 2, 4, 8])

Every helper returns the underlying result object, so the quick API is
also a gentle entry point into the full framework.

"""

from typing import Any

import numpy as np


def quick_online(
    d: int = 400,
    lr: float = 0.5,
    reg_param: float = 0.0,
    noise: float = 0.0,
    t_max: float = 10.0,
    t_steps: int = 51,
    n_seeds: int = 3,
    plot: bool = True,
    show: bool = False,
    verbose: bool = False,
    **kwargs: Any,
):
    """
    Online SGD for linear regression vs the exact ODE theory.

    Args:
        d: Input dimension.
        lr: Learning rate (equals the ODE eta).
        reg_param: L2 regularization lambda.
        noise: Teacher output noise variance.
        t_max: Maximum normalized time.
        t_steps: Number of time points.
        n_seeds: Number of seeds.
        plot: Draw the comparison figure.
        show: Call plt.show().
        verbose: Print progress.
        **kwargs: Extra SimulationConfig options.

    Returns:
        SimulationResult with theory attached.

    """
    from statphys.dataset import GaussianDataset
    from statphys.loss import RidgeLoss
    from statphys.model import LinearRegression
    from statphys.simulation import OnlineSimulation, SimulationConfig
    from statphys.theory.online import GaussianLinearMseEquations, ODESolver

    config = SimulationConfig.for_online(
        t_max=t_max,
        t_steps=t_steps,
        n_seeds=n_seeds,
        lr=lr,
        reg_param=reg_param,
        verbose=verbose,
        **kwargs,
    )
    dataset = GaussianDataset(d=d, rho=1.0, eta=noise)
    eqs = GaussianLinearMseEquations(rho=1.0, eta_noise=noise, lr=lr, reg_param=reg_param)
    solver = ODESolver(equations=eqs, order_params=["m", "q"])

    sim = OnlineSimulation(config)
    result = sim.run(dataset, LinearRegression, RidgeLoss(reg_param=reg_param), theory_solver=solver)

    if plot:
        from statphys.vis import plot_from_online_results

        plot_from_online_results(result, plot_type="all", show=show)
    return result


def quick_replica(
    d: int = 200,
    reg_param: float = 0.1,
    noise: float = 0.1,
    alphas: list[float] | np.ndarray | None = None,
    n_seeds: int = 3,
    plot: bool = True,
    show: bool = False,
    verbose: bool = False,
    **kwargs: Any,
):
    """
    Ridge regression at fixed alpha values vs replica theory.

    Args:
        d: Input dimension.
        reg_param: Ridge lambda.
        noise: Teacher output noise variance.
        alphas: Sample ratios; defaults to a log-spaced grid.
        n_seeds: Number of seeds.
        plot: Draw the comparison figure.
        show: Call plt.show().
        verbose: Print progress.
        **kwargs: Extra SimulationConfig options.

    Returns:
        SimulationResult with theory attached.

    """
    from statphys.dataset import GaussianDataset
    from statphys.loss import RidgeLoss
    from statphys.model import LinearRegression
    from statphys.simulation import ReplicaSimulation, SimulationConfig
    from statphys.theory.replica import GaussianLinearRidgeEquations, SaddlePointSolver

    if alphas is None:
        alphas = np.round(np.logspace(np.log10(0.5), np.log10(8), 7), 3).tolist()

    config = SimulationConfig.for_replica(
        alpha_values=list(alphas),
        n_seeds=n_seeds,
        reg_param=reg_param,
        verbose=verbose,
        **kwargs,
    )
    dataset = GaussianDataset(d=d, rho=1.0, eta=noise)
    eqs = GaussianLinearRidgeEquations(rho=1.0, eta=noise, reg_param=reg_param)
    solver = SaddlePointSolver(equations=eqs, order_params=["m", "q"], damping=0.5)

    sim = ReplicaSimulation(config)
    result = sim.run(dataset, LinearRegression, RidgeLoss(reg_param=reg_param), theory_solver=solver)

    if plot:
        from statphys.vis import plot_from_replica_results

        plot_from_replica_results(result, plot_type="all", show=show)
    return result


def quick_experiment(
    preset: str = "random_mlp",
    mode: str = "sample_complexity",
    alphas: list[float] | np.ndarray | None = None,
    t_max: float = 20.0,
    n_seeds: int = 3,
    plot: bool = True,
    show: bool = False,
    verbose: bool = False,
    preset_kwargs: dict[str, Any] | None = None,
    **run_kwargs: Any,
):
    """
    Run a general (theory-free) teacher-student preset experiment.

    Args:
        preset: Name from statphys.experiment.PRESETS
            ("random_mlp", "sparse_teacher", "spiked_teacher",
             "mismatched_width", "low_rank_attention").
        mode: "sample_complexity" (sweep alpha) or "online" (SGD dynamics).
        alphas: Sample ratios for sample_complexity mode.
        t_max: Time horizon for online mode.
        n_seeds: Number of seeds.
        plot: Draw the result figure.
        show: Call plt.show().
        verbose: Print progress.
        preset_kwargs: Options forwarded to the preset factory (e.g. d=..).
        **run_kwargs: Options forwarded to the run method (lr, batch_size, ...).

    Returns:
        ExperimentResult.

    """
    from statphys.experiment import get_preset

    exp = get_preset(preset, **(preset_kwargs or {}))

    if mode == "sample_complexity":
        if alphas is None:
            alphas = [0.5, 1.0, 2.0, 4.0, 8.0]
        result = exp.run_sample_complexity(
            alphas=alphas, n_seeds=n_seeds, verbose=verbose, **run_kwargs
        )
    elif mode == "online":
        result = exp.run_online(t_max=t_max, n_seeds=n_seeds, verbose=verbose, **run_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'sample_complexity' or 'online')")

    if plot:
        result.plot(show=show)
    return result
