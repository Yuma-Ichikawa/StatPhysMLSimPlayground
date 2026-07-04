"""
One-liner high-level API for common workflows.

These helpers wire together dataset / model / loss / theory / simulation /
visualization so that standard experiments are a single call:

    >>> import statphys
    >>> result = statphys.quick_online(d=400, lr=0.5, t_max=10)   # runs + plots
    >>> result = statphys.quick_replica(d=200, reg_param=0.1)     # runs + plots
    >>> res = statphys.quick_experiment("random_mlp", alphas=[1, 2, 4, 8])
    >>> res = statphys.quick_order_parameters("tiny_gpt")         # physics dashboard
    >>> res = statphys.quick_phase_diagram("sparse_teacher", "sparsity",
    ...                                    [0.5, 0.8, 0.95])

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
    result = sim.run(
        dataset, LinearRegression, RidgeLoss(reg_param=reg_param), theory_solver=solver
    )

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
    result = sim.run(
        dataset, LinearRegression, RidgeLoss(reg_param=reg_param), theory_solver=solver
    )

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


def quick_order_parameters(
    preset: str = "random_mlp",
    alphas: list[float] | np.ndarray | None = None,
    n_replicas: int = 4,
    plot: bool = True,
    show: bool = False,
    verbose: bool = False,
    preset_kwargs: dict[str, Any] | None = None,
    **run_kwargs: Any,
):
    """
    Physics order parameters (m_hat, q_ab, chi, Binder) for any preset.

    One line gives the full statistical-physics dashboard — teacher
    recovery, replica overlap, susceptibility, Binder cumulant, and the
    generalization error — for arbitrary architectures:

        >>> statphys.quick_order_parameters("tiny_gpt", alphas=[1, 2, 4, 8])

    Args:
        preset: Name from statphys.experiment.PRESETS
            ("random_mlp", "sparse_teacher", "spiked_teacher",
             "mismatched_width", "low_rank_attention", "hidden_manifold",
             "tiny_gpt").
        alphas: Sample ratios (default: log-spaced 0.25 ... 16).
        n_replicas: Independently trained students per alpha.
        plot: Draw the 4-panel dashboard.
        show: Call plt.show().
        verbose: Print progress.
        preset_kwargs: Options forwarded to the preset factory (e.g. d=..).
        **run_kwargs: Options forwarded to run_order_parameters
            (lr, max_epochs, weight_decay, l1_penalty, ...).

    Returns:
        ExperimentResult with per-replica and cross-replica records.

    """
    from statphys.experiment import get_preset

    exp = get_preset(preset, **(preset_kwargs or {}))
    if alphas is None:
        alphas = np.round(np.logspace(np.log10(0.25), np.log10(16), 7), 3).tolist()
    result = exp.run_order_parameters(
        alphas=alphas, n_replicas=n_replicas, verbose=verbose, **run_kwargs
    )

    if plot:
        from statphys.vis import plot_order_parameter_dashboard

        extra = ("specialization",) if "specialization" in result.records else ()
        plot_order_parameter_dashboard(result, title=preset, extra_metrics=extra, show=show)
    return result


def quick_phase_diagram(
    preset: str,
    param_name: str,
    param_values: list[float],
    alphas: list[float] | np.ndarray | None = None,
    metric: str = "m_hat",
    n_replicas: int = 3,
    plot: bool = True,
    show: bool = False,
    verbose: bool = False,
    preset_kwargs: dict[str, Any] | None = None,
    **run_kwargs: Any,
):
    """
    2D numerical phase diagram (preset parameter x alpha) in one line.

    The named preset option is swept as the control parameter:

        >>> statphys.quick_phase_diagram("sparse_teacher", "sparsity",
        ...                              [0.5, 0.8, 0.9, 0.95])

    Args:
        preset: Name from statphys.experiment.PRESETS.
        param_name: Preset keyword to sweep (e.g. "sparsity", "snr",
            "latent_dim", "noise_std").
        param_values: Values of the control parameter (rows).
        alphas: Sample ratios (columns; default log-spaced 0.25 ... 8).
        metric: Grid to plot ("m_hat", "chi_m", "test_error", ...).
        n_replicas: Replicas per grid point.
        plot: Draw the heatmap with the 0.5 contour when metric is m_hat.
        show: Call plt.show().
        verbose: Print progress.
        preset_kwargs: Fixed options forwarded to the preset factory.
        **run_kwargs: Options forwarded to run_order_parameters.

    Returns:
        PhaseDiagramResult.

    """
    from statphys.experiment import get_preset, run_phase_diagram

    if alphas is None:
        alphas = np.round(np.logspace(np.log10(0.25), np.log10(8), 6), 3).tolist()
    fixed = dict(preset_kwargs or {})

    def factory(value: float):
        return get_preset(preset, **{**fixed, param_name: value})

    result = run_phase_diagram(
        factory,
        param_name=param_name,
        param_values=param_values,
        alphas=list(alphas),
        n_replicas=n_replicas,
        verbose=verbose,
        **run_kwargs,
    )

    if plot:
        contour = 0.5 if metric == "m_hat" else None
        result.plot(metric, logx=True, contour_level=contour, show=show)
    return result
