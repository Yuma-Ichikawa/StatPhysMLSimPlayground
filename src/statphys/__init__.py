"""
StatPhys-ML: Statistical Mechanics Simulation Package for Machine Learning Theory

This package provides tools for:
- Teacher-Student model analysis
- Replica method calculations
- Online learning dynamics
- General (theory-free) teacher-student experiments for arbitrary models
- DMFT (coming soon)

Quick start (one-liners):
    >>> import statphys
    >>> statphys.quick_online(d=400, lr=0.5, t_max=10)      # SGD vs ODE theory
    >>> statphys.quick_replica(d=200, reg_param=0.1)        # ERM vs replica theory
    >>> statphys.quick_experiment("random_mlp")             # theory-free preset
    >>> statphys.quick_order_parameters("tiny_gpt")         # physics dashboard
    >>> statphys.quick_phase_diagram("sparse_teacher", "sparsity", [0.5, 0.9])

Full API example:
    >>> import statphys
    >>> from statphys.dataset import GaussianDataset
    >>> from statphys.model import LinearRegression
    >>> from statphys.loss import RidgeLoss
    >>> from statphys.simulation import ReplicaSimulation, SimulationConfig
    >>>
    >>> # Create dataset
    >>> dataset = GaussianDataset(d=500, rho=1.0, eta=0.1)
    >>>
    >>> # Configure simulation
    >>> config = SimulationConfig.for_replica(
    ...     alpha_range=(0.1, 5.0),
    ...     n_seeds=5,
    ...     reg_param=0.01
    ... )
    >>>
    >>> # Run simulation
    >>> sim = ReplicaSimulation(config)
    >>> results = sim.run(dataset, LinearRegression, RidgeLoss(0.01))
    >>>
    >>> # Visualize
    >>> from statphys.vis import ComparisonPlotter
    >>> plotter = ComparisonPlotter()
    >>> plotter.plot_theory_vs_experiment(results)

General teacher-student experiments (works for any nn.Module):
    >>> from statphys.experiment import Teacher, TeacherStudentExperiment
    >>> teacher = Teacher(my_transformer, init="low_rank",
    ...                   init_kwargs={"rank": 4})
    >>> exp = TeacherStudentExperiment(teacher, student_factory=make_student)
    >>> exp.run_sample_complexity(alphas=[1, 2, 4, 8]).plot()

"""

__version__ = "0.1.0"
__author__ = "Yuma Ichikawa"

# Keep the convenient top-level API without importing every scientific and
# plotting dependency when a focused subpackage (for example ``predictive``)
# is launched.  Values are imported and cached on first access via PEP 562.
_LAZY_IMPORTS = {
    "GaussianClassificationDataset": ("statphys.dataset", "GaussianClassificationDataset"),
    "GaussianDataset": ("statphys.dataset", "GaussianDataset"),
    "SparseDataset": ("statphys.dataset", "SparseDataset"),
    "get_dataset": ("statphys.dataset", "get_dataset"),
    "ExperimentResult": ("statphys.experiment", "ExperimentResult"),
    "Teacher": ("statphys.experiment", "Teacher"),
    "TeacherStudentDataset": ("statphys.experiment", "TeacherStudentDataset"),
    "TeacherStudentExperiment": ("statphys.experiment", "TeacherStudentExperiment"),
    "HingeLoss": ("statphys.loss", "HingeLoss"),
    "LassoLoss": ("statphys.loss", "LassoLoss"),
    "LogisticLoss": ("statphys.loss", "LogisticLoss"),
    "MSELoss": ("statphys.loss", "MSELoss"),
    "RidgeLoss": ("statphys.loss", "RidgeLoss"),
    "get_loss": ("statphys.loss", "get_loss"),
    "CommitteeMachine": ("statphys.model", "CommitteeMachine"),
    "LinearClassifier": ("statphys.model", "LinearClassifier"),
    "LinearRegression": ("statphys.model", "LinearRegression"),
    "RidgeRegression": ("statphys.model", "RidgeRegression"),
    "TwoLayerNetwork": ("statphys.model", "TwoLayerNetwork"),
    "get_model": ("statphys.model", "get_model"),
    "OnlineSimulation": ("statphys.simulation", "OnlineSimulation"),
    "ReplicaSimulation": ("statphys.simulation", "ReplicaSimulation"),
    "SimulationConfig": ("statphys.simulation", "SimulationConfig"),
    "SimulationRunner": ("statphys.simulation", "SimulationRunner"),
    "quick_experiment": ("statphys.quick", "quick_experiment"),
    "quick_online": ("statphys.quick", "quick_online"),
    "quick_order_parameters": ("statphys.quick", "quick_order_parameters"),
    "quick_phase_diagram": ("statphys.quick", "quick_phase_diagram"),
    "quick_replica": ("statphys.quick", "quick_replica"),
    "ODESolver": ("statphys.theory", "ODESolver"),
    "SaddlePointSolver": ("statphys.theory", "SaddlePointSolver"),
    "TheoryResult": ("statphys.theory", "TheoryResult"),
    "ResultsManager": ("statphys.utils.io", "ResultsManager"),
    "load_results": ("statphys.utils.io", "load_results"),
    "save_results": ("statphys.utils.io", "save_results"),
    "OrderParameterCalculator": ("statphys.utils.order_params", "OrderParameterCalculator"),
    "auto_calc_order_params": ("statphys.utils.order_params", "auto_calc_order_params"),
    "fix_seed": ("statphys.utils.seed", "fix_seed"),
    "get_device": ("statphys.utils.seed", "get_device"),
    "ComparisonPlotter": ("statphys.vis", "ComparisonPlotter"),
    "OrderParamPlotter": ("statphys.vis", "OrderParamPlotter"),
    "PhaseDiagramPlotter": ("statphys.vis", "PhaseDiagramPlotter"),
    "Plotter": ("statphys.vis", "Plotter"),
    "apply_paper_style": ("statphys.vis", "apply_paper_style"),
    "plot_from_online_results": ("statphys.vis", "plot_from_online_results"),
    "plot_from_replica_results": ("statphys.vis", "plot_from_replica_results"),
    "plot_generalization_error_alpha": ("statphys.vis", "plot_generalization_error_alpha"),
    "plot_generalization_error_time": ("statphys.vis", "plot_generalization_error_time"),
    "plot_order_params_alpha": ("statphys.vis", "plot_order_params_alpha"),
    "plot_order_params_time": ("statphys.vis", "plot_order_params_time"),
}

_SUBMODULES = {"dataset", "model", "loss", "theory", "simulation", "experiment", "vis", "utils"}

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Utils
    "fix_seed",
    "get_device",
    "save_results",
    "load_results",
    "ResultsManager",
    "OrderParameterCalculator",
    "auto_calc_order_params",
    # Dataset
    "GaussianDataset",
    "GaussianClassificationDataset",
    "SparseDataset",
    "get_dataset",
    # Model
    "LinearRegression",
    "LinearClassifier",
    "RidgeRegression",
    "CommitteeMachine",
    "TwoLayerNetwork",
    "get_model",
    # Loss
    "MSELoss",
    "RidgeLoss",
    "LassoLoss",
    "HingeLoss",
    "LogisticLoss",
    "get_loss",
    # Simulation
    "SimulationConfig",
    "ReplicaSimulation",
    "OnlineSimulation",
    "SimulationRunner",
    # General experiments
    "Teacher",
    "TeacherStudentDataset",
    "TeacherStudentExperiment",
    "ExperimentResult",
    # One-liner API
    "quick_online",
    "quick_replica",
    "quick_experiment",
    "quick_order_parameters",
    "quick_phase_diagram",
    # Theory
    "SaddlePointSolver",
    "ODESolver",
    "TheoryResult",
    # Visualization
    "Plotter",
    "ComparisonPlotter",
    "OrderParamPlotter",
    "PhaseDiagramPlotter",
    # Default plotting functions
    "apply_paper_style",
    "plot_generalization_error_alpha",
    "plot_order_params_alpha",
    "plot_generalization_error_time",
    "plot_order_params_time",
    "plot_from_replica_results",
    "plot_from_online_results",
    # Submodules
    "dataset",
    "model",
    "loss",
    "theory",
    "simulation",
    "experiment",
    "vis",
    "utils",
]


def __getattr__(name: str):
    """Lazily resolve public conveniences while preserving the legacy API."""
    from importlib import import_module

    if name in _LAZY_IMPORTS:
        module_name, attribute = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(f"statphys.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy public names to interactive tools and API documentation."""
    return sorted(set(globals()) | set(__all__))


def get_version() -> str:
    """Return the package version."""
    return __version__


def info() -> None:
    """Print package information."""
    print(f"StatPhys-ML v{__version__}")
    print(f"Author: {__author__}")
    print("\nAvailable modules:")
    print("  - dataset: Data generation (Gaussian, Sparse, Structured)")
    print("  - model: Learning models (Linear, Committee, MLP, Transformer)")
    print("  - loss: Loss functions (MSE, Ridge, LASSO, Hinge, Logistic)")
    print("  - theory: Theoretical calculations (Replica, Online ODEs)")
    print("  - simulation: Numerical experiments (with theory comparison)")
    print("  - experiment: General teacher-student experiments (any nn.Module)")
    print("  - vis: Visualization tools (plots, phase portraits, animations)")
    print("\nQuick start: statphys.quick_online(), statphys.quick_replica(),")
    print("             statphys.quick_experiment('random_mlp')")
