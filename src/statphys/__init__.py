"""
StatPhys-ML: Statistical Mechanics Simulation Package for Machine Learning Theory

This package provides tools for:
- Teacher-Student model analysis
- Replica method calculations
- Online learning dynamics
- DMFT (coming soon)

Example:
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

"""

__version__ = "0.1.0"
__author__ = "Yuma Ichikawa"

# Dataset imports
from statphys.dataset import (
    GaussianClassificationDataset,
    GaussianDataset,
    SparseDataset,
    get_dataset,
)

# Loss imports
from statphys.loss import (
    HingeLoss,
    LassoLoss,
    LogisticLoss,
    MSELoss,
    RidgeLoss,
    get_loss,
)

# Model imports
from statphys.model import (
    CommitteeMachine,
    LinearClassifier,
    LinearRegression,
    RidgeRegression,
    TwoLayerNetwork,
    get_model,
)

# Simulation imports
from statphys.simulation import (
    OnlineSimulation,
    ReplicaSimulation,
    SimulationConfig,
    SimulationRunner,
)

# Theory imports
from statphys.theory import (
    ODESolver,
    SaddlePointSolver,
    TheoryResult,
)
from statphys.utils.io import ResultsManager, load_results, save_results

# Core imports for convenient access
from statphys.utils.seed import fix_seed, get_device

# Visualization imports
from statphys.vis import (  # Default plotting functions
    ComparisonPlotter,
    OrderParamPlotter,
    PhaseDiagramPlotter,
    Plotter,
    apply_paper_style,
    plot_from_online_results,
    plot_from_replica_results,
    plot_generalization_error_alpha,
    plot_generalization_error_time,
    plot_order_params_alpha,
    plot_order_params_time,
)

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
    "vis",
    "utils",
]


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
    print("  - simulation: Numerical experiments")
    print("  - vis: Visualization tools")
