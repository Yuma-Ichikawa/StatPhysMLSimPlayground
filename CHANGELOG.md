# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `statphys.experiment`: general (theory-free) teacher-student experiments
  for arbitrary PyTorch models — `Teacher` wrapper with structured weight
  initializations (random / sparse / low-rank / orthogonal / power-law /
  binary / spiked), `TeacherStudentDataset` with configurable input
  distributions (gaussian / correlated / rademacher / sphere / custom),
  model-agnostic metrics (test error, weight overlap, linear CKA),
  `TeacherStudentExperiment` with sample-complexity sweeps and online SGD
  dynamics, and ready-made presets (`random_mlp`, `sparse_teacher`,
  `spiked_teacher`, `mismatched_width`, `low_rank_attention`)
- One-liner API: `statphys.quick_online()`, `statphys.quick_replica()`,
  `statphys.quick_experiment()`
- New visualization modules: `DynamicsPlotter` (ODE flow fields, nullclines,
  phase portraits), `OverlapMatrixPlotter` (M/Q/R heatmaps, specialization
  snapshots), `SweepPlotter` (parameter sweeps, theory-vs-experiment scatter,
  convergence diagnostics), `compute_phase_grid`, and GIF/MP4 animations
  (`animate_learning_curve`, `animate_phase_plane`, `animate_overlap_matrix`)
- `TheoryResult` now supports dict-like access (`"m" in result`, `result["m"]`)
- Online simulation: theory ODE now starts from the experiment's measured
  initial condition; regression tests for the audit fixes (`tests/test_fixes.py`,
  `tests/test_experiment.py`)

### Changed
- Exact Saad-Solla closed forms (I3/I4) for the online committee-machine ODE
- Online perceptron/hinge ODEs re-derived (quadrature-based where needed)
- `for_online()` loss scaling now uses per-loss `online_scale` (0.5 only for
  squared-error losses) and respects the `d` argument for regularization
- `I3`/`I4` in `utils.special_functions` re-implemented via multi-dimensional
  Gauss-Hermite quadrature (exact, supports erf/relu)
- Unified `TheoryType` enum (single definition in `statphys.theory.base`)
- LASSO replica scenario now uses its CGMT fixed-point for the effective noise

### Fixed
- Sign error in the online MSE steady-state formula `q*`
- Conditional mean in `logistic_gaussian_integral` (`m/rho` instead of `m/sqrt(rho)`)
- Online simulation trajectory/evaluation-grid misalignment
- Dataset `.to(device)` now moves all tensor attributes
- Device handling in `RandomFeaturesModel`; batch-dimension preservation in
  MLP/committee forward passes; parameter sweep of `alpha_range` taking effect

### Documentation
- Heuristic replica scenarios (logistic / probit / hinge / committee) now
  clearly documented as gradient-flow relaxations, with configurable damping

## [0.1.0] - 2025-02-01

### Added

#### Core Features
- **Dataset Module**: Data generation for Teacher-Student models
  - `GaussianDataset`: Gaussian input distribution
  - `GaussianClassificationDataset`: Classification with Gaussian inputs
  - `SparseDataset`: Sparse input patterns
  - `StructuredDataset`: Correlated input structures

- **Model Module**: Learning model implementations
  - `LinearRegression`: Linear regression model
  - `LinearClassifier`: Linear classification model
  - `CommitteeMachine`: Hard committee machine
  - `SoftCommitteeMachine`: Soft committee machine with smooth activations
  - `TwoLayerNetwork`: Two-layer neural network
  - `SingleLayerTransformer`: Single-layer attention mechanism

- **Loss Module**: Loss functions with regularization
  - `MSELoss`: Mean squared error
  - `RidgeLoss`: MSE with L2 regularization
  - `LassoLoss`: MSE with L1 regularization
  - `HingeLoss`: Hinge loss for classification
  - `LogisticLoss`: Logistic loss for classification

- **Theory Module**: Theoretical calculations
  - `SaddlePointSolver`: Replica method saddle-point equation solver
  - `RidgeRegressionEquations`: Pre-defined equations for ridge regression
  - `ODESolver`: Online learning ODE solver
  - `OnlineSGDEquations`: ODE equations for SGD dynamics

- **Simulation Module**: Experiment framework
  - `SimulationConfig`: Configuration management
  - `ReplicaSimulation`: Batch learning simulation
  - `OnlineSimulation`: Online learning simulation
  - `SimulationRunner`: Unified simulation interface

- **Visualization Module**: Plotting utilities
  - `ComparisonPlotter`: Theory vs experiment comparison
  - `OrderParamPlotter`: Order parameter visualization
  - `PhaseDiagramPlotter`: Phase diagram generation

#### Infrastructure
- Modern Python packaging with `pyproject.toml`
- Type hints throughout the codebase (`py.typed`)
- Comprehensive test suite with pytest
- Example notebooks and scripts

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7

[Unreleased]: https://github.com/yuma-ichikawa/statphys-ml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yuma-ichikawa/statphys-ml/releases/tag/v0.1.0
