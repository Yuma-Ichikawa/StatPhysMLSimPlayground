# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation

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
