# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Realistic settings, round 2**: `multi_index_model` preset (K-direction
  multi-index teacher; supports `k_student != k_teacher`),
  `mixture_classification` preset + `GaussianMixtureDataset` (generative
  Gaussian-mixture classification with an exact, numerically verified
  Bayes error `Phi(-mu * cos_angle)`), and `lora_finetune` preset
  (frozen "pretrained" backbone + trainable low-rank adapter, LoRA-style
  fine-tuning)
- New order parameters in `statphys.experiment.observables`:
  `subspace_overlap` (principal-angle overlap between K-dim relevant
  subspaces, permutation/basis-invariant, works with mismatched width),
  `vector_overlap` (cosine similarity, used for cluster-axis and LoRA
  adapter recovery), and `generalization_error_decomposition` (checks
  the exact identity `eps_g = 1/2(rho_f + q_f - 2 m_f)` numerically)
- `TeacherStudentExperiment(..., dataset=...)`: plug in a fully custom
  *generative* dataset (label determines input, not the other way
  around) while keeping every existing protocol and order parameter
- `run_order_parameters(..., init_scale=...)` now always records
  `weight_movement` (`||theta_final - theta_init|| / ||theta_init||`),
  the parametrization-agnostic diagnostic for the lazy (NTK/kernel) vs.
  rich (feature-learning) regime (Chizat & Bach 2019)
- Four new studies: `multi_index` (subspace recovery, matched/mismatched
  width), `mixture` (Gaussian-mixture classification vs. the analytic
  Bayes error), `lazy_rich` (lazy vs. rich regime via init scale),
  `lora` (LoRA adapter recovery vs. fine-tuning data and rank)
- `docs/order_parameters.md`: full mathematical reference (with
  derivations and literature pointers) for every order parameter and
  generalization-error formula in `statphys.experiment`
- `statphys` console command (CLI): `list`, `order-params`,
  `phase-diagram`, and `study` subcommands — full physics experiments
  without writing Python
- `TeacherStudentExperiment.run_training_dynamics`: epoch-resolved
  protocol (train/test error, m_hat, q_ab at log-spaced epochs) with an
  `init_scale` knob for grokking-style delayed generalization
- `ExperimentResult.save()` / `ExperimentResult.load()` (JSON round trip)
- Studies moved into the library (`statphys.experiment.studies`,
  exported as `STUDIES` / `run_study`); `scripts/run_phase_study.py` is
  now a thin wrapper
- Four new studies: `grokking` (delayed generalization in epoch time),
  `universality` (Gaussian universality of learning curves and its
  breakdown), `double_descent` (model-wise double descent vs student
  width), `scaling` (eps_g ~ alpha^-b exponents across architectures)
- Hidden-manifold input distribution (`input_dist="hidden_manifold"`,
  Goldt et al. 2020) for realistic low-dimensional data structure
- Presets `hidden_manifold` (MLP on manifold inputs) and `tiny_gpt`
  (minimal causal transformer teacher-student pair)
- One-liner APIs `statphys.quick_order_parameters()` and
  `statphys.quick_phase_diagram()` with automatic plotting
- Shared 4-panel dashboard `statphys.vis.plot_order_parameter_dashboard`
  (order parameters, generalization error, susceptibility, Binder)
- New studies in `scripts/run_phase_study.py`: `manifold` (data-structure
  dependence of the transition) and `gpt` (LLM-style pair)
- Physics-style order parameters for theory-free experiments
  (`statphys.experiment.observables`): function-space magnetization
  `m_hat`, replica-replica overlap `q_ab`, susceptibility `chi_m`,
  Binder cumulant, participation ratio, and a permutation-resolved
  hidden-unit `specialization_index`
- `TeacherStudentExperiment.run_order_parameters`: replica-resolved
  alpha sweeps (shared or independent training data) recording all
  order parameters plus cross-replica aggregates
- Numerical phase diagrams (`statphys.experiment.phase`):
  `run_phase_diagram` sweeps (control parameter x alpha) grids with
  heatmap/contour plotting via `PhaseDiagramResult`
- `Teacher.clean()` (noiseless labels) and
  `TeacherStudentDataset.sample_inputs()` (probe sets)
- `scripts/run_phase_study.py`: committee specialization, sparse-recovery
  finite-size scaling, 2D recovery diagram, and attention-teacher studies
- Tests for all new observables and protocols (`tests/test_observables.py`)

### Documentation
- README slimmed down to installation, quick start, and animated highlights;
  detailed material moved into `docs/` with a per-topic layout
  (`getting_started`, `components`, `experiments`, `visualization`,
  `slurm`, `concepts`, `package_structure`) indexed by `docs/README.md`
- Animated GIFs (learning curve vs theory, phase-plane dynamics,
  committee-machine specialization) embedded in the README, regenerable
  via `scripts/generate_readme_assets.py`

### Added
- Architecture zoo (`statphys.experiment.zoo`): matched teacher-student
  pairs for linear / MLP / deep MLP / 1D-CNN / LSTM / single-head attention /
  tiny-GPT (causal transformer) architectures, plus
  `architecture_experiment()` and the `scripts/verify_architectures.py`
  CLI (local or Slurm-array execution)
- Slurm utilities (`statphys.utils.slurm`): `SlurmConfig`, `render_sbatch`,
  `SlurmLauncher` (submit/state/wait) and `submit_array`, with no
  hardcoded cluster paths
- Centralized numerical constants (`statphys.utils.constants`): epsilons,
  correlation clips, Gaussian integration bounds, solver defaults
- `docs/THEORY.md`: feature ↔ literature map (Engel & Van den Broeck,
  Zdeborová & Krzakala 2016, Saad & Solla 1995, Cui et al. 2025
  dot-product-attention and attention-indexed models, ...)

### Changed
- Heuristic replica scenarios (logistic/probit/hinge) now share a
  `GradientFlowEquations` base class (joint-field integrals + damped
  relaxation in one place); committee replica damping is configurable
- Inline Gaussian CDF/PDF/tail helpers and `arccos` classification-error
  formulas replaced by the canonical `statphys.utils.special_functions`
  implementations across theory scenarios, `model/linear.py`, and
  `utils/order_params.py`
- Online hinge scenario: quadrature size exposed as `n_quad`
- Removed duplicated `tests/run_verification.py` (kept `scripts/` copy)

### Added (previous batch)
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

### Changed (previous batch)
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
