# Package Structure

Modules follow a strict separation of concerns: data generation, models, losses, theory, simulation orchestration, theory-free experiments, visualization, and shared utilities.

```
src/statphys/
├── dataset/          # Data generation
│   ├── base.py       # BaseDataset abstract class
│   ├── gaussian.py   # Gaussian, Classification, MultiOutput
│   ├── sparse.py     # Sparse, BernoulliGaussian
│   ├── structured.py # Structured, Correlated, Spiked
│   ├── glm.py        # Logistic, Probit, GaussianMixture teachers
│   ├── icl.py        # ICL linear/nonlinear regression tasks
│   ├── sequence.py   # Markov, Copy, Potts, TiedAttention, MixedSequence
│   ├── attention.py  # AttentionIndexedModel (AIM)
│   ├── fairness.py   # TeacherMixtureFairness
│   └── noisy.py      # NoisyGMMSelfDistillation
├── model/            # Learning models
│   ├── base.py       # BaseModel abstract class
│   ├── linear.py     # LinearRegression, Classifier, Ridge
│   ├── committee.py  # CommitteeMachine, SoftCommittee
│   ├── mlp.py        # TwoLayerNetwork, DeepNetwork
│   ├── random_features.py # RandomFeatures, KernelRidge, DeepLinear
│   ├── softmax.py    # SoftmaxRegression
│   ├── transformer.py # Attention, Transformer
│   └── sequence.py   # LSA, SSM, RNN, Hopfield
├── loss/             # Loss functions
│   ├── base.py       # BaseLoss abstract class (replica/online scaling)
│   ├── regression.py # MSE, Ridge, LASSO, ElasticNet, Huber
│   └── classification.py # Hinge, Logistic, Probit, Softmax, etc.
├── theory/           # Theoretical calculations
│   ├── replica/      # Replica method
│   │   ├── solver.py # SaddlePointSolver (damping, continuation)
│   │   └── scenario/ # Saddle-point equations by scenario
│   │       ├── base.py            # ReplicaEquations base class
│   │       ├── gradient_flow.py   # Shared heuristic gradient-flow base
│   │       └── gaussian_*.py      # Ridge/LASSO/logistic/hinge/probit/committee
│   ├── online/       # Online learning
│   │   ├── solver.py # ODESolver, AdaptiveODESolver
│   │   └── scenario/ # ODE equations by scenario
│   │       ├── base.py            # OnlineEquations base class
│   │       └── gaussian_*.py      # MSE/ridge/perceptron/logistic/hinge/committee
│   └── dmft/         # DMFT (coming soon)
├── simulation/       # Numerical experiments (with theory comparison)
│   ├── base.py       # BaseSimulation
│   ├── config.py     # SimulationConfig
│   ├── replica_sim.py # ReplicaSimulation
│   ├── online_sim.py  # OnlineSimulation
│   └── runner.py     # SimulationRunner
├── experiment/       # General teacher-student experiments (theory-free)
│   ├── teacher.py    # Teacher wrapper + weight-init strategies
│   ├── dataset.py    # TeacherStudentDataset (input distributions)
│   ├── metrics.py    # test error, weight overlap, CKA
│   ├── protocol.py   # TeacherStudentExperiment, ExperimentResult
│   ├── presets.py    # Ready-made setups (random_mlp, sparse_teacher, ...)
│   └── zoo.py        # Architecture zoo (linear → tiny GPT)
├── vis/              # Visualization
│   ├── comparison.py # ComparisonPlotter
│   ├── phase_diagram.py # PhaseDiagramPlotter (+ compute_phase_grid)
│   ├── order_params.py # OrderParamPlotter
│   ├── dynamics.py   # DynamicsPlotter (flow fields, phase portraits)
│   ├── overlap_matrix.py # OverlapMatrixPlotter (M/Q/R heatmaps)
│   ├── sweep.py      # SweepPlotter (sweeps, diagnostics)
│   ├── animation.py  # GIF/MP4 animations
│   └── default_plots.py # Publication-quality default plots
├── quick.py          # One-liner API (quick_online / quick_replica / quick_experiment)
└── utils/            # Utilities
    ├── special_functions.py # Gaussian functions, erf, I2/I3/I4, error formulas
    ├── integration.py # Gaussian integrals (Hermite/quad/MC)
    ├── order_params.py # Automatic order-parameter calculation
    ├── constants.py   # Centralized numerical constants
    ├── slurm.py       # Slurm job generation and submission
    ├── math.py        # Basic math utilities
    ├── seed.py        # Random seed management
    └── io.py          # Results I/O
```

## Supporting directories

| Directory | Contents |
|---|---|
| `examples/` | Runnable scripts and notebooks (galleries, replica/online demos, general experiments) |
| `scripts/` | CLI tools: `run_verification.py` (theory vs experiment), `verify_architectures.py` (zoo end-to-end, local or Slurm), `generate_readme_assets.py` (README figures/GIFs) |
| `tests/` | Pytest suite covering all modules |
| `docs/` | This documentation (see [README.md](README.md)) |
| `assets/` | Logo, diagrams, and animation GIFs used by the README |

## Design conventions

- **Datasets** own the teacher; **models** are students. The two only meet in `simulation/` or `experiment/`.
- **Theory scenarios** are stateless equation objects consumed by solvers (`SaddlePointSolver`, `ODESolver`); they never touch data.
- Exact results and heuristics are kept separate: heuristic gradient-flow replica scenarios inherit from `gradient_flow.GradientFlowEquations` and are labelled as such in [THEORY.md](THEORY.md).
- Numerical constants (epsilons, integration bounds, default solver settings) live only in `utils/constants.py`.
- Nothing under `src/` hardcodes cluster- or machine-specific paths.
