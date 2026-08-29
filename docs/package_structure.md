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
│   ├── zoo.py        # Architecture zoo (linear → tiny GPT)
│   ├── observables.py # Function-space order parameters (m_hat, q_ab, chi, Binder, ...)
│   ├── phase.py      # Numerical phase diagrams (run_phase_diagram)
│   ├── mixture.py    # Gaussian-mixture classification helpers
│   ├── online_committee.py # Exact online committee-machine SGD + plateau escape
│   └── studies.py    # Ready-made phenomenology studies (STUDIES registry, run_study)
├── frontier/         # Modern paradigms as teacher-student physics (statphys.frontier)
│   ├── common.py     # Shared training loop, overlap measure, InputSampler
│   ├── teachers.py   # Teacher taxonomy (random / structured / real-data)
│   ├── sft.py        # SFT forgetting/transfer as a two-teacher problem
│   ├── rlhf.py       # Reward-model overoptimization (Goodhart transition)
│   ├── weak_to_strong.py # Weak-to-strong generalization (PGR)
│   ├── collapse.py   # Model collapse under recursive synthetic data
│   ├── icl.py        # Emergence of in-context learning
│   ├── taxonomy.py   # Teacher taxonomy x paradigm cross experiment
│   └── studies.py    # FRONTIER_STUDIES registry (merged into statphys study)
├── vis/              # Visualization
│   ├── comparison.py # ComparisonPlotter
│   ├── phase_diagram.py # PhaseDiagramPlotter (+ compute_phase_grid)
│   ├── order_params.py # OrderParamPlotter
│   ├── dynamics.py   # DynamicsPlotter (flow fields, phase portraits)
│   ├── overlap_matrix.py # OverlapMatrixPlotter (M/Q/R heatmaps)
│   ├── sweep.py      # SweepPlotter (sweeps, diagnostics)
│   ├── animation.py  # GIF/MP4 animations
│   ├── dashboard.py  # plot_order_parameter_dashboard (4-panel physics dashboard)
│   ├── plotter.py    # Shared plotting base/style helpers
│   └── default_plots.py # Publication-quality default plots
├── quick.py          # One-liner API (quick_online / quick_replica / quick_experiment / ...)
├── ui.py             # Guided-workflow implementation (catalog/new/validate/run/inspect/
│                     # report/resume/compare/doctor) backing the `statphys` CLI
├── core/             # Immutable evidence/registry contracts shared across programs
│   ├── contracts.py, protocol_spec.py, scientific_spec.py, execution_spec.py
│   ├── evidence.py, estimates.py, provenance.py, registry.py, rng.py
├── continuation/     # Phase-continuation research program (`phase-continuation` CLI)
│   ├── core/         # Immutable schemas, registry, artifacts, metrics
│   ├── domains/      # Numerical domains: transformer, diffusion, reinforcement,
│   │                 # multiagent, cross_domain (each with exact/learned/naturalistic tiers)
│   ├── analysis/     # aggregate, coverage, discovery, evidence, finite_size, taxonomy
│   ├── orchestration/ # runner, slurm submission, paper macro generation
│   └── cli.py        # coverage/taxonomy/run-local/aggregate/... subcommands
├── phase_tensor/     # Portable finite-width tensor study (`phase-tensor` CLI)
│   ├── data.py, model.py, observables.py, optimizers.py, runner.py, report.py, paper.py
├── predictive/       # Predictive phase-continuation pipeline
│   ├── pipeline.py, schema.py, simulators.py, style.py, cli.py
├── atlas/            # Audited scientific atlas (attention-ladder sweeps, GPU evidence)
│   ├── models/, data/, observables/, analysis/ # attention_ladder, spectra, discovery, scaling
│   ├── training.py, bridge_training.py, runner.py, cluster.py, sweep.py, aggregate.py
│   └── plotting.py, artifacts.py, schema.py, cli.py
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

`frontier/`, `continuation/`, `phase_tensor/`, `predictive/`, and `atlas/` are documented in
depth in [frontier.md](frontier.md) and [phase_continuation.md](phase_continuation.md); `core/`
holds the shared, program-agnostic evidence/registry contracts those subpackages build on.

## Supporting directories

| Directory | Contents |
|---|---|
| `examples/` | Runnable scripts and notebooks (galleries, replica/online demos, general experiments) |
| `scripts/` | CLI tools: `run_verification.py`, `verify_architectures.py`, `generate_readme_assets.py`, `generate_paper_figures.py`, `run_phase_study.py`, `run_gpu_reference.sh`, `check_docs.py`, and `phase_tensor/` (Slurm/Spark-portable phase-tensor pipeline scripts) — see [scripts/README.md](../scripts/README.md) |
| `tests/` | Pytest suite covering all modules, including `tests/atlas/`, `tests/continuation/`, `tests/predictive/` — see [tests/README.md](../tests/README.md) |
| `docs/` | This documentation (see [README.md](README.md)) |
| `assets/` | Logo, diagrams, and animation GIFs used by the README |
| `evidence/` | Portable reference evidence artifacts (reports, aggregates, summaries) |
| `paper/` | LaTeX manuscript, generated figures/macros, and the reproducible PDF |

## Design conventions

- **Datasets** own the teacher; **models** are students. The two only meet in `simulation/` or `experiment/`.
- **Theory scenarios** are stateless equation objects consumed by solvers (`SaddlePointSolver`, `ODESolver`); they never touch data.
- Exact results and heuristics are kept separate: heuristic gradient-flow replica scenarios inherit from `gradient_flow.GradientFlowEquations` and are labelled as such in [THEORY.md](THEORY.md).
- Numerical constants (epsilons, integration bounds, default solver settings) live only in `utils/constants.py`.
- Nothing under `src/` hardcodes cluster- or machine-specific paths.
