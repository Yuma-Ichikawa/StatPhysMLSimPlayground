# StatPhys-ML Documentation

Documentation is organized by role: the top-level [README](../README.md) covers installation and a quick tour; everything detailed lives here.

## User guides

| Document | Contents |
|---|---|
| [getting_started.md](getting_started.md) | Installation, one-liner API, full replica/online workflows, running tests |
| [components.md](components.md) | Full catalog: 22 datasets, 19 models, 16 losses, 12 theory scenarios, utilities |
| [experiments.md](experiments.md) | General (theory-free) teacher-student experiments: `Teacher`, input distributions, metrics, presets, architecture zoo (linear → tiny GPT) |
| [visualization.md](visualization.md) | Plotters (comparison, phase diagrams, dynamics, overlap matrices, sweeps) and GIF/MP4 animations |
| [slurm.md](slurm.md) | Running experiments on Slurm clusters: single jobs, job arrays, the architecture verification CLI |
| [concepts.md](concepts.md) | Order parameters, generalization error formulas, thermodynamic limits, loss-scaling conventions, automatic order-parameter extraction |
| [order_parameters.md](order_parameters.md) | Full mathematical reference for every order parameter and generalization-error formula in `statphys.experiment`, including multi-index subspace overlap, the Gaussian-mixture Bayes error, lazy/rich weight movement, and LoRA adapter recovery |

## Theory references

| Document | Contents |
|---|---|
| [THEORY.md](THEORY.md) | Feature ↔ literature map: which module implements which result, exact vs heuristic status, pointers to recent work |
| [replica_note.md](replica_note.md) | Replica method derivations (saddle-point equations, RS ansatz) |
| [online_sgd_learning_note.md](online_sgd_learning_note.md) | Online SGD ODE derivations and concentration arguments |

## Developer references

| Document | Contents |
|---|---|
| [package_structure.md](package_structure.md) | Source-tree layout and module responsibilities |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution workflow, style, and review process |
| [../CHANGELOG.md](../CHANGELOG.md) | Release history |
