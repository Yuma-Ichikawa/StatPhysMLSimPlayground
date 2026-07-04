# Glossary — Statistical Physics ↔ Machine Learning

A dictionary between statistical-physics-of-learning jargon and the
machine-learning vocabulary most readers already know. If a term in
this codebase, [concepts.md](concepts.md), or
[order_parameters.md](order_parameters.md) is unfamiliar, look it up
here first. Aimed at ML practitioners with no statistical-mechanics
background who want to understand *why* this repository is built the
way it is.

| Physics term | ML-friendly meaning | Where it shows up here |
|---|---|---|
| **Teacher / student** | The (unknown) ground-truth function that generated the labels, and the model being trained to approximate it | `Teacher`, `TeacherStudentExperiment` |
| **Quenched disorder** | The random, fixed-once realization of the problem: the teacher's weights and the training data. "Averaging over disorder" = averaging over random seeds | `n_seeds` / `n_replicas`, `base_seed` |
| **Order parameter** | A summary statistic that captures the macroscopic state of learning, analogous to magnetization in a magnet — e.g. "how aligned is the student with the teacher?" | $\hat m$, $q_{ab}$, subspace overlap, ... ([order_parameters.md](order_parameters.md)) |
| **Magnetization $m$ / $\hat m$** | Normalized correlation between the student's and teacher's predictions — a generalization/recovery score in $[-1, 1]$ | `function_order_params`, "teacher overlap" |
| **Self-overlap $q$** | How large/confident the student's own outputs are (its "norm") | `function_order_params` |
| **Replica** | One independently trained copy of the student (same setup, different random seed/init) — used to measure *how many different solutions* fit the data equally well | `n_replicas`, `replica_overlaps` |
| **Replica overlap $q_{ab}$** | Agreement between two independently trained models — high means "there's basically one good solution" (like model ensembling agreeing strongly), low means many different, disagreeing solutions exist | `replica_overlaps`, `q_ab_mean` |
| **Susceptibility $\chi$** | How sensitive an order parameter is to noise/perturbation; spikes sharply exactly at a phase transition, like how a magnet's susceptibility to an external field diverges at the Curie temperature | `susceptibility` |
| **Binder cumulant $U_4$** | A shape-of-distribution diagnostic (kurtosis-like) that pinpoints a critical point independent of system size — practically, curves for different $d$ crossing at one point locates the transition precisely | `binder_cumulant`, finite-size scaling studies |
| **Sample ratio $\alpha = n/d$** | The "amount of data" in dimensionless units — the single control parameter replacing "number of training examples" once you fix a model size | Every sweep in this package |
| **Thermodynamic limit** | Taking $n, d \to \infty$ at fixed $\alpha$ — the regime where sharp, deterministic phase transitions emerge from noisy finite-size behavior, analogous to the law of large numbers making empirical averages exact | Underlies all replica/online theory |
| **Phase transition** | A parameter value ($\alpha_c$, a critical noise level, ...) where behavior changes qualitatively and abruptly — e.g. "with too little data the model *cannot* recover the signal at all, no matter how long you train," then suddenly it can | `run_phase_diagram`, `chi_m` peaks, `study_fss` |
| **Generalization error $\epsilon_g$** | Ordinary test error / expected loss on fresh data — same meaning as in standard ML, but derived here as an exact function of order parameters rather than only measured empirically | `test_error`, `generalization_error_decomposition` |
| **Specialization** | A hidden unit "picking" one specific direction/feature to represent, instead of a generic mixture — the numerical signature of feature learning in a committee machine | `specialization_index`, `study_committee` |
| **Condensation / condensed phase** | All replicas converge to (nearly) the same function — analogous to different training runs of a well-posed convex problem always finding the same optimum | High `q_ab_mean` |
| **Glassy phase / RSB (replica symmetry breaking)** | Many distinct, disagreeing solutions of similar quality exist (a rugged, multi-modal loss landscape) — the numerical fingerprint is low, spread-out `q_ab` across replicas | `replica_overlaps` (low/variable `q_ab`) |
| **Lazy training / NTK regime** | Training barely moves the weights from initialization; the network effectively behaves like its randomly-initialized kernel (no new features learned) | `weight_movement`, `init_scale`, `study_lazy_rich` |
| **Rich / feature-learning regime** | The opposite of lazy: weights move substantially and the network discovers task-relevant features/directions that weren't present at initialization | Small `init_scale`, high `weight_movement` |
| **Multi-index model** | A regression/classification target that depends on the input only through $K$ linear projections (the "relevant directions"); learning = finding those $K$ directions | `multi_index_model`, `subspace_overlap` |
| **Planted signal / spike** | A single directional structure hidden inside otherwise random weights, which the student must detect and recover | `spiked_teacher`, `sparse_teacher` |
| **Detectability transition (BBP)** | Below a critical signal strength, no algorithm can find the planted signal better than chance, even with infinite data — above it, recovery is possible | `spiked_teacher` |
| **Bayes error / Bayes-optimal** | The best possible generalization error achievable by *any* classifier, i.e. the error of the true data-generating decision rule | `bayes_error`, `GaussianMixtureDataset` |
| **Double descent** | Test error can *increase* then *decrease again* as model capacity grows past the interpolation threshold — counter to classical U-shaped bias-variance intuition | `study_double_descent` |
| **Grokking** | A model perfectly memorizes the training set long before it generalizes; test error suddenly collapses much later in training | `run_training_dynamics`, `study_grokking` |
| **Scaling law** | Generalization error decays as a power law in data/model/compute, $\epsilon_g \sim \alpha^{-b}$ | `study_scaling` |
| **DMFT (dynamical mean-field theory)** | The technique used to derive *exact* training-dynamics equations (not just equilibrium/converged behavior) in the high-dimensional limit — the theoretical counterpart to what this package measures numerically via `run_training_dynamics` | [THEORY.md](THEORY.md) |
| **Universality** | Many microscopically different input distributions give the *same* macroscopic learning curve, as long as low-order statistics (mean/covariance) match — the high-dimensional analogue of the central limit theorem | `study_universality` |

## Suggested reading order for newcomers

1. [getting_started.md](getting_started.md) — install and run your first
   experiment.
2. This glossary, alongside [concepts.md](concepts.md) — the vocabulary
   and the classical ($m, q, \rho$) order parameters.
3. [experiments.md](experiments.md) — the theory-free experiment API
   that works for *any* architecture.
4. [order_parameters.md](order_parameters.md) — full derivations once
   you want to understand *why* a formula holds, not just what it
   measures.
5. [THEORY.md](THEORY.md) — pointers into the literature for every
   result implemented here.
