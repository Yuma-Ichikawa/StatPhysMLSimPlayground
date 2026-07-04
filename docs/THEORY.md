# Theory Map: Features ↔ Literature

This document maps the package's functionality onto the statistical-mechanics-of-learning literature, so that users know (a) which classical/modern result each module implements, (b) what is exact vs heuristic, and (c) where to look for extensions.

For derivations, see `docs/replica_note.md` and `docs/online_sgd_learning_note.md`.

## 1. Foundational reviews

| Reference | Relevance to this package |
|---|---|
| Engel & Van den Broeck (2001), *Statistical Mechanics of Learning* | Core teacher-student formalism, Gardner analysis, order parameters `m`, `q`, `ρ` used throughout `theory/` |
| Zdeborová & Krzakala (2016), "Statistical physics of inference: thresholds and algorithms," *Adv. Phys.* 65 | Phase-transition phenomenology (impossible/hard/easy phases) that `experiment/` measures numerically |
| Gabrié (2020), "Mean-field inference methods for neural networks," *J. Phys. A* 53 | Overview of the replica/TAP/AMP toolbox underlying `theory/replica` |
| Cui (2025), "High-dimensional learning of narrow neural networks" (arXiv:2409.13904) | Modern unified treatment of committee/attention scenarios mirrored by our scenario registry |

## 2. Replica module (`statphys/theory/replica`)

| Scenario class | Setting | Literature | Status |
|---|---|---|---|
| `GaussianLinearRidgeEquations` | Ridge regression, Gaussian design | Krogh & Hertz (1992); Advani & Ganguli (2016) | Exact RS saddle point |
| `GaussianLinearMseEquations` | Unregularized MSE | idem | Exact RS |
| `GaussianLinearLassoEquations` | LASSO / sparse recovery | Bayati & Montanari (2011); Thrampoulidis et al. (2018) CGMT | Exact effective-noise fixed point |
| `GaussianLinearLogisticEquations` | Logistic regression | Salehi, Abbasi, Hassibi (2019); Aubin et al. (2020) | **Heuristic gradient-flow relaxation** (see docstring warning) |
| `GaussianLinearProbitEquations` | Probit teacher/loss | Opper & Kinzel (1996) | **Heuristic gradient-flow relaxation** |
| `GaussianLinearHingeEquations` | Perceptron/SVM, Gardner capacity | Gardner (1988); Dietrich, Opper, Sompolinsky (1999) | **Heuristic**; `critical_capacity()` returns the Gardner bound α_c = 2 at κ = 0 |
| `GaussianCommitteeMseEquations` | Soft committee machine | Schwarze & Hertz (1993); Aubin et al. (2018) "The committee machine" | **Heuristic symmetric-ansatz relaxation** |

Shared machinery: the heuristic scenarios derive from `GradientFlowEquations` (`theory/replica/scenario/gradient_flow.py`), which centralizes the joint teacher/student field integrals and the damped relaxation step. Numerical constants (integration bounds, clipping) live in `statphys/utils/constants.py`.

**Known exact alternatives** (roadmap): the logistic/probit/hinge scenarios admit exact RS saddle-point equations via proximal operators (Moreau envelopes), cf. Salehi et al. (2019) and Loureiro et al. (2021) "Learning curves of generic features maps for realistic datasets" — `utils.special_functions.moreau_envelope` and `proximal_operator` are already available as building blocks.

## 3. Online-learning module (`statphys/theory/online`)

| Scenario class | Setting | Literature | Status |
|---|---|---|---|
| `GaussianLinearMseEquations` | Online SGD, linear regression | Biehl & Schwarze (1995) | Exact ODEs |
| `GaussianLinearPerceptronEquations` | Online perceptron | Kinzel & Opper (1991); Biehl & Riegler (1994) | Exact closed form |
| `GaussianLinearHingeEquations` | Online SVM/hinge | Dietrich et al. (1999) | Exact up to 1D quadrature |
| `GaussianLinearLogisticEquations` | Online logistic | — | Quadrature-based |
| `GaussianCommitteeMseEquations` | Soft committee (erf) | Saad & Solla (1995), *Phys. Rev. E* 52, 4225 | Exact Saad-Solla `I3`/`I4` closed forms |

The `I2`/`I3`/`I4` Gaussian correlation integrals (`utils/special_functions.py`) follow Saad & Solla's appendix; for non-erf activations they fall back to multi-dimensional Gauss-Hermite quadrature.

## 4. General (theory-free) experiments (`statphys/experiment`)

Where no analytic theory exists, phase transitions are measured numerically. The presets and the architecture zoo are modeled on settings from the recent literature:

| Feature | Setting | Literature |
|---|---|---|
| `presets.random_mlp` | Random committee-style teacher, specialization plateaus | Saad & Solla (1995); Goldt et al. (2019) "Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup" |
| `presets.sparse_teacher` | Sparse linear recovery transition | Compressed sensing / Bayati & Montanari (2011) |
| `presets.spiked_teacher` | Rank-1 planted spike, detectability transition | BBP (Baik, Ben Arous, Péché 2005); Lesieur, Krzakala, Zdeborová (2017) |
| `presets.mismatched_width` | Overparameterized student | Goldt et al. (2020) "Modelling the influence of data structure on learning"; benign-overfitting literature |
| `presets.low_rank_attention`, `zoo.attention` | Low-rank dot-product attention teacher | Cui, Behrens, Krzakala, Zdeborová (2025) "A phase transition between positional and semantic learning in a solvable model of dot-product attention," *J. Stat. Mech.* |
| `zoo.tiny_gpt` | Multi-block causal transformer teacher-student | Attention-indexed models: Cui et al. (2025), arXiv:2506.01582 "Bayes optimal learning of attention-indexed models" |
| `Teacher(init="spiked"/"low_rank"/...)` | Structured teacher weights | Spiked matrix models, Lesieur et al. (2017) |
| `TeacherStudentDataset(input_dist="correlated")` | Structured inputs | Goldt et al. (2020) hidden-manifold model; Loureiro et al. (2021) |
| `metrics.linear_cka` | Representation similarity across architectures | Kornblith et al. (2019) "Similarity of neural network representations revisited" |

### Observables

- **Sample-complexity curves** `E_test(α)`, α = n/d: sharp drops locate learnability transitions (cf. the easy/hard phase diagrams in Zdeborová & Krzakala 2016).
- **Online dynamics** `E_test(t)`, t = #samples/d: plateaus and their escape times diagnose specialization transitions (Saad & Solla 1995).
- **Weight overlap / CKA**: order-parameter analogues that remain well-defined for arbitrary architectures.

## 5. What is exact vs heuristic (summary)

| Component | Guarantee |
|---|---|
| Ridge/MSE replica, LASSO effective noise | Exact in the d → ∞ RS limit |
| Online MSE / perceptron / committee (erf) ODEs | Exact in the d → ∞ limit |
| Online hinge/logistic ODEs | Exact up to numerical quadrature |
| Logistic/probit/hinge/committee replica | Heuristic gradient-flow relaxation (qualitative) |
| `experiment/` measurements | Finite-size numerics; no asymptotic claim |

When comparing simulation with theory, finite-size deviations scale as O(1/√d) for most quantities; increase `d` and the number of seeds before concluding a mismatch.
