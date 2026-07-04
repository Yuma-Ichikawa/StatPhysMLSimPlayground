# Order Parameters and Generalization Error — Full Mathematical Reference

This document collects, with derivations, **every order parameter and
generalization-error formula** used by the theory-free experiment
framework (`statphys.experiment`), from the classical single-direction
teacher-student setting through the "realistic settings" extensions
(multi-index models, Gaussian-mixture classification, lazy/rich feature
learning, LoRA-style fine-tuning). It complements
[concepts.md](concepts.md) (which covers the exact-theory modules,
`statphys.theory`) and [experiments.md](experiments.md) (which covers
the Python API).

The guiding design principle, following the classical statistical
mechanics of learning (Gardner & Derrida 1989; Seung, Sompolinsky &
Tishby 1992; Engel & Van den Broeck 2001) and its modern high-dimensional
revival (Zdeborová & Krzakala 2016; Advani, Saxe & Ganguli 2020; Goldt,
Advani, Saxe, Krzakala & Zdeborová 2019/2020; Gerace, Loureiro, Krzakala,
Mézard & Zdeborová 2020; Mignacco, Krzakala, Mézard, Urbani & Zdeborová
2020) is: **define all order parameters in function space** (on a shared
"probe" set of inputs), so that they apply to *any* architecture, not
only to models with a single weight vector — this is what makes the
package's numerics meaningful for MLPs, CNNs, attention blocks and tiny
transformers alike, where no replica computation is tractable.

---

## 1. Setup and notation

A teacher $f_t$ and a student $f_s$ are arbitrary functions
$\mathbb{R}^d \to \mathbb{R}$ (implemented as `nn.Module`s). Data is
generated as

$$
y = f_t(x) + \sigma \, \xi, \qquad \xi \sim \mathcal N(0,1),
$$

for regression (`readout="identity"`, label noise std $\sigma$ =
`noise_std`), or $y = \mathrm{sign}(f_t(x))$ with an optional label flip
probability $p_{\rm flip}$ for classification (`readout="sign"`).
Training uses $n$ samples; the fundamental control parameter throughout
the package is the **sample ratio**

$$
\alpha = n / d.
$$

All function-space order parameters below are estimated by Monte Carlo
on a shared **probe set** $\{x_i\}_{i=1}^{n_{\rm probe}}$ (independent of
the training data), implemented in `statphys.experiment.observables`.

---

## 2. Function-space overlap order parameters

### 2.1 Magnetization, self-overlap, and $\hat m$

For any student $f_s$ (`function_order_params`):

$$
m_f = \mathbb E_x[f_s(x) f_t(x)], \qquad
q_f = \mathbb E_x[f_s(x)^2], \qquad
\rho_f = \mathbb E_x[f_t(x)^2],
$$

where $f_t = $ `teacher.clean` (the noiseless teacher output — noise is
added only when *generating labels*, never when *evaluating* order
parameters). The normalized overlap ("magnetization")

$$
\hat m = \frac{m_f}{\sqrt{q_f \, \rho_f}} \in [-1, 1]
$$

is the function-space generalization of the classical $m/\sqrt{q\rho}$
of a single-direction perceptron (`concepts.md`): $\hat m = 1$ means the
student implements exactly the teacher's function (up to positive
rescaling of the readout); $\hat m \approx 0$ means the student is
uncorrelated with the teacher (typically the case for $\alpha \ll 1$
or before the specialization transition of a committee machine).

**Linear special case.** If $f_s(x) = w^\top x/\sqrt d$ and
$f_t(x) = w_0^\top x /\sqrt d$ with $x \sim \mathcal N(0, I_d)$, then
$m_f = w^\top w_0 / d = m$, $q_f = w^\top w / d = q$, $\rho_f = w_0^\top
w_0/d = \rho$ exactly recover the classical order parameters of
`concepts.md`, so $\hat m$ generalizes $m/\sqrt{q\rho}$ without change
of meaning.

### 2.2 Generalization error and its exact decomposition

The (label-noise-free) generalization error for MSE regression is,
**exactly**,

$$
\epsilon_g^{\rm clean}
= \tfrac12 \, \mathbb E_x\!\left[(f_s(x) - f_t(x))^2\right]
= \tfrac12 \left(\rho_f + q_f - 2 m_f\right),
\tag{2.1}
$$

by expanding the square — the direct function-space analogue of the
classical $\epsilon_g = \tfrac12(\rho - 2m + q)$ in `concepts.md`. With
label noise (`noise_std` $=\sigma$), the *measured* test MSE on noisy
labels is $\epsilon_g^{\rm clean} + \sigma^2/2$; `test_error` (used in
`run_sample_complexity` / `run_order_parameters`) always evaluates on
**freshly sampled noisy labels**, matching what a practitioner would
measure, while $\hat m$, $q_f$, $\rho_f$ are noise-free by
construction (`Teacher.clean`) — noise shifts $\epsilon_g$ but never
$\hat m$ (see `test_noise_does_not_affect_m_hat`).

`generalization_error_decomposition(student, teacher, X)` computes
**both sides of (2.1) directly** and returns their difference
("residual"), so the identity can be checked numerically to floating
precision — this is the package's primary sanity check that the
generalization error is being bookkept correctly (see
`tests/test_realistic_extensions.py::TestGeneralizationErrorDecomposition`
and the "properly check generalization error" requirement this
addresses directly).

For binary classification with a linear readout, the exact 0-1
generalization error of a linear classifier is (see §5 for a fully
worked, numerically verified example)

$$
\epsilon_g = \frac{1}{\pi}\arccos\!\left(\frac{m}{\sqrt{q\rho}}\right)
\quad\text{(matched-noise-free teacher/student angle)},
$$

reproduced from `concepts.md`; §5 below derives and numerically
verifies the *generative* (Gaussian-mixture) analogue,
$\epsilon_g = \Phi(-\mu \hat m)$.

### 2.3 Replica overlap, susceptibility, Binder cumulant

Train $R$ independent "replicas" (students) at the same $\alpha$
(optionally sharing training data — same quenched disorder, different
initialization/SGD path). For replicas $a, b$:

$$
q_{ab} = \mathbb E_x[f_a(x) f_b(x)], \qquad a \ne b,
$$

averaged over all $\binom{R}{2}$ pairs to give `q_ab_mean`/`q_ab_std`
(`replica_overlaps`). This is the **numerical analogue of the
replica-symmetric order parameter**: $q_{ab} \to 1$ means all replicas
converge to (essentially) the same function — a single dominant basin
("condensed"/RS phase) — while small $q_{ab}$ indicates many distinct
minima (a "glassy"/RSB-like multiplicity of solutions), exactly the
numerical diagnostic used by Zdeborová & Krzakala-style analyses to
locate phase transitions without solving saddle-point equations.

Over the $R$ per-alpha values of $\hat m$, define the **susceptibility**

$$
\chi_m = d \cdot \mathrm{Var}_{\rm replicas}[\hat m]
\tag{2.2}
$$

(`susceptibility`, with `scale=d`) — the finite-size numerical proxy
for the thermodynamic susceptibility, expected to **peak at a
continuous phase transition** and grow with $d$ there (critical
slowing-down / diverging fluctuations).

The **Binder cumulant**

$$
U_4 = 1 - \frac{\langle \hat m^4 \rangle}{3 \langle \hat m^2\rangle^2}
\tag{2.3}
$$

(`binder_cumulant`) is $0$ for a centered Gaussian order parameter
(disordered phase) and $2/3$ for a delta-distributed one (fully ordered
phase); curves $U_4(\alpha)$ computed at different $d$ **cross at the
critical point** $\alpha_c$ — the standard finite-size-scaling trick
used in `study_fss`/`study_diagram`.

### 2.4 Participation ratio and specialization index

For a batch of hidden-layer activations $H \in \mathbb R^{n\times k}$
with covariance eigenvalues $\lambda_1,\dots,\lambda_k$, the
**participation ratio**

$$
\mathrm{PR} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}
\in [1, k]
\tag{2.4}
$$

(`participation_ratio`) measures the *effective dimension* of the
representation: $\mathrm{PR}=k$ for isotropic activations, $\mathrm
{PR}\to 1$ for a fully collapsed (rank-1) representation — used to
track representation collapse/expansion across training or width.

For committee-style architectures with matched hidden width $K$, let
$M \in \mathbb R^{K\times K}$ be the (best-permutation) matched
absolute overlap matrix between student and teacher hidden units. The
**specialization index**

$$
S = \bar M_{\rm diag} - \bar M_{\rm off\text{-}diag}
\tag{2.5}
$$

(`specialization_index`) is the mean matched-pair overlap minus the
mean unmatched overlap: $S \to 1$ once each student unit has
specialized to a distinct teacher unit (the committee-machine
specialization transition studied in `study_committee`), $S \approx 0$
when hidden units remain a generic, non-specialized mixture.

---

## 3. Multi-index models and subspace overlap

A **multi-index model** generalizes the single-direction (perceptron)
teacher to $K>1$ relevant directions,

$$
y = g\!\left(W_0^\top x\right) + \sigma\xi, \qquad W_0 \in \mathbb R^{d\times K},
$$

with $g:\mathbb R^K\to\mathbb R$ possibly unknown (in the package,
$g$ is realized by a small MLP: hidden nonlinearity + linear readout).
This is the object of intense recent theoretical interest (single- and
multi-index models with $K = O(1)$ "relevant directions" embedded in
$d\to\infty$ ambient dimensions; Ben Arous, Gerace & Jacot-style
analyses; Gerace et al. 2020; Damian, Lee & Soltanolkotabi 2022;
Bietti, Bruna, Sanford & Song 2022) because it is the simplest model
capturing **feature learning**: a network must discover the $K$-
dimensional relevant subspace of $x$ before it can fit $y$.

Because $g$ can mix the $K$ directions arbitrarily, the natural order
parameter is not a per-direction cosine but the **principal angles**
between the teacher subspace $\mathrm{span}(W_0)$ and the student's
learned relevant subspace $\mathrm{span}(W)$ (first-layer weights,
$W\in\mathbb R^{d\times K_s}$, possibly with $K_s \ne K$):

1. Orthonormalize each weight matrix's row space via QR:
   $W_0 = Q_0 R_0$, $W = QR$ (columns of $Q_0\in\mathbb R^{d\times K}$,
   $Q\in\mathbb R^{d\times K_s}$ are orthonormal bases of the two
   subspaces).
2. The singular values of $Q^\top Q_0 \in \mathbb R^{K_s\times K}$ are
   $\cos\theta_1 \ge \dots \ge \cos\theta_{\min(K,K_s)}$, the cosines of
   the **principal angles** between the subspaces (Golub & Van Loan).

`subspace_overlap(W_student, W_teacher)` returns `cosines`,
`mean_cosine` (the natural scalar order parameter, $=1$ iff the
subspaces coincide, $=0$ iff they intersect trivially), and
`top_cosine` (best single aligned pair). Unlike per-unit
`specialization_index`, this is **basis- and permutation-invariant** —
it does not require the student and teacher to share the exact number
of hidden units, so recovery can be studied directly as a function of
over-/under-parameterization ($K_s \gtrless K$); see `study_multi_index`.

> **Caveat (important for interpretation).** `mean_cosine` averages
> over only $\min(K_s, K)$ principal angles. An under-parameterized
> student ($K_s < K$) is only ever compared on its best-aligned
> $K_s$-dimensional slice of the teacher subspace, which can yield a
> *deceptively high* `mean_cosine` relative to a matched or
> over-parameterized student that must (and does) align more
> directions simultaneously. Always read `subspace_overlap` together
> with $\epsilon_g$, and prefer comparing models at matched $K_s$ when
> possible.

The preset is `multi_index_model(d, k_teacher, k_student, ...)`.

---

## 4. Lazy vs. rich regimes: weight movement

`run_order_parameters(..., init_scale=s)` multiplies **all** initial
student parameters $\theta_0$ by $s$ before training, and always
records

$$
\text{weight\_movement} =
\frac{\lVert \theta_{\rm final} - \theta_0 \rVert_2}{\lVert \theta_0 \rVert_2}
\tag{4.1}
$$

— a parametrization-agnostic diagnostic of how far training moves the
weights *relative to their own scale*. This operationalizes the
**lazy training / NTK regime vs. rich / feature-learning regime**
dichotomy of Chizat, Oyallon & Bach (2019): scaling up the initial
weights by $s$ shrinks the *relative* displacement needed to fit the
data by $O(1/s)$ in the large-$s$ limit, effectively linearizing
training around the (random-feature) initialization — the model
behaves like its NTK/kernel approximation and **cannot learn new
features**. Small $s$ (standard init) allows genuine feature learning:
the relevant-direction subspace can be discovered from data (large
`subspace_overlap`/`specialization_index` growth), at the cost of a
non-convex, harder optimization landscape.

`study_lazy_rich` sweeps $s$ at fixed $\alpha$ for a tanh-committee
teacher and shows the expected monotonic relationship: weight movement
decreases roughly as $1/s$, while $\epsilon_g$ and $1-\hat m$ increase
sharply once the student is pinned in the lazy regime and can no
longer specialize to the teacher's $K$ relevant directions. This is a
numerically light diagnostic of the same physics discussed in
mean-field vs. NTK parametrization comparisons (e.g., Yang & Hu's
$\mu$P framework, and DMFT-based analyses of feature learning by
Zdeborová's group and collaborators; see [THEORY.md](THEORY.md)).

---

## 5. Gaussian-mixture classification: an exactly checkable $\epsilon_g$

Every other setting in this package is **discriminative**: $x$ is
drawn first, then $y = f_t(x)$. The classification-of-mixtures model
(Deng, Kammoun & Thrampoulidis 2019; Mai & Liao 2019; Mignacco,
Krzakala, Mézard, Urbani & Zdeborová 2020, "The role of regularization
in classification of high-dimensional noisy Gaussian mixture") is
**generative**: the label comes first,

$$
y \sim \mathrm{Unif}\{-1, +1\}, \qquad
x = y\, \mu\, v + z, \quad z\sim\mathcal N(0, I_d),
\tag{5.1}
$$

for a fixed unit "cluster axis" $v \in \mathbb R^d$ ($\lVert v\rVert=1$)
and separation $\mu$ (`GaussianMixtureDataset`). It is used here
specifically because it has an **exact, closed-form generalization
error**, which makes it an ideal correctness check for the numerical
machinery.

**Derivation.** For a linear classifier $x \mapsto \mathrm{sign}(w^\top x)$
with $w$ fixed, the decision variable

$$
z_w = \frac{w^\top x}{\lVert w\rVert}
= y\,\mu\, \frac{w^\top v}{\lVert w \rVert} + \frac{w^\top z}{\lVert w\rVert}
\sim \mathcal N\!\left(y\,\mu\cos\theta,\; 1\right),
$$

where $\cos\theta = w^\top v /(\lVert w\rVert \lVert v\rVert) =$
`vector_overlap(w, v)`, since $w^\top z/\lVert w\rVert \sim \mathcal
N(0,1)$ for any fixed direction $w$. The classifier errs when $z_w$ has
the wrong sign relative to $y$, i.e. with probability

$$
\boxed{\ \epsilon_g(w) = \Phi\!\left(-\mu \cos\theta\right)\ },
\qquad \Phi(\cdot) = \text{standard normal CDF}.
\tag{5.2}
$$

At the Bayes-optimal classifier $w = v$ ($\cos\theta = 1$), this is the
**Bayes error** $\epsilon_g^\star = \Phi(-\mu)$
(`bayes_error(mu)`); at chance level ($\cos\theta = 0$),
$\epsilon_g = \Phi(0) = 1/2$, as expected.

`mixture_classification(d, mu)` pairs this dataset with an
`oracle_teacher()` whose `clean(x) = sign(v \cdot x)` (the
Bayes-consistent rule), and registers `cluster_overlap` $= \cos\theta$
as a custom metric, so that `run_order_parameters` can be used exactly
as in the discriminative settings (`TeacherStudentExperiment(...,
dataset=...)`, §7). `study_mixture` overlays the *measured* test error
against $\Phi(-\mu\cos\theta)$ evaluated at the *measured* overlap, and
`tests/test_realistic_extensions.py::TestGaussianMixtureBayesError`
verifies (5.2) numerically (both at $w=v$ and at a random $w$) to
within Monte Carlo error — the package's most direct, literature-
grounded validation that generalization error is computed and
interpreted correctly.

---

## 6. LoRA-style low-rank fine-tuning

Low-Rank Adaptation (LoRA; Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang
& Chen 2021) freezes a large "pretrained" weight matrix $W_0$ and
learns only a low-rank update $\Delta = BA$, $B\in\mathbb R^{h\times
r}$, $A \in \mathbb R^{r\times d}$, $r \ll d$, initialized at $B=0$ so
that $\Delta=0$ at the start of fine-tuning. This is now the dominant
paradigm for adapting large pretrained models and is a natural target
for a teacher-student treatment: the fine-tuning **task** itself is a
teacher, of which we can control the (true) rank.

The preset `lora_finetune(d, hidden, rank_true, rank_student)`
constructs:

- a frozen random backbone $W_0 \in \mathbb R^{h\times d}$, shared
  ("pretrained") by both teacher and student;
- a **teacher** update $\Delta_t = B_t A_t$ of fixed rank
  `rank_true`, drawn once (not zero-initialized — it represents "the
  correctly fine-tuned model");
- a **student** update $\Delta_s = B_s A_s$ of rank `rank_student`,
  zero-initialized (as in real LoRA) and trained from
  $n = \alpha d$ "fine-tuning" samples, with $W_0$ frozen throughout
  (`requires_grad=False`).

Both use $h = W_0(x)$, $y = c^\top \tanh(( W_0+\Delta)x)$ for a
trainable readout head $c$ (fine-tuned jointly with the adapter, as is
common practice). The natural order parameter is the **matrix cosine
overlap** between the learned and true low-rank updates,

$$
\text{adapter\_overlap} =
\frac{\langle \Delta_s, \Delta_t\rangle_F}
{\lVert \Delta_s\rVert_F \, \lVert \Delta_t \rVert_F}
= \texttt{vector\_overlap}(\Delta_s, \Delta_t),
\tag{6.1}
$$

i.e. `vector_overlap` applied to the flattened matrices (Frobenius
inner product) — the direct analogue of $\hat m$ for **low-rank matrix
recovery** rather than vector recovery, structurally the same
recovery-transition question as `spiked_teacher` (rank-1 BBP spike) and
`sparse_teacher` (compressed sensing), now with the model-mismatch
axis (`rank_student` vs. `rank_true`) made explicit. `study_lora`
sweeps $\alpha = n_{\rm finetune}/d$ and `rank_student`, tracking
`adapter_overlap` and $\epsilon_g$ — matched rank gives the cleanest
recovery at large $\alpha$; too-small rank caps achievable overlap;
too-large rank has more free parameters to fit noise at small $\alpha$
(a rank-analogue of the double-descent width sweep in
`study_double_descent`).

---

## 7. Custom generative datasets: `dataset=...`

Every setting above except Gaussian-mixture classification is
*discriminative* ($x$ sampled, then $y=f_t(x)$), implemented by
`TeacherStudentDataset`. `TeacherStudentExperiment(..., dataset=my_ds)`
accepts any object implementing

```python
class MyDataset:
    def sample(self, n) -> tuple[Tensor, Tensor]: ...        # (X, y)
    def sample_inputs(self, n) -> Tensor: ...                 # X only
    def get_config(self) -> dict: ...                         # for logging
```

in place of the default, enabling **generative** data models (label
determines $x$, as in §5) while still using every protocol
(`run_sample_complexity`, `run_order_parameters`,
`run_training_dynamics`) and order parameter in this document
unmodified. `teacher` is still required and must expose a `.clean(x)`
oracle consistent with the generative process (for the Gaussian
mixture, `sign(v \cdot x)`, the Bayes-consistent rule) so that
`function_order_params` / $\hat m$ remain meaningful.

---

## 8. Summary table: setting → order parameter → formula

| Setting | Preset / study | Order parameter | Formula |
|---|---|---|---|
| Linear / sparse / spiked teacher | `sparse_teacher`, `spiked_teacher` | $\hat m$, $\epsilon_g$ | (2.1), linear $f$ |
| Committee machine | `random_mlp`, `study_committee` | specialization $S$ | (2.5) |
| Attention / tiny GPT | `low_rank_attention`, `tiny_gpt` | $\hat m$, $\epsilon_g$ (no closed form) | (2.1) |
| Hidden manifold | `hidden_manifold`, `study_manifold` | $\hat m$, $\epsilon_g$ vs. latent dim | (2.1) |
| Grokking (epoch time) | `run_training_dynamics`, `study_grokking` | $\hat m(t)$, train/test $\epsilon(t)$ | (2.1) at each epoch |
| Universality | `study_universality` | $\epsilon_g(\alpha)$ across input laws | (2.1) |
| Double descent | `study_double_descent` | $\epsilon_g$ vs. width | (2.1) |
| Scaling laws | `study_scaling` | exponent $b$ in $\epsilon_g\sim\alpha^{-b}$ | fit of (2.1) |
| Multi-index model | `multi_index_model`, `study_multi_index` | subspace overlap | §3 |
| Lazy vs. rich | `study_lazy_rich` | weight movement | (4.1) |
| Gaussian-mixture classification | `mixture_classification`, `study_mixture` | cluster overlap $\cos\theta$, $\epsilon_g$ | (5.2), verified |
| LoRA fine-tuning | `lora_finetune`, `study_lora` | adapter overlap | (6.1) |

---

## References

- Gardner, E. & Derrida, B. (1989). Three unfinished works on the
  optimal storage capacity of networks. *J. Phys. A*.
- Seung, H. S., Sompolinsky, H. & Tishby, N. (1992). Statistical
  mechanics of learning from examples. *Phys. Rev. A*.
- Engel, A. & Van den Broeck, C. (2001). *Statistical Mechanics of
  Learning*. Cambridge University Press.
- Zdeborová, L. & Krzakala, F. (2016). Statistical physics of
  inference: thresholds and algorithms. *Advances in Physics*.
- Advani, M. S., Saxe, A. M. & Ganguli, S. (2020). High-dimensional
  dynamics of generalization error in neural networks. *Neural
  Networks*.
- Goldt, S., Mézard, M., Krzakala, F. & Zdeborová, L. (2020). Modeling
  the influence of data structure on learning in neural networks: the
  hidden manifold model. *Phys. Rev. X*.
- Gerace, F., Loureiro, B., Krzakala, F., Mézard, M. & Zdeborová, L.
  (2020). Generalisation error in learning with random features and
  the hidden manifold model. *ICML*.
- Ben Arous, G., Gheissari, R. & Jagannath, A. (2021+); Damian, A.,
  Lee, J. & Soltanolkotabi, M. (2022); Bietti, A., Bruna, J., Sanford,
  C. & Song, M. J. (2022). Multi-index models and feature learning
  (representative works on the multi-index model literature).
- Mignacco, F., Krzakala, F., Mézard, M., Urbani, P. & Zdeborová, L.
  (2020). The role of regularization in classification of
  high-dimensional noisy Gaussian mixture. *ICML*.
- Deng, Z., Kammoun, A. & Thrampoulidis, C. (2019); Mai, X. & Liao, Z.
  (2019). High-dimensional Gaussian-mixture classification asymptotics.
- Chizat, L., Oyallon, E. & Bach, F. (2019). On lazy training in
  differentiable programming. *NeurIPS*.
- Yang, G. & Hu, E. (2021). Feature learning in infinite-width neural
  networks (the $\mu$P parametrization).
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
  Wang, L. & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large
  Language Models. *arXiv:2106.09685*.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I. & Misra, V. (2022).
  Grokking: Generalization beyond overfitting on small algorithmic
  datasets. *arXiv:2201.02177*.
- Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019). Reconciling modern
  machine-learning practice and the classical bias-variance trade-off.
  *PNAS*. Nakkiran, P. et al. (2021). Deep double descent. *JSTAT*.

See also [THEORY.md](THEORY.md) for the mapping between the exact
replica/online-learning theory modules (`statphys.theory`) and the
same literature.
