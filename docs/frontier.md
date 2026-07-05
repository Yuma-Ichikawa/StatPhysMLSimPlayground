# The numerical frontier: modern learning paradigms as teacher-student physics

`statphys.frontier` treats today's *theoretically intractable* training
paradigms — supervised fine-tuning (SFT), RLHF-style reward optimization,
weak supervision, synthetic-data loops, and in-context learning — as
minimal teacher-student systems, and measures them with the **same order
parameters** (function-space overlaps, generalization error, phase
diagrams) used for the exactly solvable models in the rest of the package.

The philosophy: exact theory (replica, DMFT, ODE dynamics) currently ends
at committee machines, generalized linear models, and shallow feature
maps. The paradigms below are where that theory is headed. Mapping their
phenomenology **numerically first** — with physics-grade observables —
means that when the theory arrives, the phase diagrams are already drawn.

All studies run from the CLI and save a figure + raw JSON:

```bash
statphys study sft            # forgetting / transfer phase diagram
statphys study rlhf           # reward overoptimization (Goodhart)
statphys study weak_to_strong # weak supervisor -> strong student
statphys study collapse       # model collapse under synthetic data
statphys study icl            # emergence of in-context learning
statphys study taxonomy       # teacher structure x paradigm cross table (§6)
```

Add `--quick` for a minutes-scale smoke run. Python entry points:
`run_finetune`, `run_overoptimization`, `run_weak_to_strong`,
`run_collapse`, `run_icl` and their `sweep_*` counterparts, all in
`statphys.frontier`.

Throughout, the basic observable is the normalized function-space overlap
on a probe set,

$$
\hat m[f, g] \;=\;
\frac{\mathbb{E}_x[f(x)\,g(x)]}{\sqrt{\mathbb{E}_x[f(x)^2]\;\mathbb{E}_x[g(x)^2]}}
\;\in\; [-1, 1],
$$

the architecture-agnostic magnetization already used in
[order_parameters.md](order_parameters.md). For MSE regression it carries
the generalization error exactly:
$\varepsilon_g = \tfrac12(\rho_f + q_f - 2 m_f)$.

---

## 1. SFT: fine-tuning as a two-teacher problem

**Setting** (`statphys.frontier.sft`). A tanh-MLP student is pretrained
on task A (teacher $f_A$, $n_{\rm pre} = \alpha_{\rm pre} d$ samples) and
then fine-tuned on task B, whose teacher shares the architecture but has
interpolated weights,

$$
W_B \;=\; \rho\, W_A + \sqrt{1 - \rho^2}\; W_\perp ,
\qquad \rho \in [0, 1],
$$

so the *task similarity* $\rho$ is an exact control dial (built by
`correlated_teacher`). Fine-tuning uses $n_{\rm ft} = \alpha_{\rm ft} d$
fresh task-B samples.

**Order parameters.**

| symbol | definition | meaning |
|---|---|---|
| $m_A(t)$ | $\hat m[f_{\rm student}(t), f_A]$ | retained task-A skill |
| $m_B(t)$ | $\hat m[f_{\rm student}(t), f_B]$ | fine-tuning progress |
| $F$ | $m_A(0^+) - m_A(\infty)$ | catastrophic forgetting |
| $\Delta_B$ | $m_B^{\rm pretrained} - m_B^{\rm scratch}$ | transfer gain (same budget) |

**Phenomenology.** Forgetting $F$ grows toward low similarity and large
fine-tuning budgets; the transfer gain $\Delta_B$ changes sign across the
$(\rho, \alpha_{\rm ft})$ plane, giving a *positive/negative transfer
phase boundary* (cyan contour below). At $\rho = 1$ fine-tuning is just
more data and $F \approx 0$; at $\rho = 0$ the tasks compete for the same
weights and A is erased at a rate set by $\alpha_{\rm ft}$.

![SFT phase diagram](../assets/frontier/sft.png)

---

## 2. RLHF: reward-model overoptimization (Goodhart's law)

**Setting** (`statphys.frontier.rlhf`). The *gold* reward is a nonlinear
teacher plus an **off-distribution penalty** — the part of true utility
that in-distribution preference data can never reveal:

$$
r^\star(x) \;=\; T(x)\;-\;\lambda\,
\mathrm{ReLU}\!\Big(\tfrac1d \lVert x \rVert^2 - \tau\Big),
$$

which vanishes on the base distribution $\mathcal N(0, I_d)$ but makes
genuinely off-distribution inputs bad. A proxy reward model $\hat r$
(student, same architecture as $T$) is trained on
$n_{\rm pref} = \alpha_r d$ *in-distribution* Bradley–Terry pairs,

$$
P(x_1 \succ x_2) = \sigma\!\big(\beta\, [\, r^\star(x_1) - r^\star(x_2)\,]\big).
$$

Two policies optimize against the proxy, each with an exact KL budget:

- **shift policy** (default): $\pi_\mu = \mathcal N(\mu, I_d)$ trained by
  reparameterized ascent on
  $\mathbb E_{\pi_\mu}[\hat r] - c\,\mathrm{KL}(\pi_\mu \| \pi_0)$, with
  $\mathrm{KL} = \lVert\mu\rVert^2/2$; sweeping the penalty $c$ traces
  the optimization-strength axis. This policy *can leave* the data
  distribution — the KL-regularized analogue of PPO-style RLHF.
- **best-of-$n$**: keep the proxy-argmax of $n$ base samples;
  $\mathrm{KL}(n) = \log n - (n-1)/n$ (Hilton & Gao). BoN stays
  supported on the base distribution.

**Order parameters.**

| symbol | definition | meaning |
|---|---|---|
| $G(\mathrm{KL})$ | $\mathbb E_\pi[r^\star]$, z-scored | gold reward under optimization |
| $P(\mathrm{KL})$ | $\mathbb E_\pi[\hat r]$, z-scored | proxy reward (monotone) |
| $P - G$ | hacking gap | Goodhart divergence |
| $\mathrm{KL}^*(\alpha_r)$ | $\arg\max_{\rm KL} G$ | overoptimization onset |
| $G_{\max}(\alpha_r)$ | $\max_{\rm KL} G$ | achievable aligned utility |
| $m_{\rm RM}$ | $\hat m[\hat r, r^\star]$ | reward-model quality (in-distribution) |

**Phenomenology.** Under the shift policy the proxy reward rises
monotonically while the gold reward peaks and *turns over* — the
Goodhart transition, exactly as in the large-scale scans of Gao,
Schulman & Hilton (2023) but with a known ground truth. Both the
turnover point $\mathrm{KL}^*$ and the peak utility $G_{\max}$ grow with
the reward-model data budget $\alpha_r$, tracing a phase boundary
between the *aligned* regime (optimize more) and the *hacked* regime
(optimizing hurts) in the $(\alpha_r, \mathrm{KL})$ plane. BoN at the
same KL shows *no* turnover — it cannot leave the base support — a
sharp numerical statement that **how** the KL budget is spent matters as
much as its size.

![reward overoptimization](../assets/frontier/rlhf.png)

---

## 3. Weak-to-strong generalization

**Setting** (`statphys.frontier.weak_to_strong`). A three-level chain:

1. **truth** $f^*$ — a tanh MLP teacher,
2. **weak supervisor** — a *small-capacity* student trained on
   $\alpha_w d$ true labels (reaches $m_{\rm weak} < 1$),
3. **strong student** — a *large-capacity* net trained **only on the weak
   supervisor's labels** ($\alpha_s d$ fresh inputs), never seeing truth.

This is the statistical core of Burns et al. (2023): can a student
surpass its teacher-of-record by exploiting a better prior?

**Order parameters.**

| symbol | definition | meaning |
|---|---|---|
| $m_{\rm weak}$ | $\hat m[f_{\rm weak}, f^*]$ | supervisor quality |
| $m_{\rm strong}$ | $\hat m[f_{\rm strong}, f^*]$ | student's *true* skill |
| $m_{\rm imit}$ | $\hat m[f_{\rm strong}, f_{\rm weak}]$ | imitation fidelity |
| PGR | $\dfrac{m_{\rm strong} - m_{\rm weak}}{m_{\rm ceil} - m_{\rm weak}}$ | performance gap recovered |

with $m_{\rm ceil}$ the strong architecture trained on true labels at the
same budget.

**Phenomenology.** Points above the diagonal in the
$m_{\rm strong}$-vs-$m_{\rm weak}$ plane are weak-to-strong *gain*: the
strong prior averages out the supervisor's unstructured errors instead of
copying them. The PGR surface over $(\alpha_w, \alpha_s)$ locates the
crossover from *imitation* (PGR $\le 0$, the student just clones the
supervisor including its mistakes) to *generalization beyond the
supervisor* (PGR $> 0$).

![weak-to-strong](../assets/frontier/weak_to_strong.png)

---

## 4. Model collapse under recursive synthetic data

**Setting** (`statphys.frontier.collapse`). Generation 0 trains on
$\alpha d$ true labels. Every later generation trains on labels produced
by the previous generation, with a fraction $p_{\rm real}$ of real
teacher labels mixed in:

$$
y_i = \begin{cases}
 f^*(x_i) + \xi & \text{w.p. } p_{\rm real}\\[2pt]
 f_{g}(x_i) & \text{otherwise,}
\end{cases}
\qquad g \to g+1 .
$$

The teacher-student reduction of Shumailov et al. (2024): does a
self-consuming training loop lose the signal?

**Order parameters.**

| symbol | definition | meaning |
|---|---|---|
| $m(g)$ | $\hat m[f_g, f^*]$ | surviving signal at generation $g$ |
| $q(g)/q(0)$ | output-variance ratio | tail / diversity loss (collapse signature) |
| $m(g_{\max})$ vs $p_{\rm real}$ | terminal overlap | anchoring phase boundary |

**Phenomenology.** With $p_{\rm real} = 0$ the overlap decays generation
by generation (each retraining adds an error random walk that is never
corrected) and the output variance shrinks — the two standard collapse
signatures. A modest real-data fraction pins the fixed point near the
teacher: the terminal-overlap curve vs $p_{\rm real}$ is the *collapse
boundary*.

![model collapse](../assets/frontier/collapse.png)

---

## 5. Emergence of in-context learning

**Setting** (`statphys.frontier.icl`). A small causal transformer is
pretrained on prompts of linear-regression examples,

$$
(x_1, y_1, \dots, x_k, y_k, x_{\rm q}) \mapsto y_{\rm q},
\qquad y = w^\top x / \sqrt d,
$$

where each prompt's task $w$ is drawn from a **finite pool** of
$N_{\rm tasks}$ fixed teachers (Raventós et al. 2023). Evaluation
distinguishes *seen* tasks (from the pool) from *unseen* tasks
($w \sim \mathcal N(0, I_d)$) — solving unseen tasks requires genuine
in-context regression, not retrieval.

**Order parameters.**

| symbol | definition | meaning |
|---|---|---|
| $S_{\rm ICL}$ | $1 - \varepsilon_{\rm unseen} / \varepsilon_{\rm null}$ | in-context learning ability |
| $S_{\rm memo}$ | $1 - \varepsilon_{\rm seen} / \varepsilon_{\rm null}$ | pool memorization |
| ridge alignment | $\mathrm{corr}[f_{\rm TF}, f_{\rm ridge}]$ on unseen prompts | which *algorithm* the net implements |

where $f_{\rm ridge}$ is the Bayes-optimal in-context ridge predictor
computed from the same prompt (`ridge_predictor`).

**Phenomenology.** An *algorithmic transition* in $N_{\rm tasks}$: below
a task-diversity threshold the network memorizes the pool
($S_{\rm memo}$ high, $S_{\rm ICL} \approx 0$); above it, a qualitatively
different solution takes over — unseen-task error collapses onto the
ridge curve and the ridge alignment jumps toward 1. This is an emergence
transition in the space of algorithms, well outside current exact theory.

![ICL emergence](../assets/frontier/icl.png)

---

## 6. The teacher taxonomy: teacher structure × paradigm

**Setting** (`statphys.frontier.teachers` + `statphys.frontier.taxonomy`).
Every paradigm above takes a teacher as input, and the *statistics of
that teacher* — weight structure, and the data manifold it acts on — are
a physics axis of their own (structured weights and hidden-manifold
inputs shift transitions in the exactly solvable settings; the same
question is open for the frontier paradigms). The taxonomy makes this
axis explicit and table-managed:

| name | family | weights | inputs |
|---|---|---|---|
| `random_mlp` | random | i.i.d. Gaussian | Gaussian |
| `linear` | random | single index (Gaussian vector) | Gaussian |
| `sparse_mlp` | structured | 90% zeros (compressible) | Gaussian |
| `low_rank_mlp` | structured | rank-2 factorized | Gaussian |
| `power_law_mlp` | structured | heavy-tailed (Pareto-like) | Gaussian |
| `spiked_mlp` | structured | Gaussian + rank-1 spike | Gaussian |
| `binary_mlp` | structured | Rademacher ±1 | Gaussian |
| `random_digits` | structured | i.i.d. Gaussian | real images (sklearn digits) |
| `trained_digits` | trained | **trained on digit parity** | real images (sklearn digits) |

`trained_digits` is a network genuinely trained on real 8×8 digit images
(parity regression, accuracy > 0.9), evaluated on the real image
manifold; `random_digits` is its control — same (real) input statistics,
unstructured weights — so the *trained-weights effect* and the
*real-data effect* can be separated.

Every entry returns `(teacher, input_sampler, d)` and plugs into every
paradigm through the `teacher=` / `input_sampler=` arguments of
`run_finetune`, `run_overoptimization`, `run_weak_to_strong`,
`run_collapse`.

**The cross experiment** (`statphys study taxonomy`) probes every
(teacher, paradigm) cell with one headline order parameter:

| paradigm | probe conditions | headline order parameter |
|---|---|---|
| `sft` | $\rho = 0.5$, $\alpha_{\rm ft} = 4$ | forgetting $F$ |
| `rlhf` | $\alpha_r = 8$, shift policy | overoptimization onset $\mathrm{KL}^*$ |
| `w2s` | $\alpha_w = 4$, $\alpha_s = 16$ | performance gap recovered (PGR) |
| `collapse` | $p_{\rm real} = 0$, 8 generations | signal drop $m(0) - m(g_{\max})$ |

and renders a bar-panel figure plus a markdown table
(`phase_results/taxonomy.md`), with cells reported as mean ± std over
seeds. The physics question: are the frontier phenomena *universal*
across teacher structure (as Gaussian universality would suggest for
weak structure), or does planted structure (low-rank, spiked, trained
weights) shift the boundaries — and in which direction?

![taxonomy](../assets/frontier/taxonomy.png)

**Observed** (d = 32–64, 2 seeds; see `phase_results/taxonomy.md` for
the numbers). Teacher structure is *not* a spectator variable — it moves
every frontier boundary, and in different directions per paradigm:

- **RLHF hackability tracks teacher complexity.** The linear teacher is
  hardest to hack (KL\* ≈ 17): a proxy trained on its preferences stays
  faithful far out-of-distribution. The low-rank teacher is the most
  hackable (KL\* ≈ 0.4, a ~40× shift of the onset), with structured
  nonlinear teachers in between (KL\* ≈ 2–4). The more structured and
  nonlinear the gold reward, the earlier Goodhart sets in.
- **Weak-to-strong gains die for trained real-data teachers.** PGR is
  large for the linear teacher (~0.7) and moderate for most structured
  ones (~0.5), but consistently ≈ 0 for the digits-trained teacher —
  the weak supervisor's errors on a real manifold are apparently
  inherited rather than averaged out.
- **Collapse speed is teacher-dependent.** Sparse and digits-trained
  teachers lose signal fastest under the synthetic loop
  (Δm ≈ 0.12 in 8 generations) while the spiked teacher barely decays
  (Δm ≈ 0.01): functions concentrated on few directions survive
  self-distillation, high-entropy ones do not.
- **Forgetting is comparatively universal** (F ≈ 0.4–0.8 across the
  whole taxonomy), consistent with SFT forgetting being a property of
  the *protocol* (interference of the fine-tuning task) more than of
  the teacher's weight statistics.

(The ICL setting is excluded from the cross: its "teacher" is a task
distribution rather than a single network, so the weight-structure axis
does not apply.)

---

## Design notes (for extending the frontier)

- **One shared toolbox.** All settings use `frontier/common.py` — one
  training loop (`train_regression`), one overlap
  (`model_overlap`/`output_overlap`), one correlated-teacher constructor
  — so a new paradigm is one new module with a `run_*` and a `sweep_*`
  function plus a `study_*` entry in `frontier/studies.py`.
- **Teachers and paradigms are orthogonal.** Data distributions enter
  only through `InputSampler` (a function `n -> (n, d)`) and teachers
  through the `teacher=` argument, so adding a teacher to
  `TEACHER_TAXONOMY` (e.g. a network trained on text windows or CIFAR
  patches) immediately makes it available to *every* paradigm and to
  the cross experiment.
- **Registry integration.** `FRONTIER_STUDIES` is merged into the main
  `STUDIES` registry, so every new study is automatically available via
  `statphys study <name>` and `statphys list`.
- **Same observables as theory.** Wherever the exactly solvable settings
  use $m, q, \varepsilon_g$, the frontier settings use the function-space
  analogues, so numerical phase diagrams here are directly comparable to
  future replica/DMFT results.
- **Candidate extensions** (same pattern): DPO vs RLHF loss geometry,
  multi-agent self-play with population order parameters, RL policy
  distillation in contextual bandits, curriculum-order phase effects,
  and KL-regularized policy optimization replacing BoN.

## References

- Gao, Schulman, Hilton, *Scaling Laws for Reward Model
  Overoptimization*, ICML 2023.
- Burns et al., *Weak-to-Strong Generalization*, 2023.
- Shumailov et al., *AI models collapse when trained on recursively
  generated data*, Nature 2024.
- Garg, Tsipras, Liang, Valiant, *What Can Transformers Learn
  In-Context?*, NeurIPS 2022.
- Raventós, Paul, Chen, Ganguli, *Pretraining task diversity and the
  emergence of non-Bayesian in-context learning for regression*,
  NeurIPS 2023.
- Zdeborová & Krzakala, *Statistical physics of inference*, Adv. Phys.
  2016 — the theory program these experiments run ahead of.
