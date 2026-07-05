# General Teacher-Student Experiments

The `statphys.experiment` subpackage runs **theory-free** numerical experiments: any PyTorch module (MLPs, CNNs, LSTMs, attention blocks, LLM-style transformers, ...) can act as teacher and/or student, and phase-transition-like phenomena are measured purely numerically — no replica or ODE solution required.

## Basic usage

```python
import torch.nn as nn
from statphys.experiment import Teacher, TeacherStudentExperiment

# Teacher: any nn.Module, with structured weights
teacher = Teacher(
    nn.Sequential(nn.Linear(200, 32), nn.ReLU(), nn.Linear(32, 1)),
    init="low_rank",              # see "Weight structures" below
    init_kwargs={"rank": 4},
    noise_std=0.05,               # label noise (regression)
)

exp = TeacherStudentExperiment(
    teacher=teacher,
    student_factory=lambda: nn.Sequential(
        nn.Linear(200, 32), nn.ReLU(), nn.Linear(32, 1)),
    input_dist="correlated",      # see "Input distributions" below
    input_kwargs={"ar_coeff": 0.5},
)

# Sample-complexity sweep: transitions appear as sharp error drops
result = exp.run_sample_complexity(alphas=[0.5, 1, 2, 4, 8, 16], n_seeds=5)
result.plot(logy=True)

# Single-pass online SGD in normalized time t = #samples/d
result = exp.run_online(t_max=50, lr=0.1)
```

`ExperimentResult` stores per-seed records for every metric; `result.mean(name)` / `result.std(name)` return seed statistics, and `result.to_dict()` round-trips through JSON.

## Teacher weight structures

`Teacher(init=...)` (and the standalone `init_weights_`) supports:

| `init` | Structure | Typical phenomenon |
|---|---|---|
| `"normal"` | i.i.d. Gaussian | Classic random teacher |
| `"sparse"` | Fraction `sparsity` of entries zeroed | Compressed-sensing-like recovery |
| `"low_rank"` | $W = UV^\top/\sqrt{r}$ | Structured/attention-like teachers |
| `"orthogonal"` | Orthonormal rows/columns | Isometric propagation |
| `"power_law"` | Heavy-tailed (Pareto-like), exponent `alpha` | Outlier-dominated teachers |
| `"binary"` | Rademacher $\pm$ scale | Discrete perceptron teachers |
| `"spiked"` | Gaussian + rank-1 spike of strength `snr` | BBP-style detectability transitions |

Readouts: `readout="identity"` (regression, optional `noise_std`), `readout="sign"` (binary classification, optional `flip_prob`), or any callable.

## Input distributions

`TeacherStudentDataset(input_dist=...)`:

| `input_dist` | Description |
|---|---|
| `"gaussian"` | $x \sim \mathcal{N}(0, I_d)$ |
| `"correlated"` | $x \sim \mathcal{N}(0, C)$ with `cov` matrix or AR(1) `ar_coeff` |
| `"rademacher"` | $x_i \in \{-1, +1\}$ |
| `"sphere"` | Uniform on the sphere of radius $\sqrt{d}$ |
| `"hidden_manifold"` | $x = \phi(zF/\sqrt{D})$ with latent $z \sim \mathcal{N}(0, I_D)$ — the hidden manifold model (Goldt et al. 2020) for realistic low-dimensional data structure. Options: `latent_dim`, `nonlinearity` (`tanh`/`relu`/`sign`/`identity`), optional `feature_map` |
| callable | Any function `n -> (n, d)` tensor |

For **generative** settings where the label determines $x$ rather than
the other way around (e.g. Gaussian-mixture classification, §5 of
[order_parameters.md](order_parameters.md)), pass a fully custom
`dataset=...` object to `TeacherStudentExperiment` instead of
`input_dist` — see "Custom generative datasets" below.

## Physics order parameters

`statphys.experiment.observables` defines statistical-physics observables in
*function space* (on a shared probe set), so they apply to any architecture,
not only to models with a single weight vector. **Full derivations and the
exact generalization-error identities are in
[order_parameters.md](order_parameters.md)** — this table is a quick
reference:

| Observable | Definition | Physics reading |
|---|---|---|
| `function_order_params` | \(m_f = \mathbb{E}[f_s f_t]\), \(q_f = \mathbb{E}[f_s^2]\), and \(\hat m = m_f/\sqrt{q_f \rho_f}\) | magnetization (teacher recovery); noise-independent |
| `generalization_error_decomposition` | \(\epsilon_g = \tfrac12(\rho_f+q_f-2m_f)\), checked against the direct MSE | exact identity; validates the eps_g bookkeeping |
| `replica_overlaps` | \(q_{ab} = \mathbb{E}[f_a f_b]\) between independently trained students | replica-symmetric overlap; \(q_{ab} \to 1\) = condensed phase, small = many distinct minima |
| `susceptibility` | \(\chi_m = d \, \mathrm{Var}[\hat m]\) over replicas | peaks at the transition, sharpens with \(d\) |
| `binder_cumulant` | \(U_4 = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2\) | curves for different \(d\) cross at \(\alpha_c\) (finite-size scaling) |
| `participation_ratio` | \((\sum\lambda_i)^2/\sum\lambda_i^2\) of activation covariance | effective dimension of representations |
| `specialization_index` | permutation-matched hidden-unit overlap (matched minus unmatched) | committee-machine specialization for any matched pair |
| `subspace_overlap` | cos(principal angles) between K-dim relevant subspaces (works if \(K_s \ne K_t\)) | multi-index model recovery (Ben Arous/Gerace/Krzakala/Zdeborová-style) |
| `vector_overlap` | plain cosine similarity of two vectors/matrices (flattened) | cluster-axis recovery (mixture classification), LoRA adapter recovery |
| `weight_movement` | \(\lVert\theta_f-\theta_0\rVert/\lVert\theta_0\rVert\), always recorded by `run_order_parameters` | lazy (NTK/kernel) vs. rich (feature-learning) regime diagnostic |

### Replica-resolved sweeps

`run_order_parameters` trains `n_replicas` independent students per alpha
(sharing the training set by default, i.e. same disorder / different
dynamics) and records all of the above automatically:

```python
res = exp.run_order_parameters(alphas=[0.5, 1, 2, 4, 8], n_replicas=4)
res.mean("m_hat"), res.mean("q_ab_mean"), res.mean("chi_m"), res.mean("binder_m")

from statphys.vis import plot_order_parameter_dashboard
plot_order_parameter_dashboard(res)   # 4-panel physics dashboard incl. eps_g
```

`weight_movement` (relative distance the weights travel from
initialization, §4 of [order_parameters.md](order_parameters.md)) is
always recorded; `init_scale` multiplies the initial weights to probe
the lazy (large `init_scale`) vs. rich (feature-learning, `init_scale`
$\approx 1$) regime:

```python
res = exp.run_order_parameters(alphas=[4.0], n_replicas=5, init_scale=30.0)
res.mean("weight_movement")   # small -> lazy/NTK regime
```

One-liners for the whole pipeline (any preset, plot included):

```python
import statphys
statphys.quick_order_parameters("tiny_gpt", alphas=[1, 2, 4, 8, 16])
statphys.quick_phase_diagram("sparse_teacher", "sparsity", [0.5, 0.8, 0.95])
```

### Training dynamics (temporal transitions)

`run_training_dynamics` records train/test error, `m_hat`, and `q_ab` at
log-spaced *epochs* at fixed alpha — the protocol for grokking, plateaus,
and stagewise learning. `init_scale` multiplies the student's initial
weights (large values + weight decay induce delayed generalization):

```python
res = exp.run_training_dynamics(alpha=1.5, epochs=40000,
                                weight_decay=1e-2, init_scale=8.0)
res.plot(metrics=["train_error", "test_error"], logx=True, logy=True)
```

Results can be persisted and reloaded:

```python
res.save("out/run.json")
res = ExperimentResult.load("out/run.json")
```

### Ready-made studies

`statphys.experiment.studies` (also `statphys study <name>` on the
command line) bundles complete experiments that save JSON + figure:

| Study | Phenomenon |
|---|---|
| `committee` | Specialization transition (tanh committee) |
| `fss` | Finite-size scaling of L1 sparse recovery |
| `diagram` | 2D recovery phase diagram (sparsity x alpha) |
| `attention` | Attention-pair transition (no analytic theory) |
| `manifold` | Hidden-manifold data structure vs isotropic inputs |
| `gpt` | Tiny causal transformer order parameters |
| `grokking` | Delayed generalization in epoch time |
| `universality` | Gaussian universality of learning curves + breakdown |
| `double_descent` | Model-wise double descent vs student width |
| `scaling` | Data-scaling exponents eps_g ~ alpha^-b across architectures |
| `multi_index` | Subspace recovery in a K-direction multi-index model, matched/mismatched width |
| `mixture` | Gaussian-mixture classification; measured eps_g checked against the exact Bayes error |
| `lazy_rich` | Lazy (NTK/kernel) vs. rich (feature-learning) regime via init scale (Chizat & Bach 2019) |
| `lora` | LoRA-style low-rank fine-tuning adapter recovery |
| `plateau` | Specialization-plateau escape in online committee learning with exact Saad-Solla order parameters; ln(d) escape-time scaling (§7 of order_parameters.md) |
| `sft` | Fine-tuning as a two-teacher problem: catastrophic forgetting + transfer sign boundary in the (task similarity, alpha_ft) plane ([frontier.md](frontier.md) §1) |
| `rlhf` | Reward-model overoptimization: Goodhart transition KL*(alpha_r) under best-of-n optimization ([frontier.md](frontier.md) §2) |
| `weak_to_strong` | Weak supervisor -> strong student: PGR surface, when imitation becomes generalization ([frontier.md](frontier.md) §3) |
| `collapse` | Model collapse under recursive synthetic data; real-data anchoring boundary ([frontier.md](frontier.md) §4) |
| `icl` | Emergence of in-context learning vs pretraining task diversity ([frontier.md](frontier.md) §5) |
| `taxonomy` | Teacher structure x paradigm cross table: every taxonomy teacher (random / structured / trained-on-real-images) through every frontier probe ([frontier.md](frontier.md) §6) |

### Command-line interface

```bash
statphys list
statphys order-params tiny_gpt --alphas 1 2 4 8 --replicas 4
statphys phase-diagram sparse_teacher sparsity 0.5 0.8 0.95
statphys study universality --output-dir results
```

### 2D phase diagrams

`run_phase_diagram` sweeps (control parameter x alpha) and returns grids of
every order parameter, with heatmap plotting and contour-based boundary
estimation:

```python
from statphys.experiment import run_phase_diagram

def factory(sparsity):
    teacher = Teacher(nn.Linear(d, 1, bias=False), init="sparse",
                      init_kwargs={"sparsity": sparsity}, noise_std=0.05)
    return TeacherStudentExperiment(teacher, lambda: nn.Linear(d, 1, bias=False), d=d)

res = run_phase_diagram(factory, param_name="sparsity",
                        param_values=[0.5, 0.8, 0.9, 0.95],
                        alphas=[0.25, 0.5, 1, 2, 4], n_replicas=3)
res.plot("m_hat", contour_level=0.5)   # numerically estimated phase boundary
```

`scripts/run_phase_study.py` bundles ready-made studies (committee
specialization, finite-size scaling of sparse recovery, 2D recovery diagram,
attention teacher) producing JSON + dashboard PNGs.

## Metrics

Built-in observables (recorded automatically):

- `test_error`: generalization error on fresh samples (MSE for regression, 0-1 for binary labels; auto-detected)
- `overlap_avg`: cosine overlap of matching weight tensors (when student and teacher share shapes)

Additional model-agnostic tools in `statphys.experiment.metrics`:

- `weight_overlap(student, teacher_weights)`: per-parameter overlaps
- `linear_cka(X, Y)`: centered kernel alignment between representations
- `representation_similarity(student, teacher_model, X)`: per-layer CKA — works even when the architectures differ

Custom observables attach via the `metrics` dict:

```python
exp = TeacherStudentExperiment(
    ..., metrics={"w_norm": lambda student, ds: sum(p.norm().item()**2 for p in student.parameters())},
)
```

## Presets

`statphys.experiment.presets` bundles interesting ready-made setups:

| Preset | Setting |
|---|---|
| `random_mlp` | Random-weight MLP teacher, matched student (specialization) |
| `sparse_teacher` | Sparse linear teacher (recovery transition vs α) |
| `spiked_teacher` | Rank-1 spiked teacher (BBP-style detectability) |
| `mismatched_width` | Overparameterized student, narrow teacher |
| `low_rank_attention` | Low-rank attention teacher (toy LLM-like setting) |
| `hidden_manifold` | MLP pair on hidden-manifold inputs (realistic data structure) |
| `tiny_gpt` | Minimal causal transformer teacher-student pair (LLM-style) |
| `multi_index_model` | K-direction multi-index teacher; `k_student` may differ from `k_teacher` (subspace recovery, §3 of order_parameters.md) |
| `mixture_classification` | Generative Gaussian-mixture classification with an exact Bayes error (§5) |
| `lora_finetune` | Frozen "pretrained" backbone + trainable low-rank adapter, LoRA-style fine-tuning (§6) |

```python
from statphys.experiment import get_preset
exp = get_preset("sparse_teacher", d=400, sparsity=0.95)
exp = get_preset("multi_index_model", d=200, k_teacher=3, k_student=5)
exp = get_preset("mixture_classification", d=200, mu=2.0)
exp = get_preset("lora_finetune", d=128, hidden=16, rank_true=2, rank_student=2)
```

### Custom generative datasets

For settings where the *label determines the input* rather than the
input determining the label (e.g. classification of a Gaussian
mixture), pass a fully custom dataset object instead of `input_dist`:

```python
from statphys.experiment import GaussianMixtureDataset, TeacherStudentExperiment

d = 200
dataset = GaussianMixtureDataset(d=d, mu=2.0)
teacher = dataset.oracle_teacher()   # clean(x) = sign(v . x), Bayes-consistent

exp = TeacherStudentExperiment(
    teacher=teacher,
    student_factory=lambda: nn.Linear(d, 1, bias=False),
    d=d,
    dataset=dataset,   # overrides the default TeacherStudentDataset
)
res = exp.run_order_parameters(alphas=[1, 2, 4, 8], n_replicas=4)
```

Any object exposing `.sample(n) -> (X, y)`, `.sample_inputs(n) -> X`,
and `.get_config() -> dict` works here; see §7 of
[order_parameters.md](order_parameters.md).

## Architecture zoo

`statphys.experiment.zoo` provides matched teacher-student pairs for representative architectures. All consume flat `(n, d)` inputs; sequence models fold them into `(seq_len, d/seq_len)` tokens internally, so every architecture is interchangeable:

| Name | Architecture |
|---|---|
| `linear` | Linear map (perceptron/ridge setting) |
| `mlp` | Shallow MLP (committee-machine-like) |
| `deep_mlp` | 3-hidden-layer MLP |
| `cnn` | 1D CNN over folded token sequences |
| `lstm` | LSTM over folded token sequences |
| `attention` | Single-head self-attention block |
| `tiny_gpt` | Causal transformer (embedding + positional encoding + N blocks) — minimal LLM-style model |

```python
from statphys.experiment import ARCHITECTURES, architecture_experiment

exp = architecture_experiment(
    "tiny_gpt", d=256,
    arch_kwargs={"seq_len": 8, "d_model": 32, "n_heads": 2, "n_blocks": 2},
    teacher_init="normal",
)
result = exp.run_sample_complexity(alphas=[2, 4, 8, 16], n_seeds=3)
```

## Verification CLI

`scripts/verify_architectures.py` checks every zoo architecture end-to-end (student learns as α grows) and writes JSON + PNG per architecture:

```bash
python scripts/verify_architectures.py --arch tiny_gpt        # one architecture
python scripts/verify_architectures.py --arch all --online    # everything

# One Slurm array task per architecture (see slurm.md)
python scripts/verify_architectures.py --submit-slurm \
    --partition debug --gpus 1 --setup "source .venv/bin/activate"
```

## Complete example

See [`examples/general_teacher_student.py`](../examples/general_teacher_student.py) for sparse recovery, an attention teacher, and a fully custom setup with heavy-tailed weights + correlated inputs + custom metrics.
