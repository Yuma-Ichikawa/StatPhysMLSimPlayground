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
| callable | Any function `n -> (n, d)` tensor |

## Physics order parameters

`statphys.experiment.observables` defines statistical-physics observables in
*function space* (on a shared probe set), so they apply to any architecture,
not only to models with a single weight vector:

| Observable | Definition | Physics reading |
|---|---|---|
| `function_order_params` | \(m_f = \mathbb{E}[f_s f_t]\), \(q_f = \mathbb{E}[f_s^2]\), and \(\hat m = m_f/\sqrt{q_f \rho_f}\) | magnetization (teacher recovery); noise-independent |
| `replica_overlaps` | \(q_{ab} = \mathbb{E}[f_a f_b]\) between independently trained students | replica-symmetric overlap; \(q_{ab} \to 1\) = condensed phase, small = many distinct minima |
| `susceptibility` | \(\chi_m = d \, \mathrm{Var}[\hat m]\) over replicas | peaks at the transition, sharpens with \(d\) |
| `binder_cumulant` | \(U_4 = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2\) | curves for different \(d\) cross at \(\alpha_c\) (finite-size scaling) |
| `participation_ratio` | \((\sum\lambda_i)^2/\sum\lambda_i^2\) of activation covariance | effective dimension of representations |
| `specialization_index` | permutation-matched hidden-unit overlap (matched minus unmatched) | committee-machine specialization for any matched pair |

### Replica-resolved sweeps

`run_order_parameters` trains `n_replicas` independent students per alpha
(sharing the training set by default, i.e. same disorder / different
dynamics) and records all of the above automatically:

```python
res = exp.run_order_parameters(alphas=[0.5, 1, 2, 4, 8], n_replicas=4)
res.mean("m_hat"), res.mean("q_ab_mean"), res.mean("chi_m"), res.mean("binder_m")
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

```python
from statphys.experiment import get_preset
exp = get_preset("sparse_teacher", d=400, sparsity=0.95)
```

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
