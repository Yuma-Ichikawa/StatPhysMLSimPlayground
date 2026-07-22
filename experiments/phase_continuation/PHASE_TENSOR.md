# Empirical phase-continuation tensor

This program extends the controlled continuation studies with a trainable,
byte-level causal Transformer.  It keeps data, attention, MLP, residual
parameterization, objective, optimizer, scaling path, lifecycle, and population
as separate coordinates rather than collapsing them into parameter count.

## Reproducibility contract

- Every condition has its exact registered seed set: five for taxonomy screens
  and twelve untouched seeds for the frozen confirmation suite.
- Aggregation stops if any run is absent, failed, non-finite, or duplicated.
- Error bars are two-sided 95% Student-t intervals with degrees of freedom
  determined by the registered seed count.
- Phase boundaries are interpolated independently within each seed before aggregation.
- Corpus caches record source, revision, byte count, document count, and SHA-256.
- Run directories contain immutable specifications, provenance, metrics, trajectories, and status.
- Repository commands use environment variables and contain no installation-specific paths.

## Intensive observables

Every condition stores the three intensive risks `R_train`, `R_test`, and `R_ood`,
together with `e_gen = R_test - R_train` and `e_ood = R_ood - R_train`. Cross entropy
is divided by `log(vocabulary)`, Brier risk by its uniform-reference scale,
attention entropy by `log(context length)`, and MLP participation and Jacobian ranks
by MLP width. The registered trajectory contains five-seed mean and 95% Student-t
intervals for training risk, test risk, and their gap at every checkpoint.
Parameter count, token count, FLOPs, and wall time remain extensive audit
coordinates and are never used as order parameters.

Mechanism cards additionally record MLP activation sparsity, gated-unit saturation,
local Jacobian effective rank, residual-stream RMS and drift, two-minibatch gradient
noise scale, update-to-weight scale, and a bounded corpus summary vector. These are
diagnostics: performance and mechanism claims are always stated separately.

## Configuration families

| Configuration | Coordinate varied |
|---|---|
| `tensor_mlp.toml` | no/linear/ReLU/GELU/GEGLU/SwiGLU MLP and FFN ratio |
| `tensor_optimizer.toml` | SGD-M/AdamW/Muon/SOAP/Lion/GaLore, learning rate, and normalization |
| `tensor_objective.toml` | normalized CE-Brier objective homotopy |
| `tensor_scaling.toml` | width, depth, context, and data exponent |
| `tensor_realdata.toml` | injected control, TinyStories, SimpleStories, FineWeb-Edu, and Dolma |
| `tensor_residual.toml` | no/post/pre LayerNorm and pre-RMSNorm crossed with residual scale |

## Portable execution

Install the optional dependencies:

```bash
python -m pip install -e '.[phase-tensor]'
```

Prepare bounded corpus prefixes in a user-selected location:

```bash
phase-tensor prepare-data tinystories simplestories fineweb_edu dolma \
  --root "$STATPHYS_DATA_ROOT" --max-bytes 64000000
```

Expand an immutable manifest:

```bash
phase-continuation expand \
  experiments/phase_continuation/configs/tensor_mlp.toml \
  --manifest "$STATPHYS_MANIFEST"
```

Render a scheduler script from a profile and submit it with repository, data,
manifest, and artifact roots supplied by the environment:

```bash
phase-continuation slurm-script "$STATPHYS_MANIFEST" \
  experiments/phase_continuation/cluster/dgx_spark.toml \
  --output phase-tensor-array.sh
sbatch phase-tensor-array.sh
```

After every run is complete, aggregate and render figures:

```bash
phase-tensor aggregate "$STATPHYS_MANIFEST" \
  --runs "$STATPHYS_OUTPUT" --output aggregate.json
phase-tensor plot aggregate.json --output paper/figures \
  --taxonomy experiments/phase_continuation/phase_tensor_taxonomy.toml
phase-tensor paper aggregate.json \
  --output paper/generated/phase_tensor_results.tex
```

The plotting layer fixes every PDF at 6.4 by 4.8 inches, uses a white background,
mathematical axis labels, dashed grids, and uncertainty from the exact five-seed
aggregate. `phase_tensor_taxonomy.toml` explicitly displays untested cells.
