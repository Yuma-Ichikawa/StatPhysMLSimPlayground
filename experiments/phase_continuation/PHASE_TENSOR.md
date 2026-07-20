# Empirical phase-continuation tensor

This program extends the controlled continuation studies with a trainable,
byte-level causal Transformer.  It keeps data, attention, MLP, residual
parameterization, objective, optimizer, scaling path, lifecycle, and population
as separate coordinates rather than collapsing them into parameter count.

## Reproducibility contract

- Every confirmatory condition has exactly five distinct full-pipeline seeds.
- Aggregation stops if any run is absent, failed, non-finite, or duplicated.
- Error bars are two-sided 95% Student-t intervals with four degrees of freedom.
- Phase boundaries are interpolated independently within each seed before aggregation.
- Corpus caches record source, revision, byte count, document count, and SHA-256.
- Run directories contain immutable specifications, provenance, metrics, trajectories, and status.
- Repository commands use environment variables and contain no installation-specific paths.

## Intensive observables

Cross entropy is divided by `log(vocabulary)`, Brier risk by its uniform-reference
scale, attention entropy by `log(context length)`, MLP participation ratio by MLP
width, and activation entropy by the logarithm of MLP width.  These quantities
remain dimensionless and bounded under width, depth, context, and data
continuations.  Parameter count, token count, FLOPs, and wall time are retained
as extensive audit coordinates.

## Configuration families

| Configuration | Coordinate varied |
|---|---|
| `tensor_mlp.toml` | no/linear/ReLU/GELU/GEGLU/SwiGLU MLP and FFN ratio |
| `tensor_optimizer.toml` | SGD-M/AdamW/Muon/SOAP/Lion/GaLore, learning rate, and normalization |
| `tensor_objective.toml` | normalized CE-Brier objective homotopy |
| `tensor_scaling.toml` | width, depth, context, and data exponent |
| `tensor_realdata.toml` | injected control, TinyStories, SimpleStories, FineWeb-Edu, and Dolma |

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
phase-tensor plot aggregate.json --output paper/predictive/figures
phase-tensor paper aggregate.json \
  --output paper/predictive/generated/phase_tensor_results.tex
```

The plotting layer fixes every PDF at 6.4 by 4.8 inches, uses a white background,
and draws uncertainty from the exact five-seed aggregate.
