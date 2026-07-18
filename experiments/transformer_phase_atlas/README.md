# Transformer thermodynamics phase atlas

This directory contains only portable, declarative experiment definitions.
No user home, repository checkout, result directory, or container image is
hard-coded.  The site-specific SIF path is supplied through
`STATPHYS_ATLAS_CONTAINER`; generated manifests content-address every run.

The experiment sequence is deliberately staged:

1. `00_smoke.toml` validates the complete data → train → observable → artifact
   path.
2. `01_exact_bridge_calibration.toml` reproduces the solvable two-token model
   with its original summed loss, SGD, regularisation, and informed basins.
3. `02_architecture_*.toml` adds Q/K untieing, QKVO, heads, residual/norm, MLP,
   causality/RoPE, depth, and autoregression in separate compatible blocks.
4. `03_data_ladder.toml` holds architecture fixed while replacing D0 by D1–D5.
5. `04_finite_size_confirmatory.toml` is the preregistered finite-size sweep;
   exploratory adaptive refinements must be written to a new manifest.

Typical commands (from the repository root):

```bash
PYTHONPATH=src python -m statphys.atlas.cli manifest \
  --config experiments/transformer_phase_atlas/configs/00_smoke.toml

export STATPHYS_ATLAS_CONTAINER=/path/to/site-rocm-pytorch.sif
PYTHONPATH=src python -m statphys.atlas.cli submit \
  --manifest atlas_manifests/atlas_smoke.jsonl \
  --output-root results/transformer_phase_atlas/smoke \
  --cluster-config experiments/transformer_phase_atlas/cluster/mi300x.toml
```

The SIF path is infrastructure, not scientific configuration.  Container
identity and accelerator details are captured in each run's provenance.

