# Portable phase-tensor workflow

All code, experiment configurations, dependency pins, manifest expansion, array
execution, aggregation, figure generation, and TeX macro generation are tracked
in git. Site-specific paths, partitions, accounts, and node lists belong in a
thin adapter outside the repository.

Required environment variables:

```bash
export REPO_ROOT=/path/to/StatPhysMLSimPlayground
export MANIFEST_ROOT=/path/to/artifacts/manifests
export STATPHYS_OUTPUT=/path/to/artifacts/results
export STATPHYS_DATA_ROOT=/path/to/corpora
export PAPER_DIR="$REPO_ROOT/paper"
```

Create the immutable manifests without importing numerical dependencies:

```bash
scripts/phase_tensor/expand-manifests.sh
```

`run-array.sh` creates the fixed dependency environment from
`requirements/phase-tensor.txt` on the compute node when it is absent. Submit it
through `submit.sh` with a site-specific `ARRAY_SCRIPT`; the adapter must define
the allowed Slurm partition and hardware policy. After all five-seed tasks finish,
run `render-paper.sh` with the compute-node Python interpreter to create the strict
aggregate, paper figures, result macros, and `paper/main.pdf`.

Prepare public corpora before submitting the natural-data family:

```bash
scripts/phase_tensor/prepare-data.sh
```

Each corpus is fixed to a Hugging Face commit SHA in source and its bounded byte
prefix is recorded with a SHA-256 sidecar. The raw public corpus cache and large
run artifacts are deliberately not committed to git; all code, dependency pins,
source revisions, manifests, and artifact hashes needed to recreate or audit them
are committed.
