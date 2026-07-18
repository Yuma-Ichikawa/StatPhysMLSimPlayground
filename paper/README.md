# From a Solvable Attention Transition to Transformers

The manuscript is source-only: generated paper PDFs are intentionally not
tracked.  The previous multi-paradigm draft is preserved at
`archive/frontier_atlas.tex`.

## Structure

- `main.tex` — assembly and preamble
- `macros.tex` — notation only
- `sections/*.tex` — reorderable manuscript sections
- `references.bib` — primary-source bibliography
- `generated/results_fallback.tex` — non-numerical safe defaults
- `generated/results_macros.tex` — generated only from a verified aggregate
- `../assets/transformer_phase_atlas_schematic.png` — conceptual Figure 2

Figure 1 and all numerical macros are produced from checksum-valid completed
runs.  The paper deliberately renders a placeholder when registered evidence
is unavailable; do not hand-enter headline values.

## Reproduce results

From the repository root:

```bash
PYTHONPATH=src python -m statphys.atlas.cli manifest \
  --config experiments/transformer_phase_atlas/configs/00_smoke.toml

export STATPHYS_ATLAS_CONTAINER=/path/to/site-rocm-pytorch.sif
PYTHONPATH=src python -m statphys.atlas.cli submit \
  --manifest atlas_manifests/atlas_smoke.jsonl \
  --output-root results/transformer_phase_atlas/smoke \
  --cluster-config experiments/transformer_phase_atlas/cluster/mi300x.toml
```

The aggregate/plot commands emit tidy tables, paper figures, and
`results_macros.tex`.  Site-specific container paths are provided by an
environment variable and never written into experiment source.

## Build locally

```bash
cd paper
latexmk -pdf main.tex
```

The resulting `main.pdf` is a disposable build artifact and is ignored by
Git.
