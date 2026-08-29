# Numerical statistical-mechanics atlas paper

`main.tex` is the only manuscript entry point. Its `sections/`, `figures/`,
`generated/`, `references.bib`, and `main.pdf` are all colocated in this directory.
The repository intentionally contains no second paper tree.

Numerical values are generated only from a strict complete aggregate with exactly
five full-pipeline seeds per condition. The O(1) tensor aggregate records separate
training, IID-test, and OOD risks, their generalization gaps, and mechanism
diagnostics. A missing or failed seed aborts aggregation.

The portable phase-tensor workflow is:

```bash
phase-tensor aggregate "$STATPHYS_MANIFEST" --runs "$STATPHYS_OUTPUT" \
  --output aggregate.json
phase-tensor plot aggregate.json --output paper/figures \
  --taxonomy experiments/phase_continuation/phase_tensor_taxonomy.toml
phase-tensor paper aggregate.json --output paper/generated/phase_tensor_results.tex
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For scheduled execution, render a portable job script and provide the repository,
Python executable, manifest, and output directory through environment variables.
Before release, verify that `main.pdf` is newer than every required figure and
generated result macro.
