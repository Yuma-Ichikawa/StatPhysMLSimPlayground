# Phase Diagrams Without Solvable Models: An Empirical Teacher-Student Atlas of Modern Learning Paradigms

Paper draft (ICLR 2027 target) built entirely on this repository:
every figure is one `statphys study` command.

## Build

```bash
latexmk -pdf main.tex
```

or

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Contents

- `main.tex` — the paper (self-contained preamble; no external style file)
- `references.bib` — bibliography
- `figures/` — all figures, produced by the studies of this repository
- `main.pdf` — compiled draft

## Regenerating figures

From the repository root:

```bash
statphys study sft && statphys study rlhf && statphys study weak_to_strong \
  && statphys study collapse && statphys study icl && statphys study taxonomy
cp phase_results/{sft,rlhf,weak_to_strong,collapse,icl,taxonomy}.png paper/figures/
```
