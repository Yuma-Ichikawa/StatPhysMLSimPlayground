# Guided workflow

The guided interface presents experiments as a short, reproducible sequence:
choose a workflow, create a portable study file, validate it, run it, then
inspect and report the resulting evidence. It does not require a notebook or a
local server.

## Choose and create a study

```bash
statphys catalog
statphys new --kind order_parameters --output study.toml
```

`catalog` lists the stable user-facing workflows:

- `order_parameters`: a sweep with uncertainty-aware order parameters.
- `phase_diagram`: a two-control sweep with finite-size visualisation.
- `online`: online linear learning compared with an ODE result.
- `replica`: regularized linear regression compared with a replica solver.
- `ready_made`: a built-in exploratory study.

`statphys lab` provides the same selection interactively. The generated TOML
contains only relative output names, a root seed, replica count, and an
evidence tier. It is safe to copy to another machine or share with a
collaborator. The `seed` field is recorded for provenance but is not currently
consumed by `run`/`resume`; determinism instead comes from the fixed internal
seeding of the underlying `quick_*` routines.

## Validate and preview before computing

```bash
statphys validate study.toml
statphys preview study.toml
```

Validation checks the workflow, positive controls, replica count, and evidence
tier. An exploratory or confirmatory study is labelled as a **response shift**;
only a finite-size study can use the stronger **phase transition** wording.
`preview` estimates the number of runs, the evidence tier, and the figures a
study will produce without running any computation.

## Run, inspect, and compare

```bash
statphys run study.toml --output results
statphys inspect results/result.json
statphys resume results
statphys status results
statphys retry results
statphys compare first/result.json second/result.json
```

`run` saves the immutable `study.toml` alongside `result.json`. `resume`
checks that snapshot and performs the same deterministic run. `status` shows
the portable state and attempt metadata for a run, and `retry` starts a new
execution attempt for it. `inspect` exposes artifact-level metadata without
requiring users to know the internal module layout. `compare` reports
condition-level metric means and their differences.

## Render a visual evidence report

```bash
statphys report results/result.json --output results/report.html
```

The report is a self-contained HTML file. It renders condition-level curves
with any recorded 95% uncertainty intervals, and includes an evidence panel
for the independent-seed count, theory status, and allowed scientific wording.
Reports include no source path, host name, or server address.

## Diagnose optional capabilities

```bash
statphys doctor
```

This checks whether the optional numerical and plotting packages needed for
full simulations are importable, without revealing machine-specific details.
