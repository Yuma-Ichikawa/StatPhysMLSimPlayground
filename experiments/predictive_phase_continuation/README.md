# Predictive phase continuation

This study asks one question: can observables calibrated on a solvable anchor
predict an unseen finite-size phase boundary after controlled assumptions are
removed?

The production design uses 16 independent outer-disorder seeds and four inner
stochastic replicates.  The largest architecture variant is hidden from all
fits.  A second adaptive stage may add outer seeds near the estimated boundary;
five seeds are never treated as an exact requirement.

## Primary claims

1. Boundary transport beats a base surrogate on blinded variants.
2. Non-additive assumption interactions localize theory breakdown.
3. A domain-specific macrovariable reduces held-out boundary error.
4. Critical-window allocation improves quality per unit compute.

Transformer and diffusion are the deep controlled cases.  Reinforcement and
multi-agent experiments test whether the same predictive grammar survives in
driven systems without claiming shared exponents.

## Reproduction

```bash
python -m statphys.predictive.cli manifest \
  --config experiments/predictive_phase_continuation/config.toml \
  --output manifest.json
python -m statphys.predictive.cli slurm \
  --manifest manifest.json \
  --profile experiments/predictive_phase_continuation/cluster/dgx_spark.toml \
  --output job.sbatch
```

Cluster paths are supplied only through `STATPHYS_REPO`, `STATPHYS_MANIFEST`,
`STATPHYS_OUTPUT`, and `STATPHYS_PYTHON`.  No site path is stored in the repo.

## Evidence contract

- Raw outer-seed values and inner replicates remain in every artifact.
- Intervals use outer-disorder resampling, never pooled pseudo-replicates.
- Continuous, first-order-like, and smooth-crossover width models compete on
  largest-size prediction.
- Plots show raw seed dots and 95% hierarchical bootstrap intervals.
- Every plot is 6.4 by 4.8 inches and follows the registered journal style.
