# Phase continuation experiments

All paths are supplied at runtime. Site-specific repository, output, Python, partition,
account, and module settings belong in the works directory or environment variables.

    phase-continuation coverage coverage.toml \
      configs/transformer_atlas.toml configs/transformer_systems.toml \
      configs/domain_programs.toml configs/cross_domain.toml

    phase-continuation expand configs/transformer_atlas.toml --manifest atlas.json
    phase-continuation slurm-script atlas.json cluster/cpu.toml --output atlas.sbatch

The GPU profile runs one task per GPU with eight-way concurrency. The CPU profile bundles
four compact numerical anchors per array element. Both profiles are generic templates;
fill site-specific Slurm fields outside the reusable repository.

## Taxonomic completion contract

'taxonomy.toml' separates seven scientific layers, independent theory/realism/evidence
axes, all 15 deformation-pair interactions, and 13 continuation outcomes. The runnable
program consists of ten independent configurations so each domain can be retried without
changing the estimand. Screening grids use five registered seeds; frozen confirmation
grids use twelve untouched seeds. Every condition has at least five seeds; finite-size
evidence additionally requires a declared coordinate, path, diagnostic, and alternative.

Tier C entries are protocols, not simulated claims. Site-specific environment archives,
paths, job IDs, raw runs, and logs belong outside this repository. The immutable portable
manifest snapshot is versioned with the configs. Portable CPU and accelerator profiles
leave partition, account, Python, and setup choices to each deployment.

## Frozen confirmation suite

The broad five-seed tensor is a taxonomy screen. It is not relabeled as phase
confirmation. Four frozen configurations under 'configs/confirmation/' address
the highest-value audit gaps with twelve untouched outer seeds:

- parameter-matched MLP comparisons;
- optimizer and normalization contrasts with learning rates frozen on a disjoint
  calibration slice;
- uncapped data-scaling paths;
- byte-disjoint natural-corpus evaluation.

'evidence_tiers.toml' defines the allowed claim at each evidence level, and
'paper_claims.toml' maps every headline claim to its alternative, holdout,
falsifier, and artifact. Expand the 1,632-task confirmation manifest with
'scripts/phase_tensor/expand-confirmation-manifests.sh'. The exact paper
snapshot is stored under 'manifests/confirmation/' (136 conditions crossed
with twelve seeds); verify it with
'sha256sum -c manifests/confirmation/SHA256SUMS'.

## Dense prospective Transformer confirmation

`configs/tensor_reference_dense_confirmation.toml` is the frozen follow-up to
the public 45-run reference grid. It removes duplicate effective controls,
extends the training horizon, records denser trajectories, uses twelve fresh
outer seeds, and reserves the largest width for prospective evaluation. Its
versioned manifest contains 30 conditions and 360 tasks:

    phase-continuation expand \
      configs/tensor_reference_dense_confirmation.toml \
      --manifest manifests/confirmation/tensor_reference_dense_confirmation.json

The scheduler profile, queue policy, repository location, and artifact root are
deployment inputs and are deliberately absent from both files.
