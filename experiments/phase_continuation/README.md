# Phase continuation experiments

All paths are supplied at runtime. Site-specific repository, output, Python, partition,
account, and module settings belong in the works directory or environment variables.

    phase-continuation coverage coverage.toml \
      configs/transformer_atlas.toml configs/transformer_systems.toml \
      configs/domain_programs.toml configs/cross_domain.toml

    phase-continuation expand configs/transformer_atlas.toml --manifest atlas.json
    phase-continuation slurm-script atlas.json cluster/dgx_cpu.toml --output atlas.sbatch

The GPU profile runs one task per GPU with eight-way concurrency. The CPU profile bundles
four compact numerical anchors per array element. Both profiles are generic templates;
fill site-specific Slurm fields outside the reusable repository.

## Taxonomic completion contract

'taxonomy.toml' separates four realism tiers, the six solvability coordinates, all 15
coordinate-pair interactions, and seven possible continuation outcomes. The runnable
program consists of ten independent configurations so each domain can be retried without
changing the estimand. All confirmatory grids use the canonical five seeds and at least six
sizes.

Tier C entries are protocols, not simulated claims. Site-specific environment archives,
paths, manifests, job IDs, and logs belong outside this repository. A portable Spark profile
is provided at 'cluster/dgx_spark.toml'; deployments supply Python and setup through their
own profile.
