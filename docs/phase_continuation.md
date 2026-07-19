# Phase-continuation research program

The implementation separates immutable schemas in **core**, numerical domains in
**domains**, statistical analysis in **analysis**, and execution in **orchestration**.
Root-level modules are compatibility facades.

## Reproducibility contract

- Every confirmatory condition has exactly five distinct outer seeds.
- Nested disorder, data, initialization, minibatch, dropout, and evaluation seeds are
  derived with SHA-256 and stored in each task specification.
- Run directories are content addressed by condition and seed.
- Metrics must be finite and include the ten common coordinates.
- Aggregation refuses incomplete five-seed conditions unless explicitly requested.
- Confirmatory plots use Student-t error bars with four degrees of freedom.
- No repository script contains a site-specific absolute path.

## Suites

- transformer_atlas.toml: M0-M8, D0-D5, A-J, finite-size and hysteresis scans.
- transformer_systems.toml: MoE, retrieval, multimodal, compression, lifecycle, discovery.
- domain_programs.toml: diffusion, RL, and multi-agent programs.
- cross_domain.toml: four matched-latent cross-domain experiments.

Run the coverage command before submission. It exits nonzero when any proposal requirement
lacks a runner, metric, or five-seed configuration.
