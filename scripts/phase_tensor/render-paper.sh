#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${STATPHYS_MANIFEST:?set STATPHYS_MANIFEST to the composed manifest}"
: "${STATPHYS_OUTPUT:?set STATPHYS_OUTPUT to the run-artifact root}"
: "${PAPER_DIR:?set PAPER_DIR to the manuscript directory}"

bash "$REPO_ROOT/scripts/phase_tensor/postprocess.sh"

required_figures=(
  figure1_protocol.pdf
  figure2_anchor_validation.pdf
  figure3_transformer.pdf
  figure4_diffusion.pdf
  figure5_reinforcement.pdf
  figure6_multiagent.pdf
  figure7_predictive_bridge.pdf
  figure8_theory_breakdown.pdf
  figure09_mlp_phase_splitting.pdf
  figure10_mlp_causal_contribution.pdf
  figure11_optimizer_geometry.pdf
  figure12_optimizer_gradient_heterogeneity.pdf
  figure13_objective_homotopy.pdf
  figure14_scaling_paths.pdf
  figure15_natural_data_bridge.pdf
  figure16_natural_data_generalization.pdf
  figure17_compute_error_landscape.pdf
  figure18_theory_experiment_coverage.pdf
  figure19_training_generalization_dynamics.pdf
  figure20_mlp_mechanism_atlas.pdf
)
for figure in "${required_figures[@]}"; do
  test -s "$PAPER_DIR/figures/$figure"
done

cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

test -s main.pdf
for figure in "${required_figures[@]}"; do
  test main.pdf -nt "figures/$figure"
done
test main.pdf -nt generated/phase_tensor_results.tex
