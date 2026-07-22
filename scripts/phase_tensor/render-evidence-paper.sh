#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${SCREEN_AGGREGATE:?set SCREEN_AGGREGATE to the completed taxonomy-screen aggregate}"
: "${CONFIRM_MANIFEST:?set CONFIRM_MANIFEST to the composed confirmation manifest}"
: "${CONFIRM_OUTPUT:?set CONFIRM_OUTPUT to the confirmation run-artifact root}"
: "${PAPER_DIR:?set PAPER_DIR to the manuscript directory}"

PYTHON="${PYTHON:-python3}"
TAXONOMY_PATH="${TAXONOMY_PATH:-$REPO_ROOT/experiments/phase_continuation/phase_tensor_taxonomy.toml}"
CONFIRM_AGGREGATE="${CONFIRM_AGGREGATE:-$CONFIRM_OUTPUT/aggregate.json}"
RENDER_BASE="${RENDER_ROOT:-$CONFIRM_OUTPUT/rendered}"
mkdir -p "$RENDER_BASE"
RENDER_ROOT=$(mktemp -d "$RENDER_BASE/run.XXXXXX")
SCREEN_FIGURES="$RENDER_ROOT/screen"
CONFIRM_FIGURES="$RENDER_ROOT/confirmation"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
mkdir -p "$SCREEN_FIGURES" "$CONFIRM_FIGURES" "$PAPER_DIR/figures" "$PAPER_DIR/generated"

"$PYTHON" -m statphys.phase_tensor.cli aggregate "$CONFIRM_MANIFEST" \
  --runs "$CONFIRM_OUTPUT" --output "$CONFIRM_AGGREGATE"
"$PYTHON" -m statphys.phase_tensor.cli plot "$SCREEN_AGGREGATE" \
  --output "$SCREEN_FIGURES" --taxonomy "$TAXONOMY_PATH"
"$PYTHON" -m statphys.phase_tensor.cli plot "$CONFIRM_AGGREGATE" \
  --output "$CONFIRM_FIGURES" --taxonomy "$TAXONOMY_PATH"

screen_figures=(
  figure09_mlp_phase_splitting.pdf
  figure11_optimizer_geometry.pdf
  figure13_objective_homotopy.pdf
  figure14_scaling_paths.pdf
  figure15_natural_data_bridge.pdf
  figure17_compute_error_landscape.pdf
  figure18_theory_experiment_coverage.pdf
  figure19_training_generalization_dynamics.pdf
)
confirmation_figures=(
  figure10_mlp_causal_contribution.pdf
  figure12_optimizer_gradient_heterogeneity.pdf
  figure16_natural_data_generalization.pdf
  figure20_mlp_mechanism_atlas.pdf
)
for name in "${screen_figures[@]}"; do
  source="$SCREEN_FIGURES/$name"
  test -s "$source"
  temporary="$PAPER_DIR/figures/.${name}.tmp.$$"
  install -m 0644 "$source" "$temporary"
  mv "$temporary" "$PAPER_DIR/figures/$name"
done
for name in "${confirmation_figures[@]}"; do
  source="$CONFIRM_FIGURES/$name"
  test -s "$source"
  temporary="$PAPER_DIR/figures/.${name}.tmp.$$"
  install -m 0644 "$source" "$temporary"
  mv "$temporary" "$PAPER_DIR/figures/$name"
done

screen_macros="$PAPER_DIR/generated/.phase_tensor_results.tex.tmp.$$"
confirm_macros="$PAPER_DIR/generated/.phase_tensor_confirmation_results.tex.tmp.$$"
"$PYTHON" -m statphys.phase_tensor.cli paper "$SCREEN_AGGREGATE" \
  --output "$screen_macros"
"$PYTHON" -m statphys.phase_tensor.cli paper "$CONFIRM_AGGREGATE" \
  --output "$confirm_macros"
test -s "$screen_macros"
test -s "$confirm_macros"
mv "$screen_macros" "$PAPER_DIR/generated/phase_tensor_results.tex"
mv "$confirm_macros" "$PAPER_DIR/generated/phase_tensor_confirmation_results.tex"

for name in "${screen_figures[@]}" "${confirmation_figures[@]}"; do
  test -s "$PAPER_DIR/figures/$name"
done
