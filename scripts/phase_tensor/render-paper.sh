#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
: "${STATPHYS_MANIFEST:?set STATPHYS_MANIFEST to the composed manifest}"
: "${STATPHYS_OUTPUT:?set STATPHYS_OUTPUT to the run-artifact root}"
: "${PAPER_DIR:?set PAPER_DIR to the manuscript directory}"

bash "$REPO_ROOT/scripts/phase_tensor/postprocess.sh"

REFERENCE_AGGREGATE="${REFERENCE_AGGREGATE:-$REPO_ROOT/evidence/tensor_reference_validation/aggregate.json}"
"${PYTHON:-python3}" "$REPO_ROOT/scripts/generate_paper_figures.py" \
  --reference "$REFERENCE_AGGREGATE" \
  --confirmation-macros "$PAPER_DIR/generated/phase_tensor_confirmation_results.tex" \
  --output "$PAPER_DIR/figures" \
  --macros "$PAPER_DIR/generated/reference_results.tex"

required_figures=(
  figure1_atlas_taxonomy.pdf
  figure2_outcome_taxonomy.pdf
  figure3_observable_map.pdf
  figure4_phase_decision.pdf
  figure5_reference_response.pdf
  figure6_reference_mechanisms.pdf
  figure7_reference_verdict.pdf
  figure8_confirmation_summary.pdf
  figure9_coverage_map.pdf
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
test main.pdf -nt generated/reference_results.tex
if [ -f generated/phase_tensor_confirmation_results.tex ]; then
  test main.pdf -nt generated/phase_tensor_confirmation_results.tex
fi

if [ -n "${STATPHYS_RENDER_STATUS:-}" ]; then
  temporary_status="${STATPHYS_RENDER_STATUS}.tmp.$$"
  printf 'synchronized: aggregate, figures, TeX macros, and main.pdf\n' > "$temporary_status"
  mv "$temporary_status" "$STATPHYS_RENDER_STATUS"
fi
