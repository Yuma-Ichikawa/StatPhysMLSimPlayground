# Unified phase-continuation paper

Numerical values are generated only from a strict complete aggregate with exactly five
outer seeds per condition. The aggregate now spans ten suites, at least six sizes per
confirmatory family, four realism tiers, and the complete assumption/outcome taxonomy.

Generate the aggregate, all eleven error-bar figure files (including the eight registered
main figures), and TeX macros before compiling:

    phase-continuation plot COMPLETE_AGGREGATE --output figures/generated
    phase-continuation paper COMPLETE_AGGREGATE --output generated/results.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Without 'generated/results.tex', the manuscript deliberately compiles through the
pending-results branch. Tier C protocols remain labeled incomplete even when every runnable
Tier A--B+ job is complete.
