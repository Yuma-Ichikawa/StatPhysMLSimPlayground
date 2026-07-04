"""
CLI wrapper for the ready-made statistical-physics studies.

The studies themselves live in `statphys.experiment.studies` (importable
from user code and from the `statphys study` console command):

    committee, fss, diagram, attention, manifold, gpt,
    grokking, universality, double_descent, scaling

Usage:
    python scripts/run_phase_study.py --study all --output-dir phase_results
    python scripts/run_phase_study.py --study grokking --quick

Outputs one JSON (raw records) and one PNG (figure) per study.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from statphys.experiment.studies import STUDIES, run_study


def main() -> None:
    """Run the selected phase-transition studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="all", help=f"{sorted(STUDIES)} or 'all'")
    parser.add_argument("--output-dir", default="phase_results")
    parser.add_argument("--quick", action="store_true", help="small smoke-test sizes")
    args = parser.parse_args()
    run_study(args.study, out_dir=args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
