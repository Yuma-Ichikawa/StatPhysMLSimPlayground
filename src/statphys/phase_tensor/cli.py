"""CLI for data preparation, strict aggregation, and paper figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .data import prepare_corpus
from .plotting import plot_phase_tensor
from .paper import write_phase_tensor_results
from .report import aggregate_phase_tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase-tensor")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("names", nargs="+")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--max-bytes", type=int, default=64_000_000)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("manifest")
    aggregate.add_argument("--runs", required=True)
    aggregate.add_argument("--output", required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("aggregate")
    plot.add_argument("--output", required=True)
    plot.add_argument("--taxonomy", help="optional nine-axis coverage TOML")
    paper = commands.add_parser("paper")
    paper.add_argument("aggregate")
    paper.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        paths = [prepare_corpus(name, Path(args.root), max_bytes=args.max_bytes) for name in args.names]
        print(json.dumps({"corpora": [str(path) for path in paths]}, indent=2), flush=True)
        if os.environ.get("STATPHYS_DATA_FAST_EXIT") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    elif args.command == "aggregate":
        output = aggregate_phase_tensor(args.manifest, args.runs, args.output)
        print(json.dumps({"aggregate": str(output)}, indent=2))
    elif args.command == "plot":
        paths = plot_phase_tensor(args.aggregate, args.output, taxonomy_path=args.taxonomy)
        print(json.dumps({"figures": [str(path) for path in paths]}, indent=2))
    elif args.command == "paper":
        output = write_phase_tensor_results(args.aggregate, args.output)
        print(json.dumps({"paper_macros": str(output)}, indent=2))


if __name__ == "__main__":
    main()
