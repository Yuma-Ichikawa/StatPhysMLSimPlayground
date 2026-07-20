"""Command-line orchestration for the complete phase-continuation program."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from .core.schema import (
    Manifest,
    compose_manifests,
    expand_config,
    read_manifest,
    write_manifest,
)


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _run_local(args) -> None:
    from .runner import run_manifest_task

    manifest = read_manifest(args.manifest)
    start = max(0, args.start)
    stop = len(manifest.tasks) if args.stop is None else min(args.stop, len(manifest.tasks))
    if start > stop:
        raise ValueError("start must not exceed stop")
    completed = 0
    for index in range(start, stop):
        run_manifest_task(
            args.manifest,
            index,
            args.output,
            device=args.device,
            overwrite=args.overwrite,
        )
        completed += 1
    _print({"completed": completed, "start": start, "stop": stop})


def _states(manifest: Manifest, root: Path) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    retry_conditions: set[str] = set()
    for task in manifest.tasks:
        status_path = root / "runs" / task.task_id / "status.json"
        if not status_path.is_file():
            state = "missing"
        else:
            try:
                state = str(json.loads(status_path.read_text(encoding="utf-8")).get("state", "unknown"))
            except (OSError, json.JSONDecodeError):
                state = "corrupt"
        counts[state] += 1
        if state != "completed":
            retry_conditions.add(task.condition_id)
    return counts, sorted(retry_conditions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase-continuation")
    commands = parser.add_subparsers(dest="command", required=True)

    expand = commands.add_parser("expand", help="expand TOML into an immutable five-seed manifest")
    expand.add_argument("config")
    expand.add_argument("--manifest", required=True)

    compose = commands.add_parser("compose", help="compose complete manifests with one seed contract")
    compose.add_argument("manifests", nargs="+")
    compose.add_argument("--study", required=True)
    compose.add_argument("--output", required=True)

    run_task = commands.add_parser("run-task", help="run one manifest index")
    run_task.add_argument("manifest")
    run_task.add_argument("index", type=int)
    run_task.add_argument("--output", required=True)
    run_task.add_argument("--device", default="auto")
    run_task.add_argument("--overwrite", action="store_true")

    run_local = commands.add_parser("run-local", help="run a contiguous manifest shard")
    run_local.add_argument("manifest")
    run_local.add_argument("--output", required=True)
    run_local.add_argument("--device", default="auto")
    run_local.add_argument("--start", type=int, default=0)
    run_local.add_argument("--stop", type=int)
    run_local.add_argument("--overwrite", action="store_true")

    aggregate = commands.add_parser("aggregate", help="strict five-seed aggregation")
    aggregate.add_argument("manifest")
    aggregate.add_argument("--runs", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--allow-incomplete", action="store_true")

    plot = commands.add_parser("plot", help="make error-bar-only paper figures")
    plot.add_argument("aggregate")
    plot.add_argument("--output", required=True)

    paper = commands.add_parser("paper", help="generate TeX macros from aggregate")
    paper.add_argument("aggregate")
    paper.add_argument("--output", required=True)

    slurm = commands.add_parser("slurm-script", help="render a portable DGX array")
    slurm.add_argument("manifest")
    slurm.add_argument("profile")
    slurm.add_argument("--output", required=True)

    status = commands.add_parser("status", help="summarize registered run states")
    status.add_argument("manifest")
    status.add_argument("--runs", required=True)

    retry = commands.add_parser("retry-manifest", help="emit all five seeds for incomplete conditions")
    retry.add_argument("manifest")
    retry.add_argument("--runs", required=True)
    retry.add_argument("--output", required=True)

    coverage = commands.add_parser("coverage", help="fail if a proposal experiment is unimplemented")
    coverage.add_argument("registry")
    coverage.add_argument("configs", nargs="+")
    coverage.add_argument("--output")

    taxonomy = commands.add_parser("taxonomy", help="audit tiers, outcomes, and six-size bridges")
    taxonomy.add_argument("registry")
    taxonomy.add_argument("configs", nargs="+")
    taxonomy.add_argument("--output")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "expand":
        manifest = expand_config(args.config)
        path = write_manifest(args.manifest, manifest)
        _print({"manifest": str(path), "tasks": len(manifest.tasks), "conditions": manifest.n_conditions, "seeds": len(manifest.seeds)})
    elif args.command == "compose":
        manifest = compose_manifests(args.manifests, args.study)
        _print({"manifest": str(write_manifest(args.output, manifest)), "tasks": len(manifest.tasks)})
    elif args.command == "run-task":
        from .runner import run_manifest_task

        _print(run_manifest_task(args.manifest, args.index, args.output, device=args.device, overwrite=args.overwrite))
    elif args.command == "run-local":
        _run_local(args)
    elif args.command == "aggregate":
        from .aggregate import aggregate_manifest

        _print(aggregate_manifest(args.manifest, args.runs, args.output, allow_incomplete=args.allow_incomplete))
    elif args.command == "plot":
        from .plotting import plot_all

        _print({"figures": [str(path) for path in plot_all(args.aggregate, args.output)]})
    elif args.command == "paper":
        from .paper import write_paper_results

        _print({"paper_macros": str(write_paper_results(args.aggregate, args.output))})
    elif args.command == "slurm-script":
        from .slurm import write_array_script

        _print({"script": str(write_array_script(args.manifest, args.profile, args.output))})
    elif args.command == "status":
        manifest = read_manifest(args.manifest)
        counts, conditions = _states(manifest, Path(args.runs))
        _print({"tasks": len(manifest.tasks), "states": dict(counts), "incomplete_conditions": len(conditions)})
    elif args.command == "retry-manifest":
        manifest = read_manifest(args.manifest)
        _, conditions = _states(manifest, Path(args.runs))
        selected = tuple(task for task in manifest.tasks if task.condition_id in set(conditions))
        retry_manifest = Manifest(
            study=manifest.study,
            seeds=manifest.seeds,
            tasks=selected,
            config_hash=manifest.config_hash,
            metadata={**dict(manifest.metadata), "retry_of": str(args.manifest)},
        )
        _print({"manifest": str(write_manifest(args.output, retry_manifest)), "tasks": len(selected)})
    elif args.command == "coverage":
        from .analysis.coverage import validate_coverage, write_coverage_report

        report = validate_coverage(args.registry, args.configs)
        if args.output:
            write_coverage_report(report, args.output)
        _print(report)
        if not report["ok"]:
            raise SystemExit(2)
    elif args.command == "taxonomy":
        from .analysis.taxonomy import validate_taxonomy

        report = validate_taxonomy(args.registry, args.configs)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        _print(report)
        if not report["ok"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
