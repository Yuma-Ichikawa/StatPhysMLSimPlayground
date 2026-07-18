"""Command line interface for registered Transformer-atlas experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _manifest(args: argparse.Namespace) -> int:
    from .sweep import load_sweep, write_manifest

    specs, metadata = load_sweep(args.config)
    destination = Path(args.output or f"atlas_manifests/{metadata['name']}.jsonl")
    write_manifest(destination, specs, metadata)
    print(json.dumps({"manifest": str(destination), "runs": len(specs), **metadata}, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    from .runner import run_experiment
    from .sweep import read_manifest

    specs, _ = read_manifest(args.manifest)
    if args.index < 0 or args.index >= len(specs):
        raise IndexError(f"run index {args.index} outside [0, {len(specs) - 1}]")
    repo = Path(args.repo).resolve() if args.repo else Path.cwd().resolve()
    summary = run_experiment(
        specs[args.index],
        args.output_root,
        device=args.device,
        repo=repo,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _submit(args: argparse.Namespace) -> int:
    from .cluster import load_cluster_profile, submit_manifest
    from .sweep import read_manifest

    specs, metadata = read_manifest(args.manifest)
    profile = load_cluster_profile(args.cluster_config, args.profile)
    job_id = submit_manifest(
        args.manifest,
        args.output_root,
        profile,
        n_runs=len(specs),
        workdir=args.workdir,
        job_name=args.job_name or str(metadata.get("name", "atlas"))[:48],
        dry_run=args.dry_run,
    )
    print(job_id)
    return 0


def _status(args: argparse.Namespace) -> int:
    from collections import Counter

    from .artifacts import RunArtifactStore
    from .sweep import read_manifest

    specs, _ = read_manifest(args.manifest)
    store = RunArtifactStore(args.output_root)
    counts: Counter[str] = Counter()
    missing: list[str] = []
    for spec in specs:
        status = store.run_dir(spec) / "status.json"
        if status.exists():
            counts[json.loads(status.read_text()).get("state", "unknown")] += 1
        else:
            counts["missing"] += 1
            missing.append(spec.run_id)
    report = {"total": len(specs), "states": dict(sorted(counts.items())), "missing": missing}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if counts.get("failed", 0) else 0


def _aggregate(args: argparse.Namespace) -> int:
    from .aggregate import aggregate_artifacts

    report = aggregate_artifacts(
        args.output_root,
        destination=args.destination,
        manifest=args.manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .plotting import generate_paper_figures

    report = generate_paper_figures(args.aggregate, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statphys-atlas",
        description="Registered positional--semantic Transformer phase-atlas experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("manifest", help="expand TOML into an immutable JSONL manifest")
    command.add_argument("--config", required=True)
    command.add_argument("--output")
    command.set_defaults(func=_manifest)

    command = sub.add_parser("run", help="execute one manifest row")
    command.add_argument("--manifest", required=True)
    command.add_argument("--index", required=True, type=int)
    command.add_argument("--output-root", required=True)
    command.add_argument("--device", default="auto")
    command.add_argument("--repo")
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(func=_run)

    command = sub.add_parser("submit", help="submit a manifest as a Slurm array")
    command.add_argument("--manifest", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--cluster-config", required=True)
    command.add_argument("--profile", default="cluster")
    command.add_argument("--workdir", default=".")
    command.add_argument("--job-name")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(func=_submit)

    command = sub.add_parser("status", help="summarize registered run states")
    command.add_argument("--manifest", required=True)
    command.add_argument("--output-root", required=True)
    command.set_defaults(func=_status)

    command = sub.add_parser("aggregate", help="create a tidy registered result table")
    command.add_argument("--output-root", required=True)
    command.add_argument("--destination", required=True)
    command.add_argument("--manifest")
    command.set_defaults(func=_aggregate)

    command = sub.add_parser("plot", help="render all registered paper figures")
    command.add_argument("--aggregate", required=True)
    command.add_argument("--output-dir", required=True)
    command.set_defaults(func=_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, IndexError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

