"""CLI for predictive phase continuation."""

from __future__ import annotations

import argparse
import json

from .pipeline import aggregate, audit_aggregate, build_adaptive_manifest, build_manifest, plot_results, render_slurm, run_slice, write_paper_results


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="predictive-phase")
    commands = root.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest"); manifest.add_argument("--config", required=True); manifest.add_argument("--output", required=True)
    adaptive = commands.add_parser("adaptive-manifest"); adaptive.add_argument("--base-manifest", required=True); adaptive.add_argument("--aggregate", required=True); adaptive.add_argument("--config", required=True); adaptive.add_argument("--output", required=True)
    run = commands.add_parser("run"); run.add_argument("--manifest", required=True); run.add_argument("--output", required=True); run.add_argument("--start", type=int, required=True); run.add_argument("--stop", type=int, required=True); run.add_argument("--device", default="auto")
    agg = commands.add_parser("aggregate"); agg.add_argument("--manifest", required=True); agg.add_argument("--runs", required=True); agg.add_argument("--output", required=True)
    audit = commands.add_parser("audit"); audit.add_argument("--aggregate", required=True); audit.add_argument("--output", required=True)
    plot = commands.add_parser("plot"); plot.add_argument("--aggregate", required=True); plot.add_argument("--output", required=True)
    paper = commands.add_parser("paper"); paper.add_argument("--aggregate", required=True); paper.add_argument("--audit", required=True); paper.add_argument("--output", required=True)
    slurm = commands.add_parser("slurm"); slurm.add_argument("--manifest", required=True); slurm.add_argument("--profile", required=True); slurm.add_argument("--output", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "manifest": result = {"manifest": str(build_manifest(args.config).write(args.output))}
    elif args.command == "adaptive-manifest":
        payload = build_adaptive_manifest(args.base_manifest, args.aggregate, args.config)
        payload.write(args.output)
        result = {"manifest": args.output, "tasks": len(payload.tasks)}
    elif args.command == "run": result = run_slice(args.manifest, args.output, args.start, args.stop, args.device)
    elif args.command == "aggregate":
        payload = aggregate(args.manifest, args.runs, args.output)
        result = {"output": args.output, "tasks": payload["registered_tasks"], "conditions": len(payload["conditions"])}
    elif args.command == "audit":
        payload = audit_aggregate(args.aggregate, args.output)
        result = {"output": args.output, "metric_groups": len(payload["statistics"]), "saturated_groups": len(payload["saturated_metrics"])}
    elif args.command == "plot": result = {"figures": [str(path) for path in plot_results(args.aggregate, args.output)]}
    elif args.command == "paper": result = {"results": str(write_paper_results(args.aggregate, args.audit, args.output))}
    else: result = {"script": str(render_slurm(args.manifest, args.profile, args.output))}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
