"""
Command-line interface: run physics experiments without writing Python.

Installed as the `statphys` console command:

    statphys list
    statphys order-params tiny_gpt --alphas 1 2 4 8 --replicas 4
    statphys phase-diagram sparse_teacher sparsity 0.5 0.8 0.95
    statphys study grokking --quick
    statphys study all --output-dir phase_results

Every command saves a JSON with the raw records and a PNG figure into
the output directory (default: ./statphys_results).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out_dir(args) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cmd_list(_args) -> int:
    from statphys.experiment import ARCHITECTURES, PRESETS
    from statphys.experiment.studies import STUDIES

    print("presets (for order-params / phase-diagram):")
    for name in sorted(PRESETS):
        print(f"  {name}")
    print("\narchitectures (zoo):")
    for name in sorted(ARCHITECTURES):
        print(f"  {name}")
    print("\nstudies (for study):")
    for name in sorted(STUDIES):
        print(f"  {name}")
    return 0


def _cmd_order_params(args) -> int:
    import statphys

    preset_kwargs = {}
    if args.d is not None:
        preset_kwargs["d"] = args.d
    res = statphys.quick_order_parameters(
        args.preset,
        alphas=args.alphas,
        n_replicas=args.replicas,
        plot=False,
        verbose=not args.silent,
        preset_kwargs=preset_kwargs,
        lr=args.lr,
        max_epochs=args.epochs,
    )
    out = _out_dir(args)
    res.save(out / f"order_params_{args.preset}.json")

    from statphys.vis import plot_order_parameter_dashboard

    extra = ("specialization",) if "specialization" in res.records else ()
    fig, _ = plot_order_parameter_dashboard(res, title=args.preset, extra_metrics=extra)
    fig.savefig(out / f"order_params_{args.preset}.png", dpi=140)
    print(f"saved -> {out}/order_params_{args.preset}.json, .png")
    return 0


def _cmd_phase_diagram(args) -> int:
    import statphys

    preset_kwargs = {}
    if args.d is not None:
        preset_kwargs["d"] = args.d
    res = statphys.quick_phase_diagram(
        args.preset,
        args.param,
        args.values,
        alphas=args.alphas,
        n_replicas=args.replicas,
        plot=False,
        verbose=not args.silent,
        preset_kwargs=preset_kwargs,
        lr=args.lr,
        max_epochs=args.epochs,
    )
    out = _out_dir(args)
    import json

    name = f"phase_diagram_{args.preset}_{args.param}"
    (out / f"{name}.json").write_text(json.dumps(res.to_dict(), indent=2))
    fig, _ = res.plot(args.metric, logx=True, contour_level=0.5 if args.metric == "m_hat" else None)
    fig.savefig(out / f"{name}.png", dpi=140)
    print(f"saved -> {out}/{name}.json, .png")
    return 0


def _cmd_study(args) -> int:
    from statphys.experiment.studies import run_study

    run_study(args.name, out_dir=args.output_dir, quick=args.quick)
    return 0


def _cmd_catalog(_args) -> int:
    from statphys.ui import catalog

    print(json.dumps({"workflows": catalog()}, indent=2))
    return 0


def _cmd_new(args) -> int:
    from statphys.ui import write_study_template

    write_study_template(args.output, args.kind)
    print(json.dumps({"created": Path(args.output).name, "kind": args.kind}, indent=2))
    return 0


def _cmd_validate(args) -> int:
    from statphys.ui import validate_study

    print(json.dumps(validate_study(args.study), indent=2, sort_keys=True))
    return 0


def _cmd_run(args) -> int:
    from statphys.ui import run_study

    artifact = run_study(args.study, args.output)
    print(json.dumps({"result": artifact.name}, indent=2))
    return 0


def _cmd_resume(args) -> int:
    from statphys.ui import run_study

    run_directory = Path(args.run_id)
    study = run_directory / "study.toml"
    if not study.is_file():
        raise ValueError("resume expects a run directory containing its immutable study.toml")
    artifact = run_study(study, run_directory)
    print(json.dumps({"result": artifact.name, "mode": "deterministic_rerun"}, indent=2))
    return 0


def _cmd_inspect(args) -> int:
    from statphys.ui import inspect_artifact

    print(json.dumps(inspect_artifact(args.artifact), indent=2, sort_keys=True))
    return 0


def _cmd_compare(args) -> int:
    from statphys.ui import compare_artifacts

    print(json.dumps(compare_artifacts(args.left, args.right), indent=2, sort_keys=True))
    return 0


def _cmd_report(args) -> int:
    from statphys.ui import render_report

    report = render_report(args.artifact, args.output)
    print(json.dumps({"report": report.name}, indent=2))
    return 0


def _cmd_doctor(_args) -> int:
    from statphys.ui import doctor

    result = doctor()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_lab(args) -> int:
    from statphys.ui import STUDY_KINDS, write_study_template

    if args.kind is None:
        print("Choose a workflow:")
        choices = list(STUDY_KINDS)
        for index, kind in enumerate(choices, start=1):
            print(f"  {index}. {kind}: {STUDY_KINDS[kind]}")
        selected = input("Workflow [1]: ").strip() or "1"
        try:
            kind = choices[int(selected) - 1]
        except (ValueError, IndexError) as error:
            raise ValueError("choose one of the displayed workflow numbers") from error
    else:
        kind = args.kind
    write_study_template(args.output, kind)
    print(
        json.dumps({"created": Path(args.output).name, "next": "statphys validate STUDY"}, indent=2)
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="sample ratios n/d")
    parser.add_argument("--replicas", type=int, default=4, help="students per grid point")
    parser.add_argument("--d", type=int, default=None, help="input dimension override")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="max training epochs")
    parser.add_argument("--output-dir", default="statphys_results")
    parser.add_argument("--silent", action="store_true", help="suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="statphys",
        description="Statistical-physics teacher-student experiments from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list available presets, architectures, and studies")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser(
        "order-params",
        help="order-parameter sweep (m_hat, q_ab, chi, Binder, eps_g) for a preset",
    )
    p.add_argument("preset", help="preset name (see `statphys list`)")
    _add_common(p)
    p.set_defaults(func=_cmd_order_params)

    p = sub.add_parser("phase-diagram", help="2D phase diagram (preset parameter x alpha)")
    p.add_argument("preset", help="preset name")
    p.add_argument("param", help="preset keyword to sweep (e.g. sparsity, snr, latent_dim)")
    p.add_argument("values", type=float, nargs="+", help="control-parameter values")
    p.add_argument("--metric", default="m_hat", help="grid to plot")
    _add_common(p)
    p.set_defaults(func=_cmd_phase_diagram)

    p = sub.add_parser("study", help="run a ready-made study (or 'all')")
    p.add_argument("name", help="study name (see `statphys list`)")
    p.add_argument("--output-dir", default="phase_results")
    p.add_argument("--quick", action="store_true", help="small smoke-test sizes")
    p.set_defaults(func=_cmd_study)

    p = sub.add_parser("catalog", help="show guided workflows and their scientific purpose")
    p.set_defaults(func=_cmd_catalog)

    p = sub.add_parser("new", help="create a validated portable study template")
    p.add_argument("--output", default="study.toml")
    p.add_argument(
        "--kind",
        choices=("order_parameters", "phase_diagram", "online", "replica", "ready_made"),
        default="order_parameters",
    )
    p.set_defaults(func=_cmd_new)

    p = sub.add_parser("validate", help="validate a portable study TOML before computation")
    p.add_argument("study")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("run", help="run a validated study and write a portable result artifact")
    p.add_argument("study")
    p.add_argument("--output")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("resume", help="deterministically rerun a study from its run directory")
    p.add_argument("run_id")
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser(
        "inspect", help="summarize an artifact without opening implementation internals"
    )
    p.add_argument("artifact")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("compare", help="compare condition-level means from two artifacts")
    p.add_argument("left")
    p.add_argument("right")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("report", help="create a self-contained visual evidence report")
    p.add_argument("artifact")
    p.add_argument("--output", default="statphys_report.html")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("doctor", help="check optional capabilities without exposing host details")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("lab", help="guided creation of a portable study TOML")
    p.add_argument("--output", default="study.toml")
    p.add_argument(
        "--kind", choices=("order_parameters", "phase_diagram", "online", "replica", "ready_made")
    )
    p.set_defaults(func=_cmd_lab)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the statphys console command."""
    args = build_parser().parse_args(argv)
    if args.command in {"list", "order-params", "phase-diagram", "study"}:
        import matplotlib

        matplotlib.use("Agg")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
