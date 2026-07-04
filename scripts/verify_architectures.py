"""
Teacher-student verification across the architecture zoo.

For each architecture (linear, mlp, deep_mlp, cnn, lstm, attention,
tiny_gpt) this script runs a matched teacher-student experiment:
a sample-complexity sweep and/or online SGD dynamics, checks that the
student actually learns (test error decreases with data), and writes
JSON results + PNG figures.

Designed to run standalone or as a Slurm (array) job:

    # Single architecture
    python scripts/verify_architectures.py --arch tiny_gpt

    # All architectures sequentially
    python scripts/verify_architectures.py --arch all

    # Submit one Slurm array task per architecture
    python scripts/verify_architectures.py --submit-slurm --partition debug --gpus 1

Outputs land in --output-dir (default: verification_results/), all paths
are relative to the working directory.

"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from statphys.experiment.zoo import ARCHITECTURES, architecture_experiment
from statphys.utils.seed import fix_seed

# Per-architecture experiment settings sized for a quick but meaningful check
DEFAULT_D = 256
DEFAULT_SEQ_LEN = 8
ARCH_SETTINGS: dict[str, dict] = {
    "linear": {"arch_kwargs": {}, "alphas": [0.5, 1.0, 2.0, 4.0, 8.0]},
    "mlp": {"arch_kwargs": {"hidden": 16}, "alphas": [1.0, 2.0, 4.0, 8.0, 16.0]},
    "deep_mlp": {"arch_kwargs": {"hidden": 32, "depth": 3}, "alphas": [2.0, 4.0, 8.0, 16.0]},
    "cnn": {"arch_kwargs": {"seq_len": DEFAULT_SEQ_LEN, "channels": 16}, "alphas": [2.0, 4.0, 8.0, 16.0]},
    "lstm": {"arch_kwargs": {"seq_len": DEFAULT_SEQ_LEN, "hidden": 16}, "alphas": [2.0, 4.0, 8.0, 16.0]},
    "attention": {"arch_kwargs": {"seq_len": DEFAULT_SEQ_LEN, "d_model": 16}, "alphas": [2.0, 4.0, 8.0, 16.0]},
    "tiny_gpt": {
        "arch_kwargs": {"seq_len": DEFAULT_SEQ_LEN, "d_model": 32, "n_heads": 2, "n_blocks": 2},
        "alphas": [2.0, 4.0, 8.0, 16.0],
    },
}


def verify_one(
    arch: str,
    d: int,
    n_seeds: int,
    max_epochs: int,
    device: str,
    output_dir: Path,
    run_online: bool,
) -> dict:
    """Run sample-complexity (+ optional online) verification for one arch."""
    settings = ARCH_SETTINGS[arch]
    fix_seed(0)
    t0 = time.time()

    exp = architecture_experiment(
        arch,
        d=d,
        teacher_init="normal",
        arch_kwargs=settings["arch_kwargs"],
        device=device,
    )

    result = exp.run_sample_complexity(
        alphas=settings["alphas"],
        n_seeds=n_seeds,
        max_epochs=max_epochs,
        lr=1e-2,
        n_test=1024,
        verbose=True,
    )
    errors = result.mean("test_error")
    improved = bool(errors[-1] < errors[0])
    monotone_frac = float(np.mean(np.diff(errors) <= 1e-6))

    summary = {
        "arch": arch,
        "d": d,
        "device": device,
        "alphas": settings["alphas"],
        "test_error_mean": errors.tolist(),
        "test_error_std": result.std("test_error").tolist(),
        "learns": improved,
        "monotone_fraction": monotone_frac,
        "wall_time_sec": round(time.time() - t0, 1),
    }

    fig, _ = result.plot(logy=True)
    fig.suptitle(f"{arch}: sample complexity (d={d})")
    fig.savefig(output_dir / f"{arch}_sample_complexity.png", dpi=120)

    if run_online:
        online = exp.run_online(t_max=10.0, t_steps=15, n_seeds=max(1, n_seeds - 1),
                                lr=0.05, n_test=1024, verbose=False)
        online_err = online.mean("test_error")
        summary["online_test_error_first_last"] = [float(online_err[0]), float(online_err[-1])]
        summary["online_learns"] = bool(online_err[-1] < online_err[0])
        fig2, _ = online.plot()
        fig2.suptitle(f"{arch}: online SGD dynamics")
        fig2.savefig(output_dir / f"{arch}_online.png", dpi=120)

    with open(output_dir / f"{arch}_result.json", "w") as f:
        json.dump(summary, f, indent=2)

    status = "OK" if improved else "NO-LEARNING"
    print(f"[{arch}] {status}: E_test {errors[0]:.4f} -> {errors[-1]:.4f} "
          f"({summary['wall_time_sec']}s)")
    return summary


def submit_slurm(args: argparse.Namespace) -> None:
    """Submit one Slurm array task per architecture."""
    from statphys.utils.slurm import SlurmConfig, SlurmLauncher, submit_array

    archs = sorted(ARCHITECTURES)
    commands = [
        f"python {Path(__file__).name if args.relative_script else 'scripts/verify_architectures.py'}"
        f" --arch {a} --d {args.d} --n-seeds {args.n_seeds}"
        f" --max-epochs {args.max_epochs} --device {args.device}"
        f" --output-dir {args.output_dir}"
        + (" --online" if args.online else "")
        for a in archs
    ]
    config = SlurmConfig(
        job_name="ts-verify",
        partition=args.partition,
        cpus_per_task=args.cpus,
        gpus=args.gpus,
        time_limit=args.time_limit,
        setup_lines=list(args.setup) if args.setup else [],
    )
    launcher = SlurmLauncher(script_dir="slurm_scripts", log_dir="slurm_logs")
    job_id = submit_array(commands, config, launcher=launcher,
                          max_parallel=args.max_parallel, dry_run=args.dry_run)
    print(f"submitted array job: {job_id}")
    print("task -> arch mapping:")
    for i, a in enumerate(archs):
        print(f"  {i}: {a}")


def main() -> int:
    """Parse CLI arguments and run (or submit) the verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="all",
                        help=f"Architecture ({', '.join(sorted(ARCHITECTURES))}, or 'all')")
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="verification_results")
    parser.add_argument("--online", action="store_true",
                        help="Also run online SGD dynamics")
    # Slurm submission options
    parser.add_argument("--submit-slurm", action="store_true",
                        help="Submit as a Slurm job array instead of running locally")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--setup", nargs="*", default=["source .venv/bin/activate"],
                        help="Setup lines run before the job command")
    parser.add_argument("--relative-script", action="store_true",
                        help="Reference this script by bare filename in Slurm commands")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.submit_slurm:
        submit_slurm(args)
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archs = sorted(ARCHITECTURES) if args.arch == "all" else [args.arch]
    summaries = []
    for arch in archs:
        summaries.append(
            verify_one(arch, args.d, args.n_seeds, args.max_epochs,
                       args.device, output_dir, args.online)
        )

    all_ok = all(s["learns"] for s in summaries)
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"all_learn": all_ok, "results": summaries}, f, indent=2)

    print("\n=== verification summary ===")
    for s in summaries:
        print(f"  {s['arch']:<10} learns={s['learns']} "
              f"E: {s['test_error_mean'][0]:.4f} -> {s['test_error_mean'][-1]:.4f}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
