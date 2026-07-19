"""Portable Slurm array rendering for DGX GPU and CPU shards."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex
import tomllib

from ..core.schema import Manifest, read_manifest


@dataclass(frozen=True)
class SlurmProfile:
    partition: str = ""
    account: str = ""
    qos: str = ""
    constraint: str = ""
    gpus_per_task: int = 1
    cpus_per_task: int = 8
    memory: str = "64G"
    time_limit: str = "24:00:00"
    max_parallel: int = 8
    python: str = "python3"
    modules: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    tasks_per_array: int = 1
    signal_seconds: int = 120
    extra_sbatch: tuple[str, ...] = ()


def load_profile(path: str | Path) -> SlurmProfile:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(raw.get("slurm", raw))
    for key in ("modules", "setup", "extra_sbatch"):
        if key in payload:
            payload[key] = tuple(str(value) for value in payload[key])
    profile = SlurmProfile(**payload)
    if profile.gpus_per_task < 0 or profile.cpus_per_task < 1:
        raise ValueError("invalid GPU/CPU allocation")
    if profile.max_parallel < 1 or profile.tasks_per_array < 1:
        raise ValueError("array parallelism and shard size must be positive")
    return profile


def _directive(flag: str, value: str) -> str:
    return f"#SBATCH --{flag}={value}" if value else ""


def render_array_script(manifest: Manifest, profile: SlurmProfile) -> str:
    count = len(manifest.tasks)
    arrays = math.ceil(count / profile.tasks_per_array)
    directives = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=phase-cont",
        f"#SBATCH --array=0-{arrays - 1}%{profile.max_parallel}",
        _directive("partition", profile.partition),
        _directive("account", profile.account),
        _directive("qos", profile.qos),
        _directive("constraint", profile.constraint),
        f"#SBATCH --cpus-per-task={profile.cpus_per_task}",
        f"#SBATCH --mem={profile.memory}",
        f"#SBATCH --time={profile.time_limit}",
        f"#SBATCH --signal=B:USR1@{profile.signal_seconds}",
        "#SBATCH --output=logs/%x-%A_%a.out",
        "#SBATCH --error=logs/%x-%A_%a.err",
    ]
    if profile.gpus_per_task:
        directives.append(f"#SBATCH --gres=gpu:{profile.gpus_per_task}")
    directives.extend(f"#SBATCH {line}" for line in profile.extra_sbatch)
    body = [
        line for line in directives if line
    ] + [
        "",
        "set -euo pipefail",
        ': "${STATPHYS_MANIFEST:?export STATPHYS_MANIFEST to the generated manifest}"',
        ': "${STATPHYS_REPO:?export STATPHYS_REPO to the repository root}"',
        ': "${STATPHYS_OUTPUT:?export STATPHYS_OUTPUT to the run artifact root}"',
        f'STATPHYS_PYTHON="${{STATPHYS_PYTHON:-{shlex.quote(profile.python)}}}"',
        'cd "$STATPHYS_REPO"',
        'mkdir -p "$STATPHYS_OUTPUT" logs',
        'export PYTHONPATH="$STATPHYS_REPO/src${PYTHONPATH:+:$PYTHONPATH}"',
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
    ]
    body.extend(f"module load {shlex.quote(module)}" for module in profile.modules)
    body.extend(profile.setup)
    if profile.tasks_per_array == 1:
        body.append(
            'srun "$STATPHYS_PYTHON" -m statphys.continuation.cli run-task '
            '"$STATPHYS_MANIFEST" "$SLURM_ARRAY_TASK_ID" --output "$STATPHYS_OUTPUT" --device auto'
        )
    else:
        body.extend(
            [
                f"START=$((SLURM_ARRAY_TASK_ID * {profile.tasks_per_array}))",
                f"STOP=$((START + {profile.tasks_per_array}))",
                f'if [ "$STOP" -gt {count} ]; then STOP={count}; fi',
                'srun "$STATPHYS_PYTHON" -m statphys.continuation.cli run-local '
                '"$STATPHYS_MANIFEST" --start "$START" --stop "$STOP" '
                '--output "$STATPHYS_OUTPUT" --device auto',
            ]
        )
    return "\n".join(body) + "\n"


def write_array_script(
    manifest: Manifest | str | Path,
    profile: SlurmProfile | str | Path,
    output_path: str | Path,
) -> Path:
    loaded_manifest = read_manifest(manifest) if not isinstance(manifest, Manifest) else manifest
    loaded_profile = load_profile(profile) if not isinstance(profile, SlurmProfile) else profile
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_array_script(loaded_manifest, loaded_profile), encoding="utf-8")
    destination.chmod(destination.stat().st_mode | 0o111)
    return destination
