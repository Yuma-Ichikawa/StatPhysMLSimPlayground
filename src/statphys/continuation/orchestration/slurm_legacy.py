"""Portable one-GPU-per-task Slurm arrays for an eight-GPU DGX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import shlex
import tomllib

from ..schema import Manifest, read_manifest


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

    def __post_init__(self) -> None:
        if self.gpus_per_task < 1 or self.cpus_per_task < 1 or self.max_parallel < 1:
            raise ValueError("Slurm GPU/CPU/parallel counts must be positive")


def load_profile(path: str | Path) -> SlurmProfile:
    raw = tomllib.loads(Path(path).read_text())
    table = dict(raw.get("cluster", raw))
    table["modules"] = tuple(table.get("modules", ()))
    table["setup"] = tuple(table.get("setup", ()))
    return SlurmProfile(**table)


def render_array_script(manifest: Manifest, profile: SlurmProfile) -> str:
    directives = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=phase-continuation",
        f"#SBATCH --array=0-{len(manifest.tasks) - 1}%{profile.max_parallel}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={profile.cpus_per_task}",
        f"#SBATCH --gres=gpu:{profile.gpus_per_task}",
        f"#SBATCH --mem={profile.memory}",
        f"#SBATCH --time={profile.time_limit}",
        "#SBATCH --output=slurm-%A_%a.out",
        "#SBATCH --error=slurm-%A_%a.err",
    ]
    for flag, value in (
        ("partition", profile.partition),
        ("account", profile.account),
        ("qos", profile.qos),
        ("constraint", profile.constraint),
    ):
        if value:
            directives.append(f"#SBATCH --{flag}={value}")

    dollar = "$"
    body = [
        "",
        "set -euo pipefail",
        ': "' + dollar + '{STATPHYS_REPO:?set STATPHYS_REPO to the repository root}"',
        ': "' + dollar + '{STATPHYS_MANIFEST:?set STATPHYS_MANIFEST to the manifest JSON}"',
        ': "' + dollar + '{STATPHYS_OUTPUT:?set STATPHYS_OUTPUT to the run artifact root}"',
        f'PYTHON="{dollar}{{STATPHYS_PYTHON:-{shlex.quote(profile.python)}}}"',
    ]
    for module in profile.modules:
        body.append(f"module load {shlex.quote(module)}")
    body.extend(profile.setup)
    body.extend(
        (
            f'cd "{dollar}STATPHYS_REPO"',
            f'export PYTHONPATH="{dollar}STATPHYS_REPO/src{dollar}{{PYTHONPATH:+:{dollar}PYTHONPATH}}"',
            f'"{dollar}PYTHON" -m statphys.continuation.cli run-task '
            f'"{dollar}STATPHYS_MANIFEST" "{dollar}SLURM_ARRAY_TASK_ID" '
            f'--output "{dollar}STATPHYS_OUTPUT" --device cuda',
        )
    )
    return "\n".join(directives + body) + "\n"


def write_array_script(
    manifest: Manifest | str | Path,
    profile: SlurmProfile | str | Path,
    output_path: str | Path,
) -> Path:
    registered = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    selected = load_profile(profile) if isinstance(profile, (str, Path)) else profile
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_array_script(registered, selected))
    target.chmod(0o750)
    return target


__all__ = [
    "SlurmProfile",
    "load_profile",
    "render_array_script",
    "write_array_script",
]
