"""Portable Slurm-array orchestration for the phase atlas."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from statphys.utils.slurm import SlurmConfig, SlurmLauncher


@dataclass(frozen=True)
class ClusterProfile:
    """Site policy kept outside experiment definitions.

    A container location is read from ``container_environment`` at submission
    time.  Consequently neither library code nor reproducibility scripts need
    a user-specific absolute path.
    """

    partition: str | None = None
    gpus: int = 1
    cpus: int = 4
    memory: str = "16G"
    time_limit: str = "06:00:00"
    max_parallel: int = 8
    module_init: str = "/etc/profile.d/modules.sh"
    modules: tuple[str, ...] = ("singularity",)
    container_environment: str = "STATPHYS_ATLAS_CONTAINER"
    container_runtime: str = "singularity"
    container_accelerator_flag: str = "--rocm"
    container_workdir: str = "/workspace"
    python: str = "python3"
    extra_directives: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ClusterProfile":
        values = dict(raw)
        for key in ("modules", "extra_directives"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def setup_lines(self) -> list[str]:
        lines: list[str] = []
        if self.module_init:
            lines.append(f"source {shlex.quote(self.module_init)}")
        if self.modules:
            lines.append("module load " + " ".join(shlex.quote(item) for item in self.modules))
        for name, value in sorted(self.environment.items()):
            if not name.replace("_", "").isalnum() or not name[0].isalpha():
                raise ValueError(f"invalid environment variable name: {name!r}")
            lines.append(f"export {name}={shlex.quote(value)}")
        return lines

    def slurm(self, job_name: str) -> SlurmConfig:
        return SlurmConfig(
            job_name=job_name,
            partition=self.partition,
            cpus_per_task=self.cpus,
            gpus=self.gpus,
            mem=self.memory,
            time_limit=self.time_limit,
            setup_lines=self.setup_lines(),
            extra_directives=list(self.extra_directives),
        )


def load_cluster_profile(path: str | Path, profile: str = "cluster") -> ClusterProfile:
    import tomllib

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if profile not in raw:
        raise KeyError(f"profile table [{profile}] not found in {path}")
    return ClusterProfile.from_mapping(raw[profile])


def container_command(
    profile: ClusterProfile,
    *,
    manifest: str | Path,
    output_root: str | Path,
    device: str = "auto",
) -> str:
    """Render one array-task payload without embedding a host path."""

    variable = profile.container_environment
    if not variable.replace("_", "").isalnum() or not variable[0].isalpha():
        raise ValueError(f"invalid container environment variable: {variable!r}")
    manifest_arg = shlex.quote(str(manifest))
    output_arg = shlex.quote(str(output_root))
    inner = (
        f"cd {shlex.quote(profile.container_workdir)} && "
        f"PYTHONPATH={shlex.quote(profile.container_workdir + '/src')} "
        f"{shlex.quote(profile.python)} -m statphys.atlas.cli run "
        f"--manifest {manifest_arg} --index \"$SLURM_ARRAY_TASK_ID\" "
        f"--output-root {output_arg} --device {shlex.quote(device)}"
    )
    bind = f'"$PWD:{profile.container_workdir}"'
    image = f'"${{{variable}:?set {variable} to a readable SIF image}}"'
    return " ".join(
        part
        for part in (
            shlex.quote(profile.container_runtime),
            "exec",
            profile.container_accelerator_flag,
            "--bind",
            bind,
            image,
            "bash -lc",
            shlex.quote(inner),
        )
        if part
    )


def submit_manifest(
    manifest: str | Path,
    output_root: str | Path,
    profile: ClusterProfile,
    *,
    n_runs: int,
    workdir: str | Path | None = None,
    job_name: str = "atlas",
    dry_run: bool = False,
) -> str:
    """Submit a registered manifest as a bounded-concurrency Slurm array."""

    if n_runs < 1:
        raise ValueError("cannot submit an empty manifest")
    if os.environ.get(profile.container_environment) is None:
        raise RuntimeError(
            f"set {profile.container_environment} to the site container image before submission"
        )
    root = Path(workdir or Path.cwd()).resolve()
    manifest_path = Path(manifest)
    output_path = Path(output_root)
    try:
        manifest_arg = manifest_path.resolve().relative_to(root)
        output_arg = output_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest and output root must be inside the submitted workdir") from exc
    launcher = SlurmLauncher(
        script_dir=root / "slurm_scripts",
        log_dir="slurm_logs",
        workdir=root,
    )
    array = f"0-{n_runs - 1}%{max(1, profile.max_parallel)}"
    command = container_command(
        profile, manifest=manifest_arg, output_root=output_arg, device="auto"
    )
    return launcher.submit(command, profile.slurm(job_name), array=array, dry_run=dry_run)

