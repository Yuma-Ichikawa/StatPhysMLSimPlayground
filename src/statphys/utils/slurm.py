"""
Slurm job generation and submission utilities.

Generates sbatch scripts and submits them programmatically, without any
hardcoded cluster paths: the working directory defaults to the current
directory, the environment setup is user-supplied, and logs go to a
relative directory.

Components:
    - SlurmConfig: resource/directive configuration for one job
    - render_sbatch: turn (config, command) into an sbatch script string
    - SlurmLauncher: write + submit scripts, query state, wait for jobs
    - submit_array: sweep a list of commands as a Slurm job array

Example:
    >>> from statphys.utils.slurm import SlurmConfig, SlurmLauncher
    >>> cfg = SlurmConfig(job_name="ts-mlp", partition="debug",
    ...                   gpus=1, time_limit="01:00:00",
    ...                   setup_lines=["source .venv/bin/activate"])
    >>> launcher = SlurmLauncher(log_dir="slurm_logs")
    >>> job_id = launcher.submit("python scripts/verify_architectures.py mlp", cfg)
    >>> launcher.wait([job_id])

"""

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SlurmConfig:
    """
    Resource configuration for a Slurm job.

    Attributes:
        job_name: Slurm job name (%x in log patterns).
        partition: Partition/queue name (None = cluster default).
        nodes: Number of nodes.
        ntasks: Number of tasks.
        cpus_per_task: CPU cores per task.
        gpus: GPUs per node (rendered as --gres=gpu:N). 0 disables.
        mem: Memory request, e.g. "16G" (None = partition default).
        time_limit: Wall-clock limit "HH:MM:SS" (None = partition default).
        setup_lines: Shell lines run before the payload command
            (module loads, venv activation, environment variables).
        extra_directives: Raw additional "#SBATCH ..." lines.

    """

    job_name: str = "statphys"
    partition: str | None = None
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 4
    gpus: int = 0
    mem: str | None = None
    time_limit: str | None = None
    setup_lines: list[str] = field(default_factory=list)
    extra_directives: list[str] = field(default_factory=list)


def render_sbatch(
    command: str,
    config: SlurmConfig,
    log_dir: str = "slurm_logs",
    workdir: str | Path | None = None,
    array: str | None = None,
) -> str:
    """
    Render an sbatch script for a command.

    Args:
        command: Shell command to run (may reference $SLURM_ARRAY_TASK_ID).
        config: Resource configuration.
        log_dir: Directory for stdout/stderr logs (relative to workdir).
        workdir: Working directory of the job (default: submission cwd).
        array: Optional job-array spec, e.g. "0-6" or "0-6%2".

    Returns:
        The sbatch script contents.

    """
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={config.job_name}")
    if config.partition:
        lines.append(f"#SBATCH --partition={config.partition}")
    lines.append(f"#SBATCH --nodes={config.nodes}")
    lines.append(f"#SBATCH --ntasks={config.ntasks}")
    lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
    if config.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{config.gpus}")
    if config.mem:
        lines.append(f"#SBATCH --mem={config.mem}")
    if config.time_limit:
        lines.append(f"#SBATCH --time={config.time_limit}")
    if array:
        lines.append(f"#SBATCH --array={array}")
        lines.append(f"#SBATCH --output={log_dir}/%x_%A_%a.out")
        lines.append(f"#SBATCH --error={log_dir}/%x_%A_%a.err")
    else:
        lines.append(f"#SBATCH --output={log_dir}/%x_%j.out")
        lines.append(f"#SBATCH --error={log_dir}/%x_%j.err")
    lines.extend(config.extra_directives)

    lines.append("")
    lines.append("set -euo pipefail")
    if workdir is not None:
        lines.append(f"cd {shlex.quote(str(workdir))}")
    lines.append(f"mkdir -p {shlex.quote(log_dir)}")
    lines.extend(config.setup_lines)
    lines.append("")
    lines.append(command)
    lines.append("")
    return "\n".join(lines)


class SlurmLauncher:
    """
    Writes and submits sbatch scripts, and tracks job state.

    Args:
        script_dir: Where generated scripts are written.
        log_dir: Where job logs go (passed into the scripts).
        workdir: Working directory for jobs (default: current directory).

    """

    def __init__(
        self,
        script_dir: str | Path = "slurm_scripts",
        log_dir: str = "slurm_logs",
        workdir: str | Path | None = None,
    ):
        self.script_dir = Path(script_dir)
        self.log_dir = log_dir
        self.workdir = workdir

    def submit(
        self,
        command: str,
        config: SlurmConfig,
        array: str | None = None,
        dry_run: bool = False,
    ) -> str:
        """
        Render, write, and sbatch a job.

        Args:
            command: Payload shell command.
            config: Resource configuration.
            array: Optional job-array spec.
            dry_run: If True, write the script but do not submit;
                returns the script path instead of a job id.

        Returns:
            Slurm job id (or the script path when dry_run).

        """
        script = render_sbatch(
            command, config, log_dir=self.log_dir, workdir=self.workdir, array=array
        )
        self.script_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_dir / f"{config.job_name}.sbatch"
        path.write_text(script)
        if dry_run:
            return str(path)

        out = subprocess.run(
            ["sbatch", "--parsable", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # --parsable prints "jobid[;cluster]"
        return out.stdout.strip().split(";")[0]

    @staticmethod
    def state(job_id: str) -> str:
        """
        Return the Slurm state of a job ("RUNNING", "COMPLETED", ...).

        Falls back to sacct when the job has left the squeue.
        """
        out = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )
        st = out.stdout.strip()
        if st:
            return st.splitlines()[0]
        out = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
            capture_output=True,
            text=True,
        )
        st = out.stdout.strip()
        return st.splitlines()[0].strip() if st else "UNKNOWN"

    def wait(
        self,
        job_ids: list[str],
        poll_sec: float = 15.0,
        timeout_sec: float | None = None,
    ) -> dict[str, str]:
        """
        Block until all jobs leave the queue.

        Args:
            job_ids: Jobs to wait for.
            poll_sec: Polling interval.
            timeout_sec: Optional overall timeout.

        Returns:
            Mapping job id -> final state.

        """
        t0 = time.time()
        pending = set(job_ids)
        final: dict[str, str] = {}
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
        while pending:
            for jid in list(pending):
                st = self.state(jid)
                if any(st.startswith(t) for t in terminal):
                    final[jid] = st
                    pending.discard(jid)
            if pending:
                if timeout_sec is not None and time.time() - t0 > timeout_sec:
                    for jid in pending:
                        final[jid] = "WAIT_TIMEOUT"
                    break
                time.sleep(poll_sec)
        return final


def submit_array(
    commands: list[str],
    config: SlurmConfig,
    launcher: SlurmLauncher | None = None,
    max_parallel: int | None = None,
    dry_run: bool = False,
) -> str:
    """
    Submit a list of commands as a single Slurm job array.

    Each array task runs one command, dispatched on $SLURM_ARRAY_TASK_ID.

    Args:
        commands: One shell command per array task.
        config: Resource configuration (applies to each task).
        launcher: Launcher to use (default: fresh one in cwd).
        max_parallel: Cap on simultaneously running tasks.
        dry_run: Write the script without submitting.

    Returns:
        Job id (or script path when dry_run).

    """
    if not commands:
        raise ValueError("commands is empty")
    launcher = launcher or SlurmLauncher()

    case_lines = ["case ${SLURM_ARRAY_TASK_ID} in"]
    for i, cmd in enumerate(commands):
        case_lines.append(f"  {i}) {cmd} ;;")
    case_lines.append("  *) echo \"unknown task ${SLURM_ARRAY_TASK_ID}\" >&2; exit 1 ;;")
    case_lines.append("esac")
    payload = "\n".join(case_lines)

    spec = f"0-{len(commands) - 1}"
    if max_parallel:
        spec += f"%{max_parallel}"
    return launcher.submit(payload, config, array=spec, dry_run=dry_run)
