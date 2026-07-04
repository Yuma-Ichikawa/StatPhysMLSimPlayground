# Running on Slurm Clusters

`statphys.utils.slurm` generates and submits sbatch scripts programmatically. Nothing is hardcoded: partitions, accounts, module loads, and virtual-environment activation are all configuration.

## Configuration

```python
from statphys.utils.slurm import SlurmConfig

cfg = SlurmConfig(
    job_name="sweep",
    partition="debug",          # your cluster's partition
    gpus=1,                     # --gres=gpu:N (0 = CPU only)
    cpus_per_task=4,
    memory="16G",
    time_limit="01:00:00",
    setup_lines=[               # run before the payload, e.g. venv/modules
        "source .venv/bin/activate",
    ],
    extra_directives=["--account=myproj"],   # any additional #SBATCH lines
)
```

`render_sbatch(command, cfg)` returns the full script text if you want to inspect or save it without submitting.

## Single jobs

```python
from statphys.utils.slurm import SlurmLauncher

launcher = SlurmLauncher(log_dir="slurm_logs")     # stdout/err per job
job_id = launcher.submit("python my_experiment.py --alpha 2.0", cfg)

launcher.wait([job_id], poll_sec=30)               # block until finished
print(SlurmLauncher.state(job_id))                 # squeue/sacct-based state
```

## Job arrays

One task per command, with an optional concurrency cap:

```python
from statphys.utils.slurm import submit_array

commands = [f"python my_experiment.py --alpha {a}" for a in (0.5, 1, 2, 4, 8)]
job_id = submit_array(commands, cfg, launcher=launcher, max_parallel=2)
launcher.wait([job_id])
```

Internally this writes a command list file and dispatches `#SBATCH --array=0-N%max_parallel`, so each α value runs as its own task with separate logs.

## Architecture verification on Slurm

The verification CLI submits one array task per zoo architecture:

```bash
python scripts/verify_architectures.py --submit-slurm \
    --partition debug --gpus 1 \
    --setup "source .venv/bin/activate" \
    --output-dir verification_results
```

Each task writes `verification_results/<arch>.json` (metrics, pass/fail) and `<arch>.png` (learning curves). A run passes when test error decreases sufficiently as α grows — i.e., the student demonstrably learns from the teacher.

## Tips

- `setup_lines` is the right place for `module load cuda`, `conda activate`, or `source .venv/bin/activate`.
- Use `render_sbatch` to debug the generated script before submitting.
- `launcher.submit(..., dry_run=True)` writes the script without calling `sbatch` and returns its path — handy for CI or local testing.
