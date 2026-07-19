from pathlib import Path

from statphys.continuation.analysis.nested import five_seed_interval
from statphys.continuation.schema import expand_config
from statphys.continuation.slurm import load_profile, render_array_script


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "phase_continuation"


def test_nested_interval_is_exactly_five_outer_seeds():
    interval = five_seed_interval([1, 2, 3, 4, 5])
    assert interval["mean"] == 3.0
    assert interval["ci95_low"] < interval["mean"] < interval["ci95_high"]


def test_generated_slurm_script_is_portable_and_uses_full_gpu_array():
    manifest = expand_config(EXPERIMENT / "configs" / "cross_domain.toml")
    profile = load_profile(EXPERIMENT / "cluster" / "dgx_gpu.toml")
    script = render_array_script(manifest, profile)
    assert "%8" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "/mnt/" not in script
    assert "STATPHYS_REPO" in script
