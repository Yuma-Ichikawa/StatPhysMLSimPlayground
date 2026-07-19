from pathlib import Path

from statphys.continuation.analysis.coverage import validate_coverage


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "phase_continuation"


def test_every_proposal_requirement_is_implemented_and_configured():
    report = validate_coverage(
        EXPERIMENT / "coverage.toml",
        [
            EXPERIMENT / "configs" / "transformer_atlas.toml",
            EXPERIMENT / "configs" / "transformer_systems.toml",
            EXPERIMENT / "configs" / "domain_programs.toml",
            EXPERIMENT / "configs" / "cross_domain.toml",
            EXPERIMENT / "configs" / "learned_transformer.toml",
            EXPERIMENT / "configs" / "learned_diffusion.toml",
            EXPERIMENT / "configs" / "learned_reinforcement.toml",
            EXPERIMENT / "configs" / "learned_multiagent.toml",
            EXPERIMENT / "configs" / "continuation_diagnostics.toml",
        ],
    )
    assert report["ok"], report
    assert report["required"] == report["registered"]
