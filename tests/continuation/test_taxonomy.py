from pathlib import Path

from statphys.continuation.analysis.taxonomy import validate_taxonomy

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "phase_continuation"
CONFIGS = [
    EXPERIMENT / "configs" / name
    for name in (
        "transformer_atlas.toml",
        "transformer_systems.toml",
        "domain_programs.toml",
        "cross_domain.toml",
        "learned_transformer.toml",
        "learned_diffusion.toml",
        "learned_reinforcement.toml",
        "learned_multiagent.toml",
        "continuation_diagnostics.toml",
    )
]


def test_taxonomy_is_complete_without_overclaiming_tier_c() -> None:
    report = validate_taxonomy(EXPERIMENT / "taxonomy.toml", CONFIGS)
    assert report["ok"], report
    assert report["runnable_complete"]
    assert not report["full_realism_complete"]
    assert report["coordinates"] == 6
    assert report["outcomes"] == 7
    assert report["assumption_pairs"] == 15
    assert report["tier_c_protocols"] == 4
