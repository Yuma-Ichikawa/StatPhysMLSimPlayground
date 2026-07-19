"""Machine-checkable coverage of every experiment requested by the two design documents."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any, Sequence

from ..core.registry import is_supported
from ..core.schema import validate_seed_set

REQUIRED_EXPERIMENT_IDS = frozenset(
    """
T-M0-M8 T-D0-D5 T-SCALING T-POSSEM T-HEADS T-ATTN-MLP T-ICL T-LONG-CONTEXT
T-LORA T-GLASS T-OPTIMIZER T-DATA-BRIDGE T-COT T-MOE T-RETRIEVAL T-MULTIMODAL
T-COMPRESSION T-LIFECYCLE T-GENERATION T-DISCOVERY P-FSS P-HYSTERESIS P-RATE-FUNCTION
P-FDT P-NESTED-SEEDS D-ANCHOR D-GUIDANCE D-TRAJECTORY D-LOCALITY D-MEMORIZATION
R-ANCHOR R-ENTROPY-FLOW R-GOODHART R-ROLLOUT R-OPTIMIZER R-PREFERENCE
M-ANCHOR M-DEBATE M-MINORITY M-INFLUENCE M-ROLES M-SCALING
X-DIFFUSION-LANGUAGE-RL X-DIFFUSION-POLICY-RL X-MULTIAGENT-RL X-MOE-MULTIAGENT
C-COMMON-COORDINATES C-MATCHED-LATENT C-ASSUMPTION-GRAPH C-EVIDENCE-GRADES
T-LEARNED-DECODER D-LEARNED-SCORE R-LEARNED-POLICY M-LEARNED-AGENTS
C-ASSUMPTION-PAIRS C-RENORMALIZED-SURROGATE C-CRITICAL-WINDOW C-OUTCOME-ATLAS
""".split()
)


def validate_coverage(
    registry_path: str | Path,
    config_paths: Sequence[str | Path],
) -> dict[str, Any]:
    registry = tomllib.loads(Path(registry_path).read_text(encoding="utf-8"))
    entries = list(registry.get("requirements", []))
    ids = [str(entry["id"]) for entry in entries]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    missing = sorted(REQUIRED_EXPERIMENT_IDS - set(ids))
    extra = sorted(set(ids) - REQUIRED_EXPERIMENT_IDS)
    configured: set[tuple[str, str]] = set()
    configured_sizes: dict[tuple[str, str], int] = {}
    config_reports = []
    for path in config_paths:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        seeds = tuple(raw.get("study", {}).get("seeds", ()))
        validate_seed_set(seeds)
        pairs = {
            (str(experiment["domain"]), str(experiment.get("family", "anchor")))
            for experiment in raw.get("experiments", [])
        }
        configured.update(pairs)
        for experiment in raw.get("experiments", []):
            key = (str(experiment["domain"]), str(experiment.get("family", "anchor")))
            configured_sizes[key] = max(configured_sizes.get(key, 0), len(set(experiment.get("sizes", []))))
        config_reports.append({"path": str(path), "pairs": len(pairs), "seeds": len(seeds)})
    invalid = []
    unconfigured = []
    undersized = []
    for entry in entries:
        domain, family = str(entry["domain"]), str(entry["family"])
        if entry.get("status") != "implemented" or not is_supported(domain, family):
            invalid.append(entry["id"])
        aliases = {domain}
        if domain == "rl":
            aliases.add("reinforcement")
        if domain == "cross":
            aliases.add("cross_domain")
        if not any((alias, family) in configured for alias in aliases):
            unconfigured.append(entry["id"])
        if max((configured_sizes.get((alias, family), 0) for alias in aliases), default=0) < 6:
            undersized.append(entry["id"])
        if not entry.get("metrics") or not entry.get("runner"):
            invalid.append(entry["id"])
    report = {
        "ok": not any((duplicates, missing, extra, invalid, unconfigured, undersized)),
        "required": len(REQUIRED_EXPERIMENT_IDS),
        "registered": len(entries),
        "duplicates": duplicates,
        "missing": missing,
        "extra": extra,
        "invalid": sorted(set(invalid)),
        "unconfigured": sorted(set(unconfigured)),
        "undersized": sorted(set(undersized)),
        "configs": config_reports,
    }
    return report


def write_coverage_report(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
