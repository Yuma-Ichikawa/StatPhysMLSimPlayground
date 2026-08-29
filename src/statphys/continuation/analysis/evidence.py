"""Conservative evidence grades; no single visual crossing can earn Grade A."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from statphys.core import EvidenceEngine


def grade_transition(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "five_seeds": bool(diagnostics.get("five_seeds", False)),
        "five_sizes": int(diagnostics.get("n_sizes", 0)) >= 5,
        "susceptibility_growth": float(diagnostics.get("peak_growth_exponent", 0.0)) > 0.0,
        "binder_support": float(diagnostics.get("binder_crossing_spread", 1.0)) < 0.15,
        "correction_tested": bool(diagnostics.get("finite_size_correction_tested", False)),
        "hysteresis_tested": bool(diagnostics.get("hysteresis_tested", False)),
        "nested_uncertainty": bool(diagnostics.get("nested_uncertainty", False)),
    }
    passed = sum(checks.values())
    vector = EvidenceEngine.assess(
        {
            "observable_registered": diagnostics.get("observable_registered", True),
            "semantic_null_passed": diagnostics.get("semantic_null_passed", False),
            "outer_seeds": diagnostics.get(
                "outer_seeds", 5 if diagnostics.get("five_seeds", False) else 0
            ),
            "n_sizes": diagnostics.get("n_sizes", 0),
            "finite_size_diagnostic": checks["susceptibility_growth"] and checks["binder_support"],
            "prospective_largest_size": diagnostics.get("prospective_largest_size", False),
            "frozen_comparison": diagnostics.get("frozen_comparison", False),
            "untouched_holdout": diagnostics.get("untouched_holdout", False),
            "matched_intervention": diagnostics.get("matched_intervention", False),
            "natural_or_realistic_endpoint": diagnostics.get("natural_endpoint", False),
            "pretrained_endpoint": diagnostics.get("pretrained_endpoint", False),
        }
    )
    return {
        "grade": vector.grade,
        "evidence_vector": vector.to_dict(),
        "checks": checks,
        "passed": passed,
    }
