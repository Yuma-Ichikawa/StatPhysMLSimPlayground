"""Conservative evidence grades; no single visual crossing can earn Grade A."""

from __future__ import annotations

from typing import Any, Mapping


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
    if passed == len(checks):
        grade = "A"
    elif passed >= 5:
        grade = "B"
    elif passed >= 3:
        grade = "C"
    else:
        grade = "insufficient"
    return {"grade": grade, "checks": checks, "passed": passed}
