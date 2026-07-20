"""Generate auditable TeX macros from a strict phase-tensor aggregate."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


def _estimate(condition: dict[str, Any], metric: str) -> str:
    value = condition["metrics"][metric]
    return f"${value['mean']:.3f}\\pm {value['ci95']:.3f}$"


def _condition(conditions: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = []
    for condition in conditions:
        accepted = True
        for key, value in criteria.items():
            actual = condition["parameters"].get(key[2:]) if key.startswith("p_") else condition.get(key)
            accepted = accepted and actual == value
        if accepted:
            matches.append(condition)
    if len(matches) != 1:
        raise RuntimeError(f"expected one condition for {criteria}, found {len(matches)}")
    return matches[0]


def write_phase_tensor_results(aggregate_path: str | Path, output_path: str | Path) -> Path:
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    conditions = aggregate["conditions"]
    macros: dict[str, str] = {
        "PhaseTensorRuns": f"{int(aggregate['tasks']):,}",
        "PhaseTensorConditions": f"{len(conditions):,}",
        "PhaseTensorBoundaries": f"{len(aggregate['boundaries']):,}",
    }
    for variant, macro in {
        "none": "MLPNoneOrder",
        "gelu": "MLPGELUOrder",
        "geglu": "MLPGEGLUOrder",
        "swiglu": "MLPSwiGLUOrder",
    }.items():
        condition = _condition(
            conditions,
            family="tensor_mlp",
            variant=variant,
            size=192,
            control=8.0,
            p_data_kind="synthetic_retrieval",
        )
        macros[macro] = _estimate(condition, "semantic_order")

    optimizer_conditions = [item for item in conditions if item["family"] == "tensor_optimizer"]
    for variant, prefix in {
        "adamw": "AdamW",
        "muon": "Muon",
        "sgd_m": "SGDM",
        "soap": "SOAP",
    }.items():
        candidates = [
            item
            for item in optimizer_conditions
            if item["variant"] == variant and item["parameters"].get("normalization") == "pre_rmsnorm"
        ]
        scores: dict[float, list[float]] = defaultdict(list)
        for item in candidates:
            rate = float(item["parameters"]["learning_rate"])
            scores[rate].append(item["metrics"]["normalized_generalization_error"]["mean"])
        best_rate = min(scores, key=lambda rate: statistics.fmean(scores[rate]))
        target = _condition(
            candidates,
            variant=variant,
            size=max(item["size"] for item in candidates),
            control=max(item["control"] for item in candidates),
            p_learning_rate=best_rate,
        )
        macros[f"{prefix}LearningRate"] = f"{best_rate:g}"
        macros[f"{prefix}Error"] = _estimate(target, "normalized_generalization_error")
        macros[f"{prefix}Gini"] = _estimate(target, "gradient_block_gini")

    for exponent, prefix in ((1.0, "LinearData"), (1.5, "ThreeHalfData"), (2.0, "QuadraticData")):
        target = _condition(
            conditions,
            family="tensor_scaling",
            variant="data",
            size=192,
            control=4.0,
            p_sample_exponent=exponent,
        )
        macros[f"{prefix}ScalingError"] = _estimate(target, "normalized_generalization_error")

    for corpus, prefix in {
        "tinystories": "TinyStories",
        "simplestories": "SimpleStories",
        "dolma": "Dolma",
        "fineweb_edu": "FineWebEdu",
    }.items():
        target = _condition(
            conditions,
            family="tensor_realdata",
            variant=corpus,
            size=128,
            control=4.0,
            p_activation="swiglu",
        )
        macros[f"{prefix}BitsPerByte"] = _estimate(target, "bits_per_byte")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from the strict five-seed aggregate; do not edit by hand."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items()))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = ["write_phase_tensor_results"]
