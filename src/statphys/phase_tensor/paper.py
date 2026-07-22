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


def _risk_mean(condition: dict[str, Any]) -> float:
    metrics = condition["metrics"]
    name = "normalized_test_risk" if "normalized_test_risk" in metrics else "normalized_generalization_error"
    return float(metrics[name]["mean"])


def _select_optimizer_learning_rate(
    candidates: list[dict[str, Any]],
) -> tuple[float, int, float, str]:
    """Select LR on a deterministic slice disjoint from the reported endpoint."""
    report_size = max(int(item["size"]) for item in candidates)
    report_control = max(float(item["control"]) for item in candidates if int(item["size"]) == report_size)
    slices = sorted({(int(item["size"]), float(item["control"])) for item in candidates})
    calibration_slices = [item for item in slices if item != (report_size, report_control)]
    if not calibration_slices:
        raise RuntimeError("optimizer LR calibration requires a slice distinct from the reporting endpoint")
    calibration_size, calibration_control = calibration_slices[0]
    calibration = [
        item for item in candidates
        if int(item["size"]) == calibration_size and float(item["control"]) == calibration_control
    ]
    scores: dict[float, list[float]] = defaultdict(list)
    for item in calibration:
        scores[float(item["parameters"]["learning_rate"])].append(_risk_mean(item))
    if not scores:
        raise RuntimeError("optimizer LR calibration slice contains no risks")
    best_rate = min(scores, key=lambda rate: (statistics.fmean(scores[rate]), rate))
    status = f"calibrated (d={calibration_size}, alpha={calibration_control:g})"
    return best_rate, calibration_size, calibration_control, status


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


def _latest_endpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("endpoint selection requires at least one condition")
    size = max(int(item["size"]) for item in candidates)
    control = max(float(item["control"]) for item in candidates if int(item["size"]) == size)
    matches = [
        item for item in candidates
        if int(item["size"]) == size and float(item["control"]) == control
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one endpoint at size={size}, control={control}, found {len(matches)}"
        )
    return matches[0]


def _confirmation_macros(aggregate: dict[str, Any]) -> dict[str, str]:
    conditions = aggregate["conditions"]
    macros: dict[str, str] = {
        "PhaseConfirmRuns": f"{int(aggregate['tasks']):,}",
        "PhaseConfirmConditions": f"{len(conditions):,}",
        "PhaseConfirmBoundaries": f"{len(aggregate['boundaries']):,}",
        "PhaseConfirmSeeds": str(len(aggregate["seeds"])),
    }
    for variant, prefix in {
        "linear": "ConfirmMLPLinear",
        "gelu": "ConfirmMLPGELU",
        "geglu": "ConfirmMLPGEGLU",
        "swiglu": "ConfirmMLPSwiGLU",
    }.items():
        target = _latest_endpoint([
            item for item in conditions
            if item["family"] == "tensor_mlp" and item["variant"] == variant
        ])
        macros[f"{prefix}Order"] = _estimate(target, "semantic_order")
        for metric, suffix in (
            ("mlp_causal_effect", "CausalEffect"),
            ("mlp_risk_delta", "RiskDelta"),
            ("mlp_ablated_risk", "AblatedRisk"),
        ):
            macros[f"{prefix}{suffix}"] = _estimate(target, metric)

    for variant, prefix in {
        "adamw": "ConfirmAdamW",
        "muon": "ConfirmMuon",
        "sgd_m": "ConfirmSGDM",
        "soap": "ConfirmSOAP",
    }.items():
        candidates = [
            item for item in conditions
            if item["family"] == "tensor_optimizer"
            and item["variant"] == variant
            and item["parameters"].get("normalization") == "pre_rmsnorm"
        ]
        rates = {float(item["parameters"]["learning_rate"]) for item in candidates}
        if len(rates) != 1:
            raise RuntimeError(f"confirmation optimizer {variant} must have one frozen rate")
        target = _latest_endpoint(candidates)
        macros[f"{prefix}LearningRate"] = f"{next(iter(rates)):g}"
        macros[f"{prefix}Error"] = _estimate(target, "normalized_test_risk")
        macros[f"{prefix}Gini"] = _estimate(target, "gradient_block_gini")

    for exponent, prefix in (
        (1.0, "ConfirmLinearData"),
        (1.5, "ConfirmThreeHalfData"),
        (2.0, "ConfirmQuadraticData"),
    ):
        target = _latest_endpoint([
            item for item in conditions
            if item["family"] == "tensor_scaling"
            and float(item["parameters"].get("sample_exponent", 0.0)) == exponent
        ])
        macros[f"{prefix}ScalingError"] = _estimate(
            target, "normalized_generalization_error"
        )
        macros[f"{prefix}CapHit"] = _estimate(target, "train_cap_hit")

    for corpus, prefix in {
        "tinystories": "ConfirmTinyStories",
        "simplestories": "ConfirmSimpleStories",
        "dolma": "ConfirmDolma",
        "fineweb_edu": "ConfirmFineWebEdu",
    }.items():
        target = _latest_endpoint([
            item for item in conditions
            if item["family"] == "tensor_realdata" and item["variant"] == corpus
        ])
        macros[f"{prefix}BitsPerByte"] = _estimate(target, "bits_per_byte")
        macros[f"{prefix}GeneralizationGap"] = _estimate(
            target, "normalized_generalization_gap"
        )
        macros[f"{prefix}Disjoint"] = _estimate(target, "corpus_split_disjoint")
    return macros


def write_phase_tensor_results(aggregate_path: str | Path, output_path: str | Path) -> Path:
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    conditions = aggregate["conditions"]
    stages = {str(item.get("stage", "")) for item in conditions}
    if stages and stages <= {"heldout_confirmation", "external_confirmation"}:
        macros = _confirmation_macros(aggregate)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = ["% Generated from the frozen confirmation aggregate; do not edit by hand."]
        lines.extend(
            f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())
        )
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination
    macros: dict[str, str] = {
        "PhaseTensorRuns": f"{int(aggregate['tasks']):,}",
        "PhaseTensorConditions": f"{len(conditions):,}",
        "PhaseTensorBoundaries": f"{len(aggregate['boundaries']):,}",
    }
    for variant, prefix in {
        "none": "MLPNone",
        "gelu": "MLPGELU",
        "geglu": "MLPGEGLU",
        "swiglu": "MLPSwiGLU",
    }.items():
        condition = _condition(
            conditions,
            family="tensor_mlp",
            variant=variant,
            size=192,
            control=8.0,
            p_data_kind="synthetic_retrieval",
        )
        macros[f"{prefix}Order"] = _estimate(condition, "semantic_order")
        for metric, suffix in (
            ("mlp_causal_effect", "CausalEffect"),
            ("mlp_risk_delta", "RiskDelta"),
            ("mlp_ablated_risk", "AblatedRisk"),
        ):
            if metric in condition["metrics"]:
                macros[f"{prefix}{suffix}"] = _estimate(condition, metric)

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
        best_rate, calibration_size, calibration_control, status = _select_optimizer_learning_rate(candidates)
        target = _condition(
            candidates,
            variant=variant,
            size=max(item["size"] for item in candidates),
            control=max(item["control"] for item in candidates),
            p_learning_rate=best_rate,
        )
        macros[f"{prefix}LearningRate"] = f"{best_rate:g}"
        risk_metric = "normalized_test_risk" if "normalized_test_risk" in target["metrics"] else "normalized_generalization_error"
        macros[f"{prefix}Error"] = _estimate(target, risk_metric)
        macros[f"{prefix}Gini"] = _estimate(target, "gradient_block_gini")
        macros[f"{prefix}LearningRateSelectionStatus"] = status
        macros[f"{prefix}LearningRateCalibrationSize"] = str(calibration_size)
        macros[f"{prefix}LearningRateCalibrationAlpha"] = f"{calibration_control:g}"

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
        macros[f"{prefix}SemanticOrder"] = _estimate(target, "semantic_order")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from the strict manifest-seed aggregate; do not edit by hand."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items()))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = ["write_phase_tensor_results"]
