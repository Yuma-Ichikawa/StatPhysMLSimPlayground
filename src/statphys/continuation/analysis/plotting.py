"""Paper figures; every empirical series is drawn with a five-seed 95% error bar."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from ..aggregate import read_aggregate

DOMAIN_ORDER = ("transformer", "diffusion", "reinforcement", "multiagent")
DOMAIN_TITLES = {
    "transformer": "Transformer",
    "diffusion": "Diffusion",
    "reinforcement": "Reinforcement learning",
    "rl": "Reinforcement learning",
    "multiagent": "Multi-agent",
    "cross_domain": "Cross-domain continuation",
}
COLORS = ("#006D77", "#E76F51", "#264653", "#E9C46A", "#2A9D8F", "#D62828", "#5F6F52", "#B56576")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")
METRIC_LABELS = {
    "order_parameter": r"$m$",
    "susceptibility": r"$\chi$",
    "binder_cumulant": r"$U_4$",
    "generalization_error": r"$e_{\rm gen}$",
    "effective_multiplicity": r"$K_{\rm eff}$",
    "interaction_range": r"$\xi$",
    "oracle_gap": r"$\Delta_{\rm oracle}$",
    "intervention_response": r"$\Delta_{\rm int}$",
    "pair_interaction_delta": r"$|\Delta_{ij}g_{\rm c}|$",
    "bridge_error": r"$\mathcal E_{\rm bridge}$",
    "response_slope": r"$\partial_g m$",
    "threshold_separation": r"$\Delta g_{\rm c}$",
}


def _control_label(name: str) -> str:
    known = {
        "sample_coefficient": r"$\alpha=N_{\rm train}/d^\gamma$",
        "noise": r"$\sigma$",
        "guidance": r"$w_{\rm CFG}$",
        "verifier_noise": r"$\epsilon_{\rm verifier}$",
        "coupling": r"$J$",
    }
    if name in known:
        return known[name]
    return "$" + name.replace("_", r"\_") + "$"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
            "axes.xmargin": 0.01,
            "axes.ymargin": 0.01,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "legend.frameon": False,
        }
    )


def _parameter_key(record: dict[str, Any]) -> str:
    return json.dumps(record.get("parameters", {}), sort_keys=True, separators=(",", ":"))


def _parameter_suffix(record: dict[str, Any]) -> str:
    parameters = record.get("parameters", {})
    for key in ("pair", "transition_kind", "data_stage", "scan_direction"):
        if key in parameters:
            return f", {key}={parameters[key]}"
    return ""


def _series(
    records: list[dict[str, Any]], metric: str
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if metric in record["metrics"]:
            groups[
                (
                    record.get("family", "anchor"),
                    record["variant"],
                    int(record["size"]),
                    _parameter_key(record),
                )
            ].append(record)
    result = []
    for (family, variant, size, _), values in sorted(groups.items()):
        label = f"{family}/{variant}{_parameter_suffix(values[0])}, N={size}"
        result.append((label, size, sorted(values, key=lambda value: float(value["control"]))))
    return result


def _representative_series(
    records: list[dict[str, Any]], metric: str, limit: int = 10
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    largest: dict[tuple[str, str, str], int] = {}
    for record in records:
        if metric not in record["metrics"]:
            continue
        key = (record.get("family", "anchor"), record["variant"], _parameter_key(record))
        largest[key] = max(largest.get(key, 0), int(record["size"]))
    selected = [
        record
        for record in records
        if metric in record["metrics"]
        and int(record["size"])
        == largest[(record.get("family", "anchor"), record["variant"], _parameter_key(record))]
    ]
    series = _series(selected, metric)
    if len(series) <= limit:
        return series
    indices = np.linspace(0, len(series) - 1, limit, dtype=int)
    return [series[index] for index in indices]


def _errorbar(
    ax: Any, values: list[dict[str, Any]], metric: str, label: str, index: int
) -> None:
    x = [float(value["control"]) for value in values]
    y = [float(value["metrics"][metric]["mean"]) for value in values]
    error = [float(value["metrics"][metric]["ci95"]) for value in values]
    ax.errorbar(
        x,
        y,
        yerr=error,
        label=label,
        color=COLORS[index % len(COLORS)],
        marker=MARKERS[index % len(MARKERS)],
        markersize=3.7,
        linewidth=1.2,
        capsize=2.2,
    )
    ax.grid(ls="--", color="0.82", linewidth=0.7, alpha=0.9)
    ax.tick_params(which="both", top=True, right=True, labelsize=12)


def _save(figure: Any, output_dir: str | Path, stem: str) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / stem
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)
    return path.with_suffix(".pdf")


def plot_phase_panels(aggregate: dict[str, Any] | str | Path, output_dir: str | Path) -> Path:
    data = read_aggregate(aggregate) if isinstance(aggregate, (str, Path)) else aggregate
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), constrained_layout=True)
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        records = [value for value in data["records"] if value["domain"] == domain]
        for index, (label, _, values) in enumerate(_representative_series(records, "order_parameter", 8)):
            _errorbar(ax, values, "order_parameter", label, index)
        ax.set_title(DOMAIN_TITLES[domain], loc="left", fontweight="bold")
        ax.set_xlabel(_control_label(records[0]["control_name"]) if records else r"$g$")
        ax.set_ylabel(r"$m$")
        if records:
            ax.legend(fontsize=5.8)
    return _save(figure, output_dir, "phase_order_parameter")


def plot_common_coordinates(
    aggregate: dict[str, Any] | str | Path, output_dir: str | Path
) -> Path:
    data = read_aggregate(aggregate) if isinstance(aggregate, (str, Path)) else aggregate
    _style()
    metrics = (
        ("effective_multiplicity", "effective multiplicity"),
        ("interaction_range", "interaction range"),
        ("oracle_gap", "oracle gap"),
    )
    figure, axes = plt.subplots(
        len(metrics), len(DOMAIN_ORDER), figsize=(6.4, 4.8), constrained_layout=True, squeeze=False
    )
    for column, domain in enumerate(DOMAIN_ORDER):
        records = [value for value in data["records"] if value["domain"] == domain]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][column]
            for index, (label, _, values) in enumerate(_representative_series(records, metric, 6)):
                _errorbar(ax, values, metric, label, index)
            if row == 0:
                ax.set_title(DOMAIN_TITLES[domain], fontweight="bold")
            if column == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, ylabel))
            if row == len(metrics) - 1:
                ax.set_xlabel(r"$g$")
            if row == 0 and records:
                ax.legend(fontsize=5.2)
    return _save(figure, output_dir, "common_coordinates")


def plot_finite_size(
    aggregate: dict[str, Any] | str | Path, output_dir: str | Path
) -> Path:
    data = read_aggregate(aggregate) if isinstance(aggregate, (str, Path)) else aggregate
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), constrained_layout=True)
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        records = [value for value in data["records"] if value["domain"] == domain]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for value in records:
            groups[(value.get("family", "anchor"), value["variant"], _parameter_key(value))].append(value)
        items = sorted(groups.items())
        if len(items) > 8:
            items = [items[index] for index in np.linspace(0, len(items) - 1, 8, dtype=int)]
        for index, ((family, variant, _), values) in enumerate(items):
            peak_values = []
            for size in sorted({int(value["size"]) for value in values}):
                candidates = [value for value in values if int(value["size"]) == size]
                peak_values.append(
                    max(candidates, key=lambda value: float(value["metrics"]["susceptibility"]["mean"]))
                )
            ax.errorbar(
                [int(value["size"]) for value in peak_values],
                [float(value["metrics"]["order_parameter"]["mean"]) for value in peak_values],
                yerr=[float(value["metrics"]["order_parameter"]["ci95"]) for value in peak_values],
                label=f"{family}/{variant}",
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                linewidth=1.2,
                capsize=2.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(DOMAIN_TITLES[domain], loc="left", fontweight="bold")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$m(g_\chi)$")
        ax.grid(ls="--", color="0.82", linewidth=0.7, alpha=0.9)
        if items:
            ax.legend(fontsize=5.5)
    return _save(figure, output_dir, "finite_size_peak")


def _registered_panel(
    data: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str,
    title: str,
    domains: Iterable[str],
    families: Iterable[str] | None,
    metrics: tuple[tuple[str, str], ...],
) -> Path:
    _style()
    allowed_domains = set(domains)
    allowed_families = None if families is None else set(families)
    records = [
        value
        for value in data["records"]
        if value["domain"] in allowed_domains
        and (allowed_families is None or value.get("family", "anchor") in allowed_families)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), constrained_layout=True)
    figure.suptitle(title, fontweight="bold")
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        metric_records = [record for record in records if metric in record["metrics"]]
        for index, (label, _, values) in enumerate(_representative_series(metric_records, metric, 10)):
            _errorbar(ax, values, metric, label, index)
        ax.set_xlabel(_control_label(metric_records[0]["control_name"]) if metric_records else r"$g$")
        ax.set_ylabel(METRIC_LABELS.get(metric, ylabel))
        if metric_records:
            ax.legend(fontsize=5.2, ncol=2 if len(metric_records) > 20 else 1)
    return _save(figure, output_dir, stem)


def plot_registered_figures(
    aggregate: dict[str, Any] | str | Path, output_dir: str | Path
) -> list[Path]:
    data = read_aggregate(aggregate) if isinstance(aggregate, (str, Path)) else aggregate
    specifications = (
        ("figure1_architecture", "Architecture and data continuation", ("transformer",), ("architecture",),
         (("order_parameter", "recovery"), ("susceptibility", "susceptibility"), ("binder_cumulant", "Binder cumulant"), ("generalization_error", "generalization error"))),
        ("figure2_heads_algorithms", "Head symmetry and attention-MLP phases", ("transformer",), ("heads", "attention_mlp"),
         (("order_parameter", "functional order"), ("effective_multiplicity", "effective heads"), ("intervention_response", "ablation response"), ("oracle_gap", "oracle gap"))),
        ("figure3_icl_lifecycle", "ICL, context, adaptation, optimization, and lifecycle", ("transformer",), ("icl", "long_context", "lora", "glass", "optimizer", "lifecycle", "learned_decoder"),
         (("order_parameter", "order"), ("generalization_error", "generalization error"), ("interaction_range", "interaction range"), ("intervention_response", "response"))),
        ("figure4_systems", "MoE, retrieval, multimodal, generation, and compression", ("transformer",), ("moe", "retrieval", "multimodal", "compression", "generation"),
         (("effective_multiplicity", "effective multiplicity"), ("generalization_error", "error"), ("interaction_range", "interaction range"), ("intervention_response", "response"))),
        ("figure5_diffusion", "Diffusion phase cube and trainable score bridge", ("diffusion",), None,
         (("order_parameter", "semantic order"), ("susceptibility", "susceptibility"), ("oracle_gap", "oracle gap"), ("intervention_response", "guidance/locality response"))),
        ("figure6_rl", "RL entropy, Goodhart, and learned-policy phase cube", ("reinforcement",), None,
         (("order_parameter", "policy order"), ("effective_multiplicity", "policy multiplicity"), ("oracle_gap", "oracle gap"), ("intervention_response", "policy response"))),
        ("figure7_multiagent", "Collective reasoning and learned communication", ("multiagent",), None,
         (("order_parameter", "collective order"), ("effective_multiplicity", "effective agents"), ("oracle_gap", "oracle gap"), ("intervention_response", "global-flip response"))),
        ("figure8_cross_domain", "Matched-latent and assumption-graph continuation", ("cross_domain",), None,
         (("pair_interaction_delta", "pair nonadditivity"), ("bridge_error", "held-out bridge error"), ("response_slope", "critical response"), ("threshold_separation", "threshold separation"))),
    )
    return [
        _registered_panel(
            data,
            output_dir,
            stem=stem,
            title=title,
            domains=domains,
            families=families,
            metrics=metrics,
        )
        for stem, title, domains, families, metrics in specifications
    ]


def plot_all(aggregate: dict[str, Any] | str | Path, output_dir: str | Path) -> list[Path]:
    return [
        plot_phase_panels(aggregate, output_dir),
        plot_common_coordinates(aggregate, output_dir),
        plot_finite_size(aggregate, output_dir),
        *plot_registered_figures(aggregate, output_dir),
    ]


__all__ = [
    "DOMAIN_ORDER", "DOMAIN_TITLES", "plot_all", "plot_common_coordinates",
    "plot_finite_size", "plot_phase_panels", "plot_registered_figures",
]
