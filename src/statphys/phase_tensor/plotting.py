"""Publication figures for the empirical phase-continuation tensor."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "1", "2", "3"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.linewidth": 1.0,
        "axes.xmargin": 0.01,
        "axes.ymargin": 0.01,
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    })


def _metric(condition: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = condition.get("metrics", {}).get(name)
    if not value:
        return None
    return float(value["mean"]), float(value["ci95"])


def _series(
    conditions: list[dict[str, Any]],
    family: str,
    metric: str,
    group: Callable[[dict[str, Any]], str],
    *,
    size: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    chosen = [item for item in conditions if item["family"] == family and (size is None or item["size"] == size)]
    buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for item in chosen:
        estimate = _metric(item, metric)
        if estimate is not None:
            buckets[group(item)].append((float(item["control"]), estimate[0], estimate[1]))
    output = {}
    for name, values in buckets.items():
        values.sort()
        output[name] = tuple(np.asarray(component) for component in zip(*values, strict=True))
    return output


def _line_figure(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    destination: Path,
    xlabel: str,
    ylabel: str,
    *,
    xscale: str = "linear",
    yscale: str = "linear",
    reference: float | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for index, (label, (x, y, error)) in enumerate(sorted(series.items())):
        ax.errorbar(
            x, y, yerr=error,
            color=COLORS[index % len(COLORS)],
            linestyle=LINE_STYLES[(index // len(COLORS)) % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=5.5,
            linewidth=1.6,
            capsize=3.0,
            label=label.replace("_", " "),
        )
    if reference is not None:
        ax.axhline(reference, color="0.35", linewidth=1.0, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(which="both", top=True, right=True)
    if series:
        ax.legend(frameon=False, ncol=2)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _optimizer_learning_rate_slice(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    optimizer_conditions = [item for item in conditions if item["family"] == "tensor_optimizer"]
    selected: list[dict[str, Any]] = []
    for variant in sorted({item["variant"] for item in optimizer_conditions}):
        variant_conditions = [item for item in optimizer_conditions if item["variant"] == variant]
        scores: dict[float, list[float]] = defaultdict(list)
        for item in variant_conditions:
            if item["parameters"].get("normalization") != "pre_rmsnorm":
                continue
            estimate = _metric(item, "normalized_generalization_error")
            if estimate is not None:
                scores[float(item["parameters"].get("learning_rate", 0.0))].append(estimate[0])
        if not scores:
            continue
        best_rate = min(scores, key=lambda rate: float(np.mean(scores[rate])))
        selected.extend(
            item
            for item in variant_conditions
            if float(item["parameters"].get("learning_rate", 0.0)) == best_rate
        )
    return selected


def plot_phase_tensor(aggregate_path: str | Path, output_directory: str | Path) -> list[Path]:
    _style()
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    conditions = aggregate["conditions"]
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    primary_mlp = [
        item
        for item in conditions
        if item["family"] == "tensor_mlp"
        and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    mlp_size = max((item["size"] for item in primary_mlp), default=None)
    outputs.append(_line_figure(
        _series(primary_mlp, "tensor_mlp", "semantic_order", lambda item: item["variant"], size=mlp_size),
        destination / "figure09_mlp_phase_splitting.pdf",
        "sample coefficient", r"semantic order $m$", reference=0.5,
    ))
    outputs.append(_line_figure(
        _series(primary_mlp, "tensor_mlp", "mlp_causal_contribution", lambda item: item["variant"], size=mlp_size),
        destination / "figure10_mlp_causal_contribution.pdf",
        "sample coefficient", "MLP causal contribution", reference=0.0,
    ))

    optimizer_slice = _optimizer_learning_rate_slice(conditions)
    optimizer_size = max((item["size"] for item in optimizer_slice), default=None)
    primary_optimizer = [
        item for item in optimizer_slice if item["parameters"].get("normalization") == "pre_rmsnorm"
    ]
    outputs.append(_line_figure(
        _series(primary_optimizer, "tensor_optimizer", "normalized_generalization_error", lambda item: item["variant"], size=optimizer_size),
        destination / "figure11_optimizer_geometry.pdf",
        "sample coefficient", r"normalized generalization error $e_g$", reference=0.0,
    ))
    outputs.append(_line_figure(
        _series(
            [item for item in optimizer_slice if item["variant"] in {"adamw", "muon"}],
            "tensor_optimizer",
            "gradient_block_gini",
            lambda item: f"{item['variant']} / {item['parameters'].get('normalization', 'default')}",
            size=optimizer_size,
        ),
        destination / "figure12_optimizer_gradient_heterogeneity.pdf",
        "sample coefficient", "gradient-block Gini coefficient",
    ))

    objective_size = max((item["size"] for item in conditions if item["family"] == "tensor_objective"), default=None)
    outputs.append(_line_figure(
        _series(
            conditions,
            "tensor_objective",
            "normalized_ce",
            lambda item: f"{item['variant']} / {item['parameters'].get('activation', 'default')}",
            size=objective_size,
        ),
        destination / "figure13_objective_homotopy.pdf",
        "sample coefficient", r"normalized cross entropy $\ell_{\rm CE}/\log V$",
    ))

    scaling = [item for item in conditions if item["family"] == "tensor_scaling"]
    scale_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    scaling_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scaling:
        label = item["variant"]
        if label == "data":
            label = f"data gamma={item['parameters'].get('sample_exponent')}"
        scaling_groups[label].append(item)
    for variant, subset in sorted(scaling_groups.items()):
        target_control = max((item["control"] for item in subset), default=None)
        points = []
        for item in subset:
            if item["control"] != target_control:
                continue
            estimate = _metric(item, "normalized_generalization_error")
            if estimate is not None:
                points.append((item["size"], estimate[0], estimate[1]))
        points.sort()
        if points:
            scale_series[variant] = tuple(np.asarray(component) for component in zip(*points, strict=True))
    outputs.append(_line_figure(
        scale_series,
        destination / "figure14_scaling_paths.pdf",
        "scaling coordinate", r"normalized generalization error $e_g$",
        xscale="log", yscale="log",
    ))

    real_size = max((item["size"] for item in conditions if item["family"] == "tensor_realdata"), default=None)
    outputs.append(_line_figure(
        _series(
            conditions,
            "tensor_realdata",
            "bits_per_byte",
            lambda item: f"{item['variant']} / {item['parameters'].get('activation')}",
            size=real_size,
        ),
        destination / "figure15_natural_data_bridge.pdf",
        "sample coefficient", "test bits per byte",
    ))
    outputs.append(_line_figure(
        _series(
            conditions,
            "tensor_realdata",
            "normalized_generalization_error",
            lambda item: f"{item['variant']} / {item['parameters'].get('activation')}",
            size=real_size,
        ),
        destination / "figure16_natural_data_generalization.pdf",
        "sample coefficient", r"normalized generalization error $e_g$", reference=0.0,
    ))

    compute_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family in ("tensor_mlp", "tensor_optimizer", "tensor_objective", "tensor_scaling", "tensor_realdata"):
        points = []
        for item in conditions:
            if item["family"] != family:
                continue
            compute = _metric(item, "training_flops_estimate")
            error = _metric(item, "normalized_generalization_error")
            if compute is not None and error is not None and error[0] > 0.0:
                points.append((compute[0], error[0], error[1]))
        points.sort()
        envelope = []
        best = float("inf")
        for point in points:
            if point[1] < best:
                envelope.append(point)
                best = point[1]
        points = envelope
        if points:
            compute_series[family] = tuple(np.asarray(component) for component in zip(*points, strict=True))
    outputs.append(_line_figure(
        compute_series,
        destination / "figure17_compute_error_landscape.pdf",
        "estimated training FLOPs", r"normalized generalization error $e_g$",
        xscale="log", yscale="log",
    ))
    return outputs


__all__ = ["plot_phase_tensor"]
