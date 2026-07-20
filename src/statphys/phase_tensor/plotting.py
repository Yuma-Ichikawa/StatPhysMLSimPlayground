"""Publication figures for the empirical phase-continuation tensor."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "1", "2", "3"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
_VARIANT_LABELS = {
    "none": "no MLP",
    "linear": "linear",
    "relu": "ReLU",
    "gelu": "GELU",
    "geglu": "GEGLU",
    "swiglu": "SwiGLU",
    "sgd_m": "SGD-M",
    "adamw": "AdamW",
    "muon": "Muon",
    "soap": "SOAP",
    "lion": "Lion",
    "galore": "GaLore",
}


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


def _decorate(ax: Any) -> None:
    ax.grid(ls="--", color="0.82", linewidth=0.7, alpha=0.9)
    ax.tick_params(which="both", top=True, right=True, labelsize=12)


def _metric(condition: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = condition.get("metrics", {}).get(name)
    if not value:
        return None
    return float(value["mean"]), float(value["ci95"])


def _label(value: str) -> str:
    return _VARIANT_LABELS.get(value, value.replace("_", " "))


def _math_label(value: str) -> str:
    """Wrap taxonomy TeX labels so matplotlib parses them as mathtext."""
    text = str(value)
    if "\\" in text and not (text.startswith("$") and text.endswith("$")):
        return f"${text}$"
    return text


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
            label=_label(label),
        )
    if reference is not None:
        ax.axhline(reference, color="0.30", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _decorate(ax)
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
            estimate = _metric(item, "normalized_test_risk") or _metric(item, "normalized_generalization_error")
            if estimate is not None:
                scores[float(item["parameters"].get("learning_rate", 0.0))].append(estimate[0])
        if not scores:
            continue
        best_rate = min(scores, key=lambda rate: float(np.mean(scores[rate])))
        selected.extend(
            item for item in variant_conditions if float(item["parameters"].get("learning_rate", 0.0)) == best_rate
        )
    return selected


def _dynamics_figure(aggregate: dict[str, Any], destination: Path) -> Path | None:
    dynamics = [
        item for item in aggregate.get("dynamics", [])
        if item["family"] == "tensor_mlp" and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    if not dynamics:
        return None
    largest_size = max(item["size"] for item in dynamics)
    largest_control = max(item["control"] for item in dynamics if item["size"] == largest_size)
    chosen = [item for item in dynamics if item["size"] == largest_size and item["control"] == largest_control]
    chosen = sorted(chosen, key=lambda item: item["variant"])[:4]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), constrained_layout=True, sharex=True)
    for index, item in enumerate(chosen):
        x = np.asarray(item["steps"], dtype=float)
        metrics = item["metrics"]
        for ax, metric, ylabel in (
            (axes[0], "history_train_risk", r"$R_{\rm train}=\ell_{\rm train}/\log V$"),
            (axes[1], "history_generalization_gap", r"$e_{\rm gen}=R_{\rm test}-R_{\rm train}$"),
        ):
            value = metrics[metric]
            mean = np.asarray(value["mean"], dtype=float)
            error = np.asarray(value["ci95"], dtype=float)
            color = COLORS[index % len(COLORS)]
            ax.plot(x, mean, color=color, marker=MARKERS[index], markersize=3.8, label=_label(item["variant"]))
            ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.16, linewidth=0.0)
            ax.set_xlabel(r"optimizer step $t$")
            ax.set_ylabel(ylabel)
            _decorate(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _mechanism_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [
        item for item in conditions
        if item["family"] == "tensor_mlp" and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    if not any("mlp_jacobian_effective_rank" in item["metrics"] for item in selected):
        return None
    size = max(item["size"] for item in selected)
    panels = (
        ("mlp_participation_fraction", r"$\mathrm{PR}_{\rm MLP}/d_{\rm ff}$"),
        ("mlp_activation_sparsity", r"$s_{\rm act}$"),
        ("mlp_gate_saturation", r"$s_{\rm gate}$"),
        ("mlp_jacobian_effective_rank", r"$\mathrm{PR}(JJ^\top)/d$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), constrained_layout=True, sharex=True)
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        series = _series(selected, "tensor_mlp", metric, lambda item: item["variant"], size=size)
        for index, (label, (x, y, error)) in enumerate(sorted(series.items())):
            ax.errorbar(x, y, yerr=error, color=COLORS[index], marker=MARKERS[index], linewidth=1.2, capsize=2.4, label=_label(label))
        ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
        ax.set_ylabel(ylabel)
        _decorate(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _coverage_figure(taxonomy_path: str | Path, destination: Path) -> Path:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    payload = tomllib.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    columns = list(payload["columns"])
    axes = list(payload["axes"])
    matrix = np.asarray([[int(column in payload["axes"][axis]) for column in columns] for axis in axes])
    labels = payload.get("axis_labels", {})
    figure, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap=ListedColormap(["#e7e7e7", "#167a8b"]), vmin=0, vmax=1, aspect="auto")
    del image
    for row, axis in enumerate(axes):
        for column, status in enumerate(columns):
            ax.text(column, row, status if matrix[row, column] else "N0", ha="center", va="center", fontsize=9)
    ax.set_xticks(np.arange(len(columns)), [_math_label(payload["column_labels"][column]) for column in columns], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(axes)), [_math_label(labels.get(axis, axis)) for axis in axes])
    ax.set_xlabel("theory-to-realism evidence tier")
    ax.set_ylabel("phase-continuation coordinate")
    ax.grid(ls="--", color="white", linewidth=0.9)
    ax.legend(
        handles=[Patch(facecolor="#167a8b", label="registered coverage"), Patch(facecolor="#e7e7e7", label="N0: untested")],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        fontsize=8,
    )
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_phase_tensor(
    aggregate_path: str | Path,
    output_directory: str | Path,
    *,
    taxonomy_path: str | Path | None = None,
) -> list[Path]:
    _style()
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    conditions = aggregate["conditions"]
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    alpha = r"$\alpha=N_{\rm train}/d^\gamma$"

    primary_mlp = [item for item in conditions if item["family"] == "tensor_mlp" and item["parameters"].get("data_kind") == "synthetic_retrieval"]
    mlp_size = max((item["size"] for item in primary_mlp), default=None)
    outputs.append(_line_figure(_series(primary_mlp, "tensor_mlp", "semantic_order", lambda item: item["variant"], size=mlp_size), destination / "figure09_mlp_phase_splitting.pdf", alpha, r"$m_{\rm task}=1-R_{\rm test}$", reference=0.5))
    outputs.append(_line_figure(_series(primary_mlp, "tensor_mlp", "mlp_causal_contribution", lambda item: item["variant"], size=mlp_size), destination / "figure10_mlp_causal_contribution.pdf", alpha, r"$\Delta_{\rm MLP}$", reference=0.0))

    optimizer_slice = _optimizer_learning_rate_slice(conditions)
    optimizer_size = max((item["size"] for item in optimizer_slice), default=None)
    primary_optimizer = [item for item in optimizer_slice if item["parameters"].get("normalization") == "pre_rmsnorm"]
    risk_metric = "normalized_test_risk" if any("normalized_test_risk" in item["metrics"] for item in primary_optimizer) else "normalized_generalization_error"
    outputs.append(_line_figure(_series(primary_optimizer, "tensor_optimizer", risk_metric, lambda item: item["variant"], size=optimizer_size), destination / "figure11_optimizer_geometry.pdf", alpha, r"$R_{\rm test}=\ell_{\rm test}/\log V$"))
    outputs.append(_line_figure(_series([item for item in optimizer_slice if item["variant"] in {"adamw", "muon"}], "tensor_optimizer", "gradient_block_gini", lambda item: f"{item['variant']} ({item['parameters'].get('normalization', 'default')})", size=optimizer_size), destination / "figure12_optimizer_gradient_heterogeneity.pdf", alpha, r"$G_{\rm block}(\|g_b\|)$"))

    objective_size = max((item["size"] for item in conditions if item["family"] == "tensor_objective"), default=None)
    outputs.append(_line_figure(_series(conditions, "tensor_objective", "normalized_ce", lambda item: f"{item['variant']} ({item['parameters'].get('activation', 'default')})", size=objective_size), destination / "figure13_objective_homotopy.pdf", alpha, r"$\tilde\ell_{\rm CE}=\ell_{\rm CE}/\log V$"))

    scaling = [item for item in conditions if item["family"] == "tensor_scaling"]
    scale_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for variant in sorted({item["variant"] for item in scaling}):
        subset = [item for item in scaling if item["variant"] == variant]
        target_control = max((item["control"] for item in subset), default=None)
        points = [
            (item["size"], estimate[0], estimate[1])
            for item in subset if item["control"] == target_control
            for estimate in [_metric(item, "normalized_test_risk") or _metric(item, "normalized_generalization_error")]
            if estimate is not None
        ]
        if points:
            points.sort()
            scale_series[variant] = tuple(np.asarray(component) for component in zip(*points, strict=True))
    outputs.append(_line_figure(scale_series, destination / "figure14_scaling_paths.pdf", r"$N_{\rm path}$", r"$R_{\rm test}$", xscale="log", yscale="log"))

    real_size = max((item["size"] for item in conditions if item["family"] == "tensor_realdata"), default=None)
    outputs.append(_line_figure(_series(conditions, "tensor_realdata", "bits_per_byte", lambda item: f"{item['variant']} ({item['parameters'].get('activation')})", size=real_size), destination / "figure15_natural_data_bridge.pdf", alpha, r"$\ell_{\rm test}/\log 2$ [bits byte$^{-1}$]"))
    has_generalization_gap = any("normalized_generalization_gap" in item["metrics"] for item in conditions)
    gap_metric = "normalized_generalization_gap" if has_generalization_gap else "normalized_generalization_error"
    gap_label = r"$e_{\rm gen}=R_{\rm test}-R_{\rm train}$" if has_generalization_gap else r"$R_{\rm test}$ (legacy aggregate)"
    outputs.append(_line_figure(_series(conditions, "tensor_realdata", gap_metric, lambda item: f"{item['variant']} ({item['parameters'].get('activation')})", size=real_size), destination / "figure16_natural_data_generalization.pdf", alpha, gap_label, reference=0.0 if has_generalization_gap else None))

    compute_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family in ("tensor_mlp", "tensor_optimizer", "tensor_objective", "tensor_scaling", "tensor_realdata", "tensor_residual"):
        subset = [item for item in conditions if item["family"] == family]
        if not subset:
            continue
        representative_size = max(item["size"] for item in subset)
        points = []
        for item in subset:
            if item["size"] != representative_size:
                continue
            estimate = _metric(item, "normalized_test_risk") or _metric(item, "normalized_generalization_error")
            compute = _metric(item, "training_flops_estimate")
            if estimate is not None and compute is not None:
                points.append((compute[0], estimate[0], estimate[1]))
        if points:
            points.sort()
            compute_series[family.replace("tensor_", "")] = tuple(np.asarray(component) for component in zip(*points, strict=True))
    outputs.append(_line_figure(compute_series, destination / "figure17_compute_error_landscape.pdf", r"$C_{\rm train}\simeq6PN_{\rm tok}$", r"$R_{\rm test}$", xscale="log"))

    if taxonomy_path is not None:
        outputs.append(_coverage_figure(taxonomy_path, destination / "figure18_theory_experiment_coverage.pdf"))
    dynamics = _dynamics_figure(aggregate, destination / "figure19_training_generalization_dynamics.pdf")
    if dynamics is not None:
        outputs.append(dynamics)
    mechanism = _mechanism_figure(conditions, destination / "figure20_mlp_mechanism_atlas.pdf")
    if mechanism is not None:
        outputs.append(mechanism)
    return outputs


__all__ = ["plot_phase_tensor"]
