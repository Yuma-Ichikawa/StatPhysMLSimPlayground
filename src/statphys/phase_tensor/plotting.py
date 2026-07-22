"""Publication figures for the empirical phase-continuation tensor."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Callable
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


FIGURE_SIZE = (6.4, 4.8)
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "h"]
COLORS = ["#176B87", "#D97706", "#2F855A", "#B8323C", "#6B5B95", "#8C5A3C", "#C44569", "#536878"]
STYLE_ORDER = (
    "none", "linear", "relu", "gelu", "geglu", "swiglu",
    "sgd_m", "adamw", "muon", "soap", "lion", "galore",
    "tinystories", "simplestories", "dolma", "fineweb_edu",
    "width", "depth", "context", "data",
    "mlp", "optimizer", "objective", "scaling", "realdata", "residual",
)
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
    "tinystories": "TinyStories",
    "simplestories": "SimpleStories",
    "dolma": "Dolma",
    "fineweb_edu": "FineWeb-Edu",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12.5,
        "axes.labelsize": 15,
        "axes.titlesize": 13.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
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


def _subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    kwargs["figsize"] = FIGURE_SIZE
    kwargs.setdefault("constrained_layout", True)
    figure, axes = plt.subplots(*args, **kwargs)
    figure.patch.set_facecolor("white")
    return figure, axes


def _decorate(ax: Any) -> None:
    ax.grid(ls="--", color="0.82", linewidth=0.7, alpha=0.9)
    ax.tick_params(which="both", top=True, right=True, labelsize=12.5)


def _metric(condition: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = condition.get("metrics", {}).get(name)
    if not value:
        return None
    mean = float(value["mean"])
    interval = float(value["ci95"])
    if not math.isfinite(mean) or not math.isfinite(interval):
        raise ValueError(f"non-finite aggregate metric {name!r}")
    return mean, interval


def _risk_metric(condition: dict[str, Any]) -> tuple[float, float] | None:
    return _metric(condition, "normalized_test_risk") or _metric(
        condition, "normalized_generalization_error"
    )


def _label(value: str) -> str:
    head, separator, tail = value.partition(" (")
    translated = _VARIANT_LABELS.get(head, head.replace("_", " "))
    return translated + (separator + tail if separator else "")


def _math_label(value: str) -> str:
    text = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", str(value))
    if "\\" in text and "$" not in text:
        return f"${text}$"
    return text


def _style_index(label: str, fallback: int = 0) -> int:
    head = label.partition(" (")[0]
    try:
        return STYLE_ORDER.index(head)
    except ValueError:
        return len(STYLE_ORDER) + fallback


def _ordered_items(series: dict[str, Any]) -> list[tuple[str, Any]]:
    return sorted(series.items(), key=lambda item: (_style_index(item[0]), item[0]))


def _errorbar(ax: Any, label: str, x: np.ndarray, y: np.ndarray, error: np.ndarray, fallback: int) -> None:
    index = _style_index(label, fallback)
    ax.errorbar(
        x,
        y,
        yerr=error,
        color=COLORS[index % len(COLORS)],
        linestyle=LINE_STYLES[(index // len(COLORS)) % len(LINE_STYLES)],
        marker=MARKERS[index % len(MARKERS)],
        markersize=5.5,
        linewidth=1.6,
        capsize=3.0,
        label=_label(label),
    )


def _series(
    conditions: list[dict[str, Any]],
    family: str,
    metric: str,
    group: Callable[[dict[str, Any]], str],
    *,
    size: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    chosen = [
        item for item in conditions
        if item["family"] == family and (size is None or item["size"] == size)
    ]
    buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for item in chosen:
        estimate = _metric(item, metric)
        if estimate is not None:
            buckets[group(item)].append((float(item["control"]), estimate[0], estimate[1]))
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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
    figure, ax = _subplots()
    for fallback, (label, (x, y, error)) in enumerate(_ordered_items(series)):
        _errorbar(ax, label, x, y, error, fallback)
    if reference is not None:
        ax.axhline(reference, color="0.30", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _decorate(ax)
    if series:
        ax.legend(frameon=False, ncol=2)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _phase_splitting_figure(
    aggregate: dict[str, Any], selected: list[dict[str, Any]], destination: Path
) -> Path:
    size = max((item["size"] for item in selected), default=None)
    chosen = [item for item in selected if item["size"] == size]
    series = _series(chosen, "tensor_mlp", "semantic_order", lambda item: item["variant"], size=size)
    figure, ax = _subplots()
    for fallback, (label, (x, y, error)) in enumerate(_ordered_items(series)):
        _errorbar(ax, label, x, y, error, fallback)
    ax.axhline(0.5, color="0.30", linewidth=1.0, linestyle=":", zorder=0)

    identified_label = True
    for boundary in aggregate.get("boundaries", []):
        if (
            boundary.get("status") == "identified"
            and boundary.get("family") == "tensor_mlp"
            and boundary.get("size") == size
            and boundary.get("parameters", {}).get("data_kind") == "synthetic_retrieval"
        ):
            value = float(boundary["semantic_order_half"]["mean"])
            ax.axvline(
                value,
                color="0.20",
                linewidth=1.1,
                linestyle="--",
                alpha=0.8,
                label="identified threshold" if identified_label else None,
            )
            identified_label = False

    censor_labels = {"left_censored": True, "right_censored": True}
    for threshold in aggregate.get("thresholds", []):
        if (
            threshold.get("family") != "tensor_mlp"
            or threshold.get("size") != size
            or threshold.get("parameters", {}).get("data_kind") != "synthetic_retrieval"
        ):
            continue
        estimates = threshold.get("estimates", [])
        for offset, estimate in enumerate(estimates):
            status = estimate.get("status")
            if status not in censor_labels:
                continue
            marker = "<" if status == "left_censored" else ">"
            wording = "left-censored threshold" if status == "left_censored" else "right-censored threshold"
            ax.scatter(
                [float(estimate["value"])],
                [0.5 + 0.008 * (offset - (len(estimates) - 1) / 2.0)],
                marker=marker,
                s=34,
                facecolors="white",
                edgecolors="#B8323C",
                linewidths=1.1,
                zorder=4,
                label=wording if censor_labels[status] else None,
            )
            censor_labels[status] = False
    ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
    ax.set_ylabel(r"$m_{\rm task}=1-R_{\rm test}$")
    _decorate(ax)
    if series:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _bounded_causal_metric(condition: dict[str, Any], branch: str) -> tuple[float, float] | None:
    direct = _metric(condition, f"{branch}_causal_effect")
    if direct is not None:
        return direct
    full = _metric(condition, "full_risk") or _metric(condition, "normalized_test_risk")
    delta = _metric(condition, f"{branch}_risk_delta")
    if delta is None:
        ablated = _metric(condition, f"{branch}_ablated_risk")
        if ablated is not None and full is not None:
            delta = (ablated[0] - full[0], ablated[1] + full[1])
    if delta is None or full is None:
        return None

    def transform(delta_value: float, full_value: float) -> float:
        return delta_value / (abs(delta_value) + max(full_value, 0.0) + 1e-12)

    mean = transform(delta[0], full[0])
    candidates = [
        transform(delta[0] + delta_sign * delta[1], max(0.0, full[0] + full_sign * full[1]))
        for delta_sign in (-1.0, 1.0)
        for full_sign in (-1.0, 1.0)
    ]
    return mean, max(abs(value - mean) for value in candidates)


def _causal_effect_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [
        item for item in conditions
        if item["family"] == "tensor_mlp"
        and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    size = max((item["size"] for item in selected), default=None)
    selected = [item for item in selected if item["size"] == size]
    panels = (
        ("attention", r"$\eta_{\mathcal{A}}=\Delta R_{\mathcal{A}}/(|\Delta R_{\mathcal{A}}|+R_{\rm full}+\epsilon)$"),
        ("mlp", r"$\eta_{\mathcal{F}}=\Delta R_{\mathcal{F}}/(|\Delta R_{\mathcal{F}}|+R_{\rm full}+\epsilon)$"),
    )
    panel_series: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for branch, _ in panels:
        buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for item in selected:
            estimate = _bounded_causal_metric(item, branch)
            if estimate is not None:
                buckets[item["variant"]].append((float(item["control"]), estimate[0], estimate[1]))
        prepared = {}
        for label, points in buckets.items():
            points.sort()
            prepared[label] = tuple(np.asarray(component) for component in zip(*points, strict=True))
        panel_series.append(prepared)
    if not any(panel_series):
        if selected:
            warnings.warn(
                "Figure 10 omitted: aggregate has only deprecated unstable causal metrics; re-aggregate raw runs",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    figure, axes = _subplots(1, 2, sharex=True, sharey=True)
    for ax, (_, ylabel), series in zip(axes, panels, panel_series, strict=True):
        for fallback, (label, (x, y, error)) in enumerate(_ordered_items(series)):
            _errorbar(ax, label, x, y, error, fallback)
        ax.axhline(0.0, color="0.30", linewidth=1.0, linestyle=":", zorder=0)
        ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.02, 1.02)
        _decorate(ax)
    if panel_series[0]:
        axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _optimizer_response_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [
        item for item in conditions
        if item["family"] == "tensor_optimizer"
        and item["parameters"].get("normalization") == "pre_rmsnorm"
    ]
    if not selected:
        return None
    size = max(item["size"] for item in selected)
    selected = [item for item in selected if item["size"] == size]
    variants = sorted({item["variant"] for item in selected}, key=lambda value: (_style_index(value), value))
    columns = 2
    rows = max(1, math.ceil(len(variants) / columns))
    figure, axes = _subplots(rows, columns, squeeze=False, sharex=True, sharey=True)
    risks = [estimate[0] for item in selected for estimate in [_risk_metric(item)] if estimate is not None]
    if not risks:
        plt.close(figure)
        return None
    vmin, vmax = min(risks), max(risks)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    mesh = None
    for ax, variant in zip(axes.flat, variants, strict=False):
        subset = [item for item in selected if item["variant"] == variant]
        coefficients = sorted({float(item["control"]) for item in subset})
        rates = sorted({float(item["parameters"]["learning_rate"]) for item in subset})
        surface = np.full((len(rates), len(coefficients)), np.nan)
        for item in subset:
            estimate = _risk_metric(item)
            if estimate is not None:
                row = rates.index(float(item["parameters"]["learning_rate"]))
                column = coefficients.index(float(item["control"]))
                surface[row, column] = estimate[0]
        mesh = ax.pcolormesh(
            coefficients,
            rates,
            np.ma.masked_invalid(surface),
            shading="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
        ax.set_ylabel(r"learning rate $\eta$")
        ax.set_title(_label(variant))
        _decorate(ax)
    for ax in axes.flat[len(variants):]:
        ax.set_visible(False)
    if mesh is not None:
        figure.colorbar(mesh, ax=list(axes.flat[:len(variants)]), label=r"$R_{\rm test}$", shrink=0.88)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _gradient_correlation_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [item for item in conditions if item["family"] == "tensor_optimizer"]
    if not selected:
        return None
    size = max(item["size"] for item in selected)
    selected = [item for item in selected if item["size"] == size]
    figure, ax = _subplots()
    plotted = False
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        groups[f"{item['variant']} ({item['parameters'].get('normalization', 'default')})"].append(item)
    for fallback, (label, items) in enumerate(_ordered_items(groups)):
        index = _style_index(label, fallback)
        first = True
        for item in items:
            geometry = _metric(item, "gradient_block_gini")
            risk = _risk_metric(item)
            if geometry is None or risk is None:
                continue
            ax.errorbar(
                geometry[0], risk[0], xerr=geometry[1], yerr=risk[1],
                color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)],
                linestyle="none", capsize=2.0, markersize=5.2,
                label=_label(label) if first else None,
            )
            first = False
            plotted = True
    if not plotted:
        plt.close(figure)
        return None
    ax.set_xlabel(r"gradient geometry $G_{\rm block}(\|g_b\|)$")
    ax.set_ylabel(r"$R_{\rm test}$")
    ax.set_title("Correlational diagnostic (no causal attribution)")
    _decorate(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _scaling_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [item for item in conditions if item["family"] == "tensor_scaling"]
    if not selected:
        return None
    panels = (
        ("width", "model_width", r"model width $d_{\rm model}$"),
        ("depth", "model_depth", r"depth $L$"),
        ("context", "context_length", r"context length $T_{\rm ctx}$"),
        ("data", "train_examples", r"training examples $N_{\rm train}$"),
    )
    figure, axes = _subplots(2, 2)
    for panel_index, (ax, (variant, xmetric, xlabel)) in enumerate(zip(axes.flat, panels, strict=True)):
        subset = [item for item in selected if item["variant"] == variant]
        target_control = max((float(item["control"]) for item in subset), default=None)
        subset = [item for item in subset if float(item["control"]) == target_control]
        buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for item in subset:
            x_estimate = _metric(item, xmetric)
            risk = _risk_metric(item)
            if x_estimate is None and variant != "data":
                x_estimate = (float(item["size"]), 0.0)
            if x_estimate is None or risk is None:
                continue
            exponent = float(item["parameters"].get("sample_exponent", 1.0))
            buckets[f"gamma={exponent:g}"].append((x_estimate[0], risk[0], risk[1]))
        for fallback, (label, points) in enumerate(sorted(buckets.items())):
            points.sort()
            x, y, error = (np.asarray(component) for component in zip(*points, strict=True))
            _errorbar(ax, label, x, y, error, fallback)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$R_{\rm test}$")
        ax.set_title(f"({chr(97 + panel_index)}) {_label(variant)}")
        if buckets:
            ax.set_xscale("log")
        _decorate(ax)
        if len(buckets) > 1:
            ax.legend(frameon=False, fontsize=7)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _natural_bridge_figure(conditions: list[dict[str, Any]], destination: Path) -> Path | None:
    selected = [item for item in conditions if item["family"] == "tensor_realdata"]
    if not selected:
        return None
    size = max(item["size"] for item in selected)
    group = lambda item: f"{item['variant']} ({item['parameters'].get('activation', 'default')})"
    panels = (
        ("semantic_order", r"natural-carrier order $m_{\rm task}=1-R_{\rm test}$", "semantic carrier"),
        ("bits_per_byte", r"transport $\ell_{\rm test}/\log 2$ [bits byte$^{-1}$]", "corpus protocol"),
    )
    figure, axes = _subplots(1, 2, sharex=True)
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        if metric == "semantic_order":
            metric_conditions = [
                item for item in selected if item["variant"] == "natural_injected"
            ]
        else:
            metric_conditions = [
                item for item in selected if item["variant"] != "natural_injected"
            ]
        series = _series(metric_conditions, "tensor_realdata", metric, group, size=size)
        for fallback, (label, (x, y, error)) in enumerate(_ordered_items(series)):
            _errorbar(ax, label, x, y, error, fallback)
        ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _decorate(ax)
        if not series:
            ax.text(
                0.5,
                0.5,
                "not defined for raw corpora",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    if axes[0].lines:
        axes[0].legend(frameon=False, fontsize=7)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _compute_regret_series(
    conditions: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    families = ("tensor_mlp", "tensor_optimizer", "tensor_objective", "tensor_scaling", "tensor_realdata", "tensor_residual")
    for family in families:
        subset = [item for item in conditions if item["family"] == family]
        if not subset:
            continue
        representative_size = max(item["size"] for item in subset)
        raw: list[tuple[float, float, float]] = []
        for item in subset:
            if item["size"] != representative_size:
                continue
            risk = _risk_metric(item)
            compute = _metric(item, "training_flops_estimate")
            if risk is not None and compute is not None:
                raw.append((compute[0], risk[0], risk[1]))
        if not raw:
            continue
        minimum = min(item[1] for item in raw)
        maximum = max(item[1] for item in raw)
        span = maximum - minimum
        normalized = [
            (compute, 0.0 if span <= 1e-12 else (risk - minimum) / span, 0.0 if span <= 1e-12 else interval / span)
            for compute, risk, interval in raw
        ]
        normalized.sort()
        output[family.replace("tensor_", "")] = tuple(
            np.asarray(component) for component in zip(*normalized, strict=True)
        )
    return output


def _dynamics_figure(aggregate: dict[str, Any], destination: Path) -> Path | None:
    dynamics = [
        item for item in aggregate.get("dynamics", [])
        if item["family"] == "tensor_mlp" and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    if not dynamics:
        return None
    largest_size = max(item["size"] for item in dynamics)
    largest_control = max(item["control"] for item in dynamics if item["size"] == largest_size)
    chosen = sorted(
        [item for item in dynamics if item["size"] == largest_size and item["control"] == largest_control],
        key=lambda item: (_style_index(item["variant"]), item["variant"]),
    )[:4]
    figure, axes = _subplots(1, 2, sharex=True)
    for fallback, item in enumerate(chosen):
        index = _style_index(item["variant"], fallback)
        x = np.asarray(item["steps"], dtype=float)
        for ax, metric, ylabel in (
            (axes[0], "history_train_risk", r"$R_{\rm train}=\ell_{\rm train}/\log V$"),
            (axes[1], "history_generalization_gap", r"$e_{\rm gen}=R_{\rm test}-R_{\rm train}$"),
        ):
            value = item["metrics"][metric]
            mean = np.asarray(value["mean"], dtype=float)
            error = np.asarray(value["ci95"], dtype=float)
            color = COLORS[index % len(COLORS)]
            ax.plot(x, mean, color=color, marker=MARKERS[index % len(MARKERS)], markersize=3.8, label=_label(item["variant"]))
            ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.16, linewidth=0.0)
            ax.set_xlabel(r"optimizer step $t$")
            ax.set_ylabel(ylabel)
            _decorate(ax)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(destination)
    plt.close(figure)
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
    figure, axes = _subplots(2, 2, sharex=True)
    for ax, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        series = _series(selected, "tensor_mlp", metric, lambda item: item["variant"], size=size)
        for fallback, (label, (x, y, error)) in enumerate(_ordered_items(series)):
            _errorbar(ax, label, x, y, error, fallback)
        ax.set_xlabel(r"$\alpha=N_{\rm train}/d^\gamma$")
        ax.set_ylabel(ylabel)
        _decorate(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _coverage_figure(taxonomy_path: str | Path, destination: Path) -> Path:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    payload = tomllib.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    columns = list(payload["columns"])
    coverage = payload.get("coverage")
    if coverage is None:
        field = payload["axes"]
        paper = {axis: [] for axis in field}
    else:
        field = coverage["field_frontier"]
        paper = coverage["this_paper"]
    axis_names = list(field)
    axis_labels = payload.get("axis_labels", {})
    panels = (("Field frontier", field, "#167A8B"), ("This paper", paper, "#C56A17"))
    figure, axes = _subplots(1, 2, sharey=True)
    for ax, (title, entries, color) in zip(axes, panels, strict=True):
        matrix = np.asarray([[int(column in entries.get(axis, [])) for column in columns] for axis in axis_names])
        ax.imshow(matrix, cmap=ListedColormap(["#ECECEC", color]), vmin=0, vmax=1, aspect="auto")
        for row, axis in enumerate(axis_names):
            for column, status in enumerate(columns):
                ax.text(column, row, status if matrix[row, column] else "N0", ha="center", va="center", fontsize=8.5)
        ax.set_xticks(np.arange(len(columns)), [_math_label(payload["column_labels"][column]) for column in columns], rotation=32, ha="right")
        ax.set_yticks(np.arange(len(axis_names)), [_math_label(axis_labels.get(axis, axis)) for axis in axis_names])
        ax.set_xlabel("evidence tier")
        ax.set_title(title)
        ax.grid(ls="--", color="white", linewidth=0.9)
        ax.tick_params(which="both", labelsize=10.5)
    axes[0].set_ylabel("phase-continuation coordinate")
    figure.legend(
        handles=[Patch(facecolor="#167A8B", label="field coverage"), Patch(facecolor="#C56A17", label="this-paper evidence"), Patch(facecolor="#ECECEC", label="N0: untested")],
        frameon=False,
        loc="lower center",
        ncol=3,
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

    primary_mlp = [
        item for item in conditions
        if item["family"] == "tensor_mlp" and item["parameters"].get("data_kind") == "synthetic_retrieval"
    ]
    outputs.append(_phase_splitting_figure(aggregate, primary_mlp, destination / "figure09_mlp_phase_splitting.pdf"))
    causal = _causal_effect_figure(primary_mlp, destination / "figure10_mlp_causal_contribution.pdf")
    if causal is not None:
        outputs.append(causal)

    optimizer = _optimizer_response_figure(conditions, destination / "figure11_optimizer_geometry.pdf")
    if optimizer is not None:
        outputs.append(optimizer)
    gradients = _gradient_correlation_figure(conditions, destination / "figure12_optimizer_gradient_heterogeneity.pdf")
    if gradients is not None:
        outputs.append(gradients)

    objective_size = max((item["size"] for item in conditions if item["family"] == "tensor_objective"), default=None)
    outputs.append(_line_figure(
        _series(conditions, "tensor_objective", "normalized_ce", lambda item: f"{item['variant']} ({item['parameters'].get('activation', 'default')})", size=objective_size),
        destination / "figure13_objective_homotopy.pdf", alpha, r"$\tilde\ell_{\rm CE}=\ell_{\rm CE}/\log V$",
    ))

    scaling = _scaling_figure(conditions, destination / "figure14_scaling_paths.pdf")
    if scaling is not None:
        outputs.append(scaling)
    natural = _natural_bridge_figure(conditions, destination / "figure15_natural_data_bridge.pdf")
    if natural is not None:
        outputs.append(natural)

    real_size = max((item["size"] for item in conditions if item["family"] == "tensor_realdata"), default=None)
    has_gap = any("normalized_generalization_gap" in item["metrics"] for item in conditions if item["family"] == "tensor_realdata")
    gap_metric = "normalized_generalization_gap" if has_gap else "normalized_generalization_error"
    gap_label = r"$e_{\rm gen}=R_{\rm test}-R_{\rm train}$" if has_gap else r"$R_{\rm test}$ (legacy aggregate)"
    outputs.append(_line_figure(
        _series(conditions, "tensor_realdata", gap_metric, lambda item: f"{item['variant']} ({item['parameters'].get('activation', 'default')})", size=real_size),
        destination / "figure16_natural_data_generalization.pdf", alpha, gap_label,
        reference=0.0 if has_gap else None,
    ))

    outputs.append(_line_figure(
        _compute_regret_series(conditions),
        destination / "figure17_compute_error_landscape.pdf",
        r"$C_{\rm train}\simeq6PN_{\rm tok}$",
        r"within-family regret $r_f=(R-R_f^{\min})/(R_f^{\max}-R_f^{\min})$",
        xscale="log",
    ))

    if taxonomy_path is not None:
        outputs.append(_coverage_figure(taxonomy_path, destination / "figure18_theory_experiment_coverage.pdf"))
    dynamics = _dynamics_figure(aggregate, destination / "figure19_training_generalization_dynamics.pdf")
    if dynamics is not None:
        outputs.append(dynamics)
    mechanism = _mechanism_figure(conditions, destination / "figure20_mlp_mechanism_atlas.pdf")
    if mechanism is not None:
        outputs.append(mechanism)
    return outputs


__all__ = ["FIGURE_SIZE", "plot_phase_tensor"]
