"""Headless, paper-oriented figures for tidy phase-atlas aggregates.

Every public plot requires a caller-supplied output directory and emits PDF
and high-resolution PNG by default. Plotting consumes aggregate tables only;
cluster paths and checkpoints never leak into reusable figure code.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import AtlasAggregate, build_ensemble_table
from .analysis import data_collapse_score, fit_finite_size_scaling_grid


_PAPER_RC: dict[str, Any] = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "mathtext.fontset": "stix",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


@dataclass(frozen=True)
class FigureOutput:
    """Paths and audit metadata for one multi-format paper figure."""

    paths: dict[str, Path]
    metadata: dict[str, Any]


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _rows(
    data: AtlasAggregate | Sequence[Mapping[str, Any]], table: str = "runs"
) -> list[Mapping[str, Any]]:
    if isinstance(data, AtlasAggregate):
        return list(data.ensembles if table == "ensembles" else data.runs)
    return list(data)


def _is_eligible(row: Mapping[str, Any]) -> bool:
    if "eligible_for_transition" in row:
        return bool(row["eligible_for_transition"])
    return bool(row.get("eligible_for_claims", True))


def _eligible(rows: Sequence[Mapping[str, Any]], enabled: bool) -> list[Mapping[str, Any]]:
    return [row for row in rows if _is_eligible(row)] if enabled else list(rows)


def _number(value: Any, component: int | None = None) -> float | None:
    if component is not None:
        if not isinstance(value, (list, tuple, np.ndarray)) or component >= len(value):
            return None
        value = value[component]
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return float(value)
    return None


def _sort_key(value: Any) -> tuple[int, Any]:
    numeric = _number(value)
    return (0, numeric) if numeric is not None else (1, str(value))


def _hierarchical_mean(
    rows: Sequence[Mapping[str, Any]], column: str, *, component: int | None = None
) -> tuple[float, float, int]:
    """Average optimizer→data→teacher and return a teacher-level 95% half-CI."""

    hierarchy: dict[Any, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        value = _number(row.get(column), component)
        if value is None:
            continue
        teacher = row.get("teacher_seed", row.get("replica", row.get("run_id", index)))
        data = row.get("data_seed", row.get("run_id", index))
        hierarchy[teacher][data].append(value)
    teacher_means = [
        float(np.mean([np.mean(values) for values in datasets.values() if values]))
        for datasets in hierarchy.values()
        if datasets
    ]
    if not teacher_means:
        return float("nan"), float("nan"), 0
    estimate = float(np.mean(teacher_means))
    half_interval = (
        float(1.96 * np.std(teacher_means, ddof=1) / np.sqrt(len(teacher_means)))
        if len(teacher_means) > 1
        else float("nan")
    )
    return estimate, half_interval, len(teacher_means)


def _series(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_column: str,
    value_column: str,
    group_column: str | None,
    component: int | None = None,
) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[Any, dict[float, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        control = _number(row.get(control_column))
        if control is None or _number(row.get(value_column), component) is None:
            continue
        group = row.get(group_column) if group_column else "all"
        grouped[group][control].append(row)
    result: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for group, controls in grouped.items():
        points = []
        for control, members in controls.items():
            mean, interval, n_teacher = _hierarchical_mean(
                members, value_column, component=component
            )
            points.append((control, mean, interval, n_teacher))
        points.sort(key=lambda item: item[0])
        result[group] = tuple(np.asarray(values) for values in zip(*points))  # type: ignore[assignment]
    return result


def save_figure(
    figure: Any,
    output_dir: str | Path,
    stem: str,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    close: bool = True,
) -> dict[str, Path]:
    """Save one figure in multiple formats under an explicit directory."""

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a simple filename component")
    normalized = tuple(str(value).lower().lstrip(".") for value in formats)
    if not normalized or any(value not in {"pdf", "png", "svg"} for value in normalized):
        raise ValueError("formats must be a non-empty subset of pdf/png/svg")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for file_format in dict.fromkeys(normalized):
        path = destination / f"{stem}.{file_format}"
        figure.savefig(path, format=file_format, dpi=dpi, bbox_inches="tight")
        paths[file_format] = path
    if close:
        _pyplot().close(figure)
    return paths


def plot_phase_map(
    data: AtlasAggregate | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    x_column: str = "spec.phase.semantic_mixture",
    y_column: str = "spec.phase.temperature",
    phase_column: str = "phase_label",
    eligible_only: bool = True,
    stem: str = "phase_map",
) -> FigureOutput:
    """Plot a categorical phase map, explicitly marking unavailable cells."""

    all_rows = _rows(data, "ensembles")
    selected = _eligible(all_rows, eligible_only)
    cells: dict[tuple[float, float], list[str]] = defaultdict(list)
    for row in selected:
        x, y = _number(row.get(x_column)), _number(row.get(y_column))
        if x is not None and y is not None:
            cells[(x, y)].append(str(row.get(phase_column, "unresolved")))
    expected_points = {
        (x, y)
        for row in all_rows
        if (x := _number(row.get(x_column))) is not None
        and (y := _number(row.get(y_column))) is not None
    }
    missing_points = sorted(expected_points - set(cells))
    if not cells and not missing_points:
        raise ValueError("no finite phase-map coordinates are available")
    consensus = {
        point: labels[0] if len(set(labels)) == 1 else "unresolved"
        for point, labels in cells.items()
    }
    palette = {
        "positional": _COLORS[0],
        "semantic": _COLORS[1],
        "coexistence": _COLORS[2],
        "disordered_or_unresolved": "#999999",
        "unresolved": "#666666",
    }
    plt = _pyplot()
    with plt.rc_context(_PAPER_RC):
        figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
        labels_present = sorted(set(consensus.values()))
        for index, label in enumerate(labels_present):
            points = np.asarray([point for point, value in consensus.items() if value == label])
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=42,
                marker=_MARKERS[index % len(_MARKERS)],
                color=palette.get(label, "#333333"),
                edgecolor="white",
                linewidth=0.45,
                label=label.replace("_", " "),
            )
        if missing_points:
            missing = np.asarray(missing_points)
            axis.scatter(
                missing[:, 0],
                missing[:, 1],
                marker="x",
                s=38,
                color="#222222",
                linewidth=1.0,
                label="missing / ineligible",
            )
        axis.set_xlabel(x_column.split(".")[-1].replace("_", " "))
        axis.set_ylabel(y_column.split(".")[-1].replace("_", " "))
        axis.set_title("Phase atlas")
        axis.legend(frameon=False, loc="best")
        axis.grid(alpha=0.2, linewidth=0.5, linestyle="--")
        paths = save_figure(figure, output_dir, stem)
    return FigureOutput(
        paths,
        {"n_cells": len(expected_points), "n_missing": len(missing_points), "labels": labels_present},
    )


def plot_order_parameters(
    data: AtlasAggregate | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    control_column: str = "spec.phase.semantic_mixture",
    group_column: str | None = "spec.phase.scaling.d_model",
    order_columns: Mapping[str, str] | None = None,
    eligible_only: bool = True,
    stem: str = "order_parameters",
) -> FigureOutput:
    """Plot functional order parameters with outer-teacher uncertainty."""

    rows = _eligible(_rows(data, "runs"), eligible_only)
    columns = order_columns or {
        "summary.functional_m_pos": r"$m_{pos}$",
        "summary.functional_m_sem": r"$m_{sem}$",
    }
    plt = _pyplot()
    n_series = 0
    with plt.rc_context(_PAPER_RC):
        figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
        for metric_index, (column, label) in enumerate(columns.items()):
            series = _series(
                rows,
                control_column=control_column,
                value_column=column,
                group_column=group_column,
            )
            for group_index, group in enumerate(sorted(series, key=_sort_key)):
                x, mean, interval, _ = series[group]
                color = _COLORS[group_index % len(_COLORS)]
                axis.plot(
                    x,
                    mean,
                    marker=_MARKERS[metric_index % len(_MARKERS)],
                    linestyle="-" if metric_index == 0 else "--",
                    color=color,
                    label=label if group_column is None else f"{label}, N={group}",
                )
                finite_error = np.where(np.isfinite(interval), interval, 0.0)
                axis.fill_between(x, mean - finite_error, mean + finite_error, color=color, alpha=0.12)
                n_series += 1
        if n_series == 0:
            plt.close(figure)
            raise ValueError("no finite order-parameter series are available")
        axis.axhline(0.0, color="#444444", linewidth=0.65, alpha=0.5)
        axis.set_xlabel(control_column.split(".")[-1].replace("_", " "))
        axis.set_ylabel("functional overlap")
        axis.set_title("Order parameters")
        axis.legend(frameon=False, ncol=2 if n_series > 4 else 1)
        axis.grid(alpha=0.2, linewidth=0.5, linestyle="--")
        paths = save_figure(figure, output_dir, stem)
    return FigureOutput(paths, {"n_series": n_series, "hierarchy": "optimizer→data→teacher"})


def plot_fluctuation_diagnostics(
    data: AtlasAggregate | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    control_column: str = "spec.phase.semantic_mixture",
    group_column: str | None = "spec.phase.scaling.d_model",
    susceptibility_column: str = "susceptibility_competition",
    binder_columns: Mapping[str, str] | None = None,
    eligible_only: bool = True,
    stem: str = "susceptibility_binder",
) -> FigureOutput:
    """Plot hierarchy-aware susceptibility and raw/centered Binder curves."""

    rows = _eligible(_rows(data, "ensembles"), eligible_only)
    binders = binder_columns or {
        "binder_competition_raw": "Binder (raw)",
        "binder_competition_centered": "Binder (centered)",
    }
    plt = _pyplot()
    n_series = 0
    with plt.rc_context(_PAPER_RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
        susceptibility_series = _series(
            rows,
            control_column=control_column,
            value_column=susceptibility_column,
            group_column=group_column,
        )
        for group_index, group in enumerate(sorted(susceptibility_series, key=_sort_key)):
            x, mean, interval, _ = susceptibility_series[group]
            axes[0].errorbar(
                x,
                mean,
                yerr=np.where(np.isfinite(interval), interval, 0.0),
                marker=_MARKERS[group_index % len(_MARKERS)],
                color=_COLORS[group_index % len(_COLORS)],
                capsize=2,
                label=f"N={group}" if group_column else "susceptibility",
            )
            n_series += 1
        for binder_index, (column, label) in enumerate(binders.items()):
            series = _series(
                rows,
                control_column=control_column,
                value_column=column,
                group_column=group_column,
            )
            for group_index, group in enumerate(sorted(series, key=_sort_key)):
                x, mean, _, _ = series[group]
                axes[1].plot(
                    x,
                    mean,
                    marker=_MARKERS[binder_index % len(_MARKERS)],
                    linestyle="-" if binder_index == 0 else "--",
                    color=_COLORS[group_index % len(_COLORS)],
                    label=f"{label}, N={group}" if group_column else label,
                )
                n_series += 1
        if n_series == 0:
            plt.close(figure)
            raise ValueError("no finite susceptibility or Binder series are available")
        axes[0].set_title("Susceptibility")
        axes[0].set_ylabel(r"$\chi$")
        axes[1].set_title("Binder cumulants")
        axes[1].set_ylabel(r"$U$")
        for axis in axes:
            axis.set_xlabel(control_column.split(".")[-1].replace("_", " "))
            axis.grid(alpha=0.2, linewidth=0.5, linestyle="--")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
        paths = save_figure(figure, output_dir, stem)
    return FigureOutput(paths, {"n_series": n_series, "fluctuation_unit": "teacher mean"})


def _complete_grid(
    rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    control_column: str,
    size_column: str,
    observable_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = sorted(
        {value for row in expected_rows if (value := _number(row.get(control_column))) is not None}
    )
    sizes = sorted(
        {value for row in expected_rows if (value := _number(row.get(size_column))) is not None}
    )
    if len(controls) < 2 or len(sizes) < 2:
        raise ValueError("finite-size collapse requires at least two controls and two sizes")
    curves = np.full((len(sizes), len(controls)), np.nan)
    missing: list[tuple[float, float]] = []
    for size_index, size in enumerate(sizes):
        for control_index, control in enumerate(controls):
            members = [
                row
                for row in rows
                if _number(row.get(size_column)) == size
                and _number(row.get(control_column)) == control
            ]
            mean, _, _ = _hierarchical_mean(members, observable_column)
            if np.isfinite(mean):
                curves[size_index, control_index] = mean
            else:
                missing.append((size, control))
    if missing:
        preview = ", ".join(f"(N={size:g}, g={control:g})" for size, control in missing[:5])
        raise ValueError(f"finite-size collapse requires a complete grid; missing {preview}")
    return np.asarray(controls), np.asarray(sizes), curves


def plot_finite_size_collapse(
    data: AtlasAggregate | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    critical_control: float,
    observable_exponent: float,
    inverse_nu: float,
    control_column: str = "spec.phase.semantic_mixture",
    size_column: str = "spec.phase.scaling.d_model",
    observable_column: str = "mean_m_sem",
    eligible_only: bool = True,
    n_grid: int = 128,
    stem: str = "fss_collapse",
) -> FigureOutput:
    """Plot raw finite-size curves and analysis-API collapse without extrapolation."""

    all_rows = _rows(data, "ensembles")
    selected = _eligible(all_rows, eligible_only)
    control, sizes, curves = _complete_grid(
        selected, all_rows, control_column, size_column, observable_column
    )
    collapse = data_collapse_score(
        control,
        sizes,
        curves,
        critical_control=critical_control,
        observable_exponent=observable_exponent,
        inverse_nu=inverse_nu,
        n_grid=n_grid,
    )
    plt = _pyplot()
    with plt.rc_context(_PAPER_RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
        for index, size in enumerate(sizes):
            color = _COLORS[index % len(_COLORS)]
            axes[0].plot(
                control,
                curves[index],
                marker=_MARKERS[index % len(_MARKERS)],
                color=color,
                label=f"N={size:g}",
            )
            axes[1].plot(
                collapse["scaled_grid"],
                collapse["collapsed_curves"][index],
                color=color,
                label=f"N={size:g}",
            )
        axes[0].axvline(critical_control, color="#222222", linestyle=":", linewidth=1.0)
        axes[0].set_xlabel(control_column.split(".")[-1].replace("_", " "))
        axes[0].set_ylabel(observable_column.split(".")[-1].replace("_", " "))
        axes[0].set_title("Finite-size curves")
        axes[1].set_xlabel(r"$(g-g_c)N^{1/\nu}$")
        axes[1].set_ylabel(r"$O N^y$")
        axes[1].set_title(f"Collapse score = {collapse['score']:.3g}")
        for axis in axes:
            axis.grid(alpha=0.2, linewidth=0.5, linestyle="--")
            axis.legend(frameon=False)
        paths = save_figure(figure, output_dir, stem)
    return FigureOutput(
        paths,
        {
            "score": float(collapse["score"]),
            "critical_control": float(critical_control),
            "observable_exponent": float(observable_exponent),
            "inverse_nu": float(inverse_nu),
            "n_sizes": int(sizes.size),
            "n_controls": int(control.size),
            "claim": "diagnostic_only" if sizes.size < 5 else "eligible_for_evidence_review",
        },
    )


def plot_spectral_specialization(
    data: AtlasAggregate | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    control_column: str = "spec.phase.semantic_mixture",
    group_column: str | None = "spec.phase.scaling.d_model",
    spectrum_columns: Mapping[str, str] | None = None,
    specialization_columns: Mapping[str, str] | None = None,
    top_k: int = 3,
    eligible_only: bool = True,
    stem: str = "architecture_ladder",
) -> FigureOutput:
    """Plot runner-compatible QK spectra and head-specialization statistics."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    rows = _eligible(_rows(data, "runs"), eligible_only)
    spectra = spectrum_columns or {
        "diagnostics.qk_singular_values_mean": "QK mean singular value",
        "summary.qk_spectral_norm_max": "QK spectral norm (max head)",
        "summary.qk_outlier_ratio_max": "QK outlier ratio (max head)",
    }
    specialization = specialization_columns or {
        "summary.specialization_strength": "specialization",
        "summary.specialization_entropy": "specialization entropy",
        "summary.effective_heads": "effective heads",
    }
    plt = _pyplot()
    n_spectral = 0
    n_specialization = 0
    with plt.rc_context(_PAPER_RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
        line_index = 0
        for column, label in spectra.items():
            values = [row.get(column) for row in rows if row.get(column) is not None]
            components: list[int | None] = (
                list(range(top_k))
                if any(isinstance(value, (list, tuple, np.ndarray)) for value in values)
                else [None]
            )
            for component in components:
                series = _series(
                    rows,
                    control_column=control_column,
                    value_column=column,
                    group_column=group_column,
                    component=component,
                )
                for group in sorted(series, key=_sort_key):
                    x, mean, _, _ = series[group]
                    component_label = label if component is None else f"{label} {component + 1}"
                    full_label = component_label if group_column is None else f"{component_label}, N={group}"
                    axes[0].plot(
                        x,
                        mean,
                        color=_COLORS[line_index % len(_COLORS)],
                        marker=_MARKERS[line_index % len(_MARKERS)],
                        label=full_label,
                    )
                    line_index += 1
                    n_spectral += 1
        line_index = 0
        for column, label in specialization.items():
            series = _series(
                rows,
                control_column=control_column,
                value_column=column,
                group_column=group_column,
            )
            for group in sorted(series, key=_sort_key):
                x, mean, _, _ = series[group]
                full_label = label if group_column is None else f"{label}, N={group}"
                axes[1].plot(
                    x,
                    mean,
                    color=_COLORS[line_index % len(_COLORS)],
                    marker=_MARKERS[line_index % len(_MARKERS)],
                    label=full_label,
                )
                line_index += 1
                n_specialization += 1
        if n_spectral + n_specialization == 0:
            plt.close(figure)
            raise ValueError("no finite spectral or specialization observables are available")
        axes[0].set_title("QK spectra")
        axes[0].set_ylabel("spectral observable")
        axes[1].set_title("Head specialization")
        axes[1].set_ylabel("specialization observable")
        for axis, count in zip(axes, (n_spectral, n_specialization)):
            axis.set_xlabel(control_column.split(".")[-1].replace("_", " "))
            axis.grid(alpha=0.2, linewidth=0.5, linestyle="--")
            if count:
                axis.legend(frameon=False, ncol=2 if count > 5 else 1)
            else:
                axis.text(0.5, 0.5, "not available", ha="center", va="center", transform=axis.transAxes)
        paths = save_figure(figure, output_dir, stem)
    return FigureOutput(
        paths,
        {"n_spectral_series": n_spectral, "n_specialization_series": n_specialization},
    )


def _decode_csv_value(value: str) -> Any:
    if value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_aggregate(source: str | Path | AtlasAggregate) -> AtlasAggregate:
    if isinstance(source, AtlasAggregate):
        return source
    path = Path(source)
    if path.is_dir():
        bundles = sorted(path.glob("*_aggregate.json"))
        candidates = bundles or sorted(path.glob("*_runs.json"))
        if not candidates:
            raise FileNotFoundError(f"no aggregate JSON found in {path}")
        path = candidates[0]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            runs = [
                {key: _decode_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        ensembles = build_ensemble_table(runs)
        return AtlasAggregate(runs, [], ensembles, [], {"source": str(path)})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        runs = payload
        ensembles = build_ensemble_table(runs)
        return AtlasAggregate(runs, [], ensembles, [], {"source": str(path)})
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("aggregate JSON must be a run list or an object containing runs")
    runs = payload["runs"]
    ensembles = payload.get("ensembles") or build_ensemble_table(runs)
    return AtlasAggregate(
        runs,
        payload.get("trajectories", []),
        ensembles,
        payload.get("claims", []),
        payload.get("metadata", {"source": str(path)}),
    )


def generate_paper_figures(
    aggregate: str | Path | AtlasAggregate,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render every supported paper figure and explicitly report missing data."""

    data = _load_aggregate(aggregate)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    def attempt(name: str, callback: Any) -> None:
        try:
            output = callback()
        except (ValueError, ImportError) as exc:
            skipped[name] = str(exc)
            return
        generated[name] = {
            "paths": {kind: str(path) for kind, path in output.paths.items()},
            "metadata": output.metadata,
        }

    attempt("phase_map", lambda: plot_phase_map(data, destination))
    attempt("order_parameters", lambda: plot_order_parameters(data, destination))
    attempt(
        "susceptibility_binder",
        lambda: plot_fluctuation_diagnostics(data, destination),
    )
    attempt(
        "spectral_specialization",
        lambda: plot_spectral_specialization(data, destination),
    )

    def finite_size() -> FigureOutput:
        eligible = _eligible(data.ensembles, True)
        controls, sizes, curves = _complete_grid(
            eligible,
            data.ensembles,
            "spec.phase.semantic_mixture",
            "spec.phase.scaling.d_model",
            "mean_m_sem",
        )
        critical_grid = np.unique(
            np.concatenate((controls, 0.5 * (controls[:-1] + controls[1:])))
        )
        fit = fit_finite_size_scaling_grid(
            controls,
            sizes,
            curves,
            critical_control_grid=critical_grid,
            observable_exponent_grid=np.linspace(0.0, 1.0, 9),
            inverse_nu_grid=np.linspace(0.25, 2.0, 8),
            n_grid=48,
        )
        return plot_finite_size_collapse(
            data,
            destination,
            critical_control=fit["critical_control"],
            observable_exponent=fit["observable_exponent"],
            inverse_nu=fit["inverse_nu"],
            n_grid=96,
        )

    attempt("finite_size_collapse", finite_size)
    report = {
        "generated": generated,
        "skipped": skipped,
        "n_generated": len(generated),
        "n_skipped": len(skipped),
        "aggregate_metadata": data.metadata,
    }
    manifest = destination / "paper_figures.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["manifest"] = str(manifest)
    return report


__all__ = [
    "FigureOutput",
    "generate_paper_figures",
    "plot_finite_size_collapse",
    "plot_fluctuation_diagnostics",
    "plot_order_parameters",
    "plot_phase_map",
    "plot_spectral_specialization",
    "save_figure",
]

