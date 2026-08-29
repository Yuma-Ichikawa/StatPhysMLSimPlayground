#!/usr/bin/env python3
"""Generate the manuscript's conceptual and quantitative figure suite.

Every output has the exact physical size 6.4 by 4.8 inches.  The quantitative
figures are derived from the public strict aggregate and generated TeX macros;
conceptual figures encode the registered taxonomy and decision rules.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

FIGSIZE = (6.4, 4.8)
COLORS = {
    "navy": "#17324D",
    "blue": "#2F6B8A",
    "teal": "#2A9D8F",
    "gold": "#D9A441",
    "orange": "#E07A3F",
    "red": "#C64B4B",
    "purple": "#6C5B8E",
    "gray": "#68737D",
    "pale": "#EFF3F5",
}
SERIES_COLORS = [COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"]]
MARKERS = ["o", "s", "^", "D"]


@dataclass(frozen=True)
class Estimate:
    mean: float
    ci95: float
    raw: tuple[float, ...]


def apply_style() -> None:
    """Apply one restrained, color-blind-aware journal style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["navy"],
            "ytick.color": COLORS["navy"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#CAD2D8",
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "mathtext.fontset": "stixsans",
            "figure.figsize": FIGSIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    kwargs["figsize"] = FIGSIZE
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(*args, **kwargs)


def save(figure: Any, destination: Path) -> None:
    """Save exact-size PDF and PNG outputs without tight-bbox size drift."""

    figure.set_size_inches(*FIGSIZE, forward=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches=None, metadata={"Creator": "StatPhysMLSimPlayground"})
    figure.savefig(destination.with_suffix(".png"), bbox_inches=None, dpi=300)
    width, height = figure.get_size_inches()
    if not (math.isclose(width, FIGSIZE[0]) and math.isclose(height, FIGSIZE[1])):
        raise RuntimeError(f"unexpected figure size {(width, height)}")
    plt.close(figure)


def add_panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        color=COLORS["navy"],
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def box(
    axis: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    *,
    title_size: float = 8.0,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.09),
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["navy"],
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.27,
        textwrap.fill(detail, width=max(10, int(width * 90))),
        ha="center",
        va="center",
        fontsize=6.6,
        color=COLORS["gray"],
    )


def arrow(axis: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": COLORS["gray"], "lw": 0.9},
    )


def plot_atlas_taxonomy(destination: Path) -> None:
    """Show the seven scientific axes and their protocol/evidence wrappers."""

    figure, axis = new_figure(layout=None)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.96,
        "A three-layer atlas: scientific coordinates, protocol, and evidence",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        va="top",
    )
    axis.text(
        0.02,
        0.905,
        "Scientific coordinates define the object; protocol and evidence define the test and claim.",
        fontsize=7.4,
        color=COLORS["gray"],
        va="top",
    )

    scientific = [
        ("1  System $D$", "Transformer, diffusion, RL, agents", COLORS["blue"]),
        ("2  State $X$", "parameters, representations, trajectories", COLORS["teal"]),
        (r"3  Deformation $\delta$", "data, architecture, objective, dynamics", COLORS["gold"]),
        (r"4  Ensemble $\Omega$", "initialization, data, environment, replica", COLORS["orange"]),
        (
            r"5  Scale path $\mathbf{N}$",
            "width, depth, data, horizon, population",
            COLORS["purple"],
        ),
        ("6  Observable $m$", "estimator + units + null + ensemble", COLORS["red"]),
        ("7  Phenomenon $H$", "recovery, specialization, crossover, arrest", COLORS["navy"]),
    ]
    x_positions = np.linspace(0.02, 0.86, len(scientific))
    width = 0.12
    for index, (title, detail, color) in enumerate(scientific):
        box(
            axis,
            (float(x_positions[index]), 0.66),
            width,
            0.17,
            title,
            detail,
            color,
            title_size=7.2,
        )
        if index:
            arrow(
                axis,
                (float(x_positions[index - 1]) + width, 0.745),
                (float(x_positions[index]) - 0.005, 0.745),
            )

    axis.text(0.02, 0.59, "PROTOCOL", fontsize=7.2, fontweight="bold", color=COLORS["blue"])
    protocol = [
        ("Question", "estimand + falsifier"),
        ("Design", "controls + finite-size path"),
        ("Inference", "outer seeds + intervals"),
        ("Validation", "null + intervention + holdout"),
    ]
    for index, (title, detail) in enumerate(protocol):
        x = 0.08 + index * 0.23
        box(axis, (x, 0.41), 0.18, 0.12, title, detail, COLORS["blue"], title_size=7.6)
        if index:
            arrow(axis, (x - 0.05, 0.47), (x - 0.008, 0.47))

    axis.text(0.02, 0.34, "EVIDENCE", fontsize=7.2, fontweight="bold", color=COLORS["purple"])
    evidence = [
        ("Theory", "exact / asymptotic / phenomenological / empirical"),
        ("Fidelity", "solvable / synthetic / natural / deployed"),
        ("Support", "semantic / replication / scale / prediction / intervention"),
        ("Outcome", "supported / censored / unresolved / invalidated"),
    ]
    for index, (title, detail) in enumerate(evidence):
        x = 0.04 + index * 0.24
        box(axis, (x, 0.15), 0.21, 0.13, title, detail, COLORS["purple"], title_size=7.6)

    axis.text(
        0.5,
        0.055,
        "A large experiment does not become exact theory; an exact result does not become externally valid.",
        ha="center",
        fontsize=7.6,
        color=COLORS["navy"],
        fontweight="bold",
    )
    save(figure, destination)


def plot_outcome_taxonomy(destination: Path) -> None:
    """Group all registered outcomes by the scientific question they answer."""

    figure, axis = new_figure(layout=None)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.96,
        "Outcome taxonomy before phase language",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        va="top",
    )
    axis.text(
        0.02,
        0.90,
        "Transport, topology, dynamics, and evidential failure are separate, non-ordinal branches.",
        fontsize=7.5,
        color=COLORS["gray"],
        va="top",
    )
    groups = [
        (
            "Transport of an existing regime",
            ["preserved", "renormalized", "rounded", "censored"],
            COLORS["blue"],
            "Does the same ordered object survive the deformation?",
        ),
        (
            "Change in regime topology",
            ["split", "merged", "new regime"],
            COLORS["teal"],
            "Did the number or adjacency of regimes change?",
        ),
        (
            "History and accessibility",
            ["hysteretic", "path-dependent", "statistical--computational separation"],
            COLORS["orange"],
            "Does preparation or optimization determine what is reached?",
        ),
        (
            "Failure or insufficient evidence",
            ["semantic failure", "unresolved", "not comparable"],
            COLORS["red"],
            "Is the observable invalid, underpowered, or incommensurate?",
        ),
    ]
    y_values = [0.70, 0.51, 0.32, 0.13]
    for (title, outcomes, color, question), y in zip(groups, y_values, strict=True):
        box(axis, (0.03, y), 0.26, 0.13, title, question, color, title_size=7.4)
        arrow(axis, (0.30, y + 0.065), (0.35, y + 0.065))
        available = 0.61
        item_width = min(0.18, available / len(outcomes) - 0.012)
        gap = (available - len(outcomes) * item_width) / max(1, len(outcomes) - 1)
        for index, outcome in enumerate(outcomes):
            x = 0.36 + index * (item_width + gap)
            patch = FancyBboxPatch(
                (x, y + 0.027),
                item_width,
                0.076,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                linewidth=0.9,
                edgecolor=color,
                facecolor=mpl.colors.to_rgba(color, 0.11),
            )
            axis.add_patch(patch)
            axis.text(
                x + item_width / 2,
                y + 0.065,
                textwrap.fill(outcome, width=max(10, int(item_width * 82))),
                ha="center",
                va="center",
                fontsize=6.8,
                color=COLORS["navy"],
            )
    save(figure, destination)


def plot_observable_map(destination: Path) -> None:
    """Make estimator, ensemble, null, and interpretation inseparable."""

    figure, axis = new_figure(layout=None)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.96,
        "Observable dictionary: estimator + ensemble + null",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        va="top",
    )
    headers = [
        "Family",
        "Representative estimator",
        "Averaging ensemble",
        "Required control",
        "Allowed reading",
    ]
    x_edges = [0.02, 0.18, 0.42, 0.62, 0.80, 0.98]
    for index, header in enumerate(headers):
        axis.text(
            (x_edges[index] + x_edges[index + 1]) / 2,
            0.855,
            header,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white",
        )
        axis.add_patch(
            plt.Rectangle(
                (x_edges[index], 0.81),
                x_edges[index + 1] - x_edges[index],
                0.09,
                facecolor=COLORS["navy"],
                edgecolor="white",
                zorder=-1,
            )
        )
    rows = [
        (
            "Risk",
            "$R_{train},R_{IID},R_{OOD}$",
            "evaluation examples",
            "chance / calibrated baseline",
            "prediction quality",
        ),
        (
            "Recovery",
            "$I(m;Z)/H(Z)$",
            "outer seeds + examples",
            "shuffled latent labels",
            "semantic order",
        ),
        (
            "Specialization",
            "head / role assignment",
            "permutation-aligned seeds",
            "permuted components",
            "symmetry breaking",
        ),
        (
            "Response",
            r"$\partial\langle m\rangle/\partial g$",
            "independent outer seeds",
            "window + scale sensitivity",
            "crossover candidate",
        ),
        (
            "Fluctuation",
            r"$N\,\mathrm{Var}_{\Omega}(m)$",
            "outer disorder only",
            "temporal variance",
            "susceptibility",
        ),
        (
            "Dynamics",
            "time-to-order, path entropy",
            "training trajectories",
            "matched compute",
            "accessibility / arrest",
        ),
        (
            "Mechanism",
            "$R_{ablated}-R_{full}$",
            "matched interventions",
            "equal-resource sham",
            "causal contribution",
        ),
    ]
    row_height = 0.095
    for row_index, row in enumerate(rows):
        y = 0.81 - (row_index + 1) * row_height
        fill = "#F4F7F8" if row_index % 2 == 0 else "white"
        axis.add_patch(
            plt.Rectangle((0.02, y), 0.96, row_height, facecolor=fill, edgecolor="#D7DEE2", lw=0.5)
        )
        for column_index, value in enumerate(row):
            display_value = value
            font_size = 6.8
            if column_index >= 2:
                display_value = textwrap.fill(
                    value,
                    width=max(
                        8,
                        int((x_edges[column_index + 1] - x_edges[column_index]) * 68),
                    ),
                )
            elif column_index == 1:
                font_size = 5.9
            axis.text(
                (x_edges[column_index] + x_edges[column_index + 1]) / 2,
                y + row_height / 2,
                display_value,
                ha="center",
                va="center",
                fontsize=font_size,
                color=COLORS["navy"] if column_index == 0 else COLORS["gray"],
                fontweight="bold" if column_index == 0 else "normal",
            )
    axis.text(
        0.02,
        0.055,
        "Rule: without ensemble, units, validity range, and null, an estimator is not a registered observable.",
        fontsize=7.4,
        color=COLORS["red"],
        fontweight="bold",
    )
    save(figure, destination)


def plot_phase_decision(destination: Path) -> None:
    """Render the finite-size decision rule and its negative outcomes."""

    figure, axis = new_figure(layout=None)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.96,
        "Finite-size claim logic: a steep learning curve is not sufficient",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        va="top",
    )
    questions = [
        (
            "1",
            "Semantic validity",
            "Does the order track a registered latent or functional property?",
        ),
        ("2", "Independent ensemble", "Are fluctuations computed over independent outer disorder?"),
        (
            "3",
            "Finite-size signature",
            "Is there an interior crossing, sharpening, drift, or collapse?",
        ),
        ("4", "Robustness", "Does the conclusion survive windows, minimum size, and alternatives?"),
        ("5", "Prospective check", "Does a frozen model predict an untouched size or deformation?"),
    ]
    y_values = [0.78, 0.63, 0.48, 0.33, 0.18]
    negative = [
        "semantic failure",
        "invalid ensemble",
        "censored / rounded",
        "unresolved",
        "model misspecification",
    ]
    for index, ((number, title, detail), y) in enumerate(zip(questions, y_values, strict=True)):
        box(
            axis,
            (0.08, y),
            0.46,
            0.11,
            f"{number}. {title}",
            detail,
            COLORS["blue"],
            title_size=7.8,
        )
        if index:
            arrow(axis, (0.31, y + 0.15), (0.31, y + 0.115))
        axis.text(
            0.565,
            y + 0.055,
            "NO",
            fontsize=6.5,
            color=COLORS["red"],
            fontweight="bold",
            va="center",
        )
        arrow(axis, (0.59, y + 0.055), (0.65, y + 0.055))
        box(
            axis,
            (0.66, y + 0.012),
            0.26,
            0.086,
            negative[index],
            "registered negative outcome",
            COLORS["red"],
            title_size=7.2,
        )
    axis.text(
        0.27, 0.105, "YES", fontsize=6.5, color=COLORS["teal"], fontweight="bold", ha="center"
    )
    arrow(axis, (0.31, 0.18), (0.31, 0.11))
    box(
        axis,
        (0.08, 0.025),
        0.46,
        0.075,
        "Bounded phase-language claim",
        "domain-specific; no automatic universality claim",
        COLORS["teal"],
        title_size=7.8,
    )
    save(figure, destination)


def load_reference(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("reference aggregate must contain non-empty records")
    return sorted(records, key=lambda row: (int(row["size"]), float(row["control"])))


def estimate(row: dict[str, Any], name: str) -> Estimate:
    metric = row["metrics"][name]
    return Estimate(
        mean=float(metric["mean"]),
        ci95=float(metric["ci95"]),
        raw=tuple(float(value) for value in metric["raw_outer_values"]),
    )


def grouped(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for size in sorted({int(row["size"]) for row in records}):
        output[size] = sorted(
            (row for row in records if int(row["size"]) == size),
            key=lambda row: float(row["control"]),
        )
    return output


def line_metric(
    axis: Any,
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    *,
    raw: bool = False,
    reference: float | None = None,
) -> None:
    for series_index, (size, rows) in enumerate(grouped(records).items()):
        x = np.asarray([float(row["control"]) for row in rows])
        estimates = [estimate(row, metric) for row in rows]
        y = np.asarray([item.mean for item in estimates])
        error = np.asarray([item.ci95 for item in estimates])
        color = SERIES_COLORS[series_index]
        if raw:
            for row, item in zip(rows, estimates, strict=True):
                jitter = np.linspace(-0.045, 0.045, len(item.raw))
                axis.scatter(
                    float(row["control"]) + jitter,
                    item.raw,
                    s=7,
                    color=color,
                    alpha=0.22,
                    linewidths=0,
                    zorder=1,
                )
        axis.errorbar(
            x,
            y,
            yerr=error,
            color=color,
            marker=MARKERS[series_index],
            lw=1.4,
            ms=4.2,
            capsize=2.3,
            label=rf"$d={size}$",
            zorder=3,
        )
    if reference is not None:
        axis.axhline(reference, color=COLORS["gray"], ls=":", lw=1.0)
    axis.set_xlabel(r"sample coefficient $\alpha$")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log", base=2)
    axis.set_xticks([0.5, 2, 8], ["0.5", "2", "8"])
    axis.grid(True)


def plot_reference_response(records: list[dict[str, Any]], destination: Path) -> None:
    """Report order, risk, generalization, and OOD behavior with seed points."""

    figure, axes = new_figure(2, 2)
    line_metric(axes[0, 0], records, "semantic_order", r"semantic order $m$", raw=True)
    line_metric(axes[0, 1], records, "normalized_test_risk", r"IID risk $R_{IID}$", raw=True)
    line_metric(
        axes[1, 0],
        records,
        "normalized_generalization_gap",
        r"gap $R_{IID}-R_{train}$",
        raw=True,
        reference=0,
    )
    for series_index, (size, rows) in enumerate(grouped(records).items()):
        x = np.asarray([float(row["control"]) for row in rows])
        iid = [estimate(row, "normalized_test_risk") for row in rows]
        ood = [estimate(row, "normalized_ood_risk") for row in rows]
        y = np.asarray([right.mean - left.mean for left, right in zip(iid, ood, strict=True)])
        conservative_error = np.asarray(
            [left.ci95 + right.ci95 for left, right in zip(iid, ood, strict=True)]
        )
        axes[1, 1].errorbar(
            x,
            y,
            yerr=conservative_error,
            color=SERIES_COLORS[series_index],
            marker=MARKERS[series_index],
            lw=1.4,
            ms=4.2,
            capsize=2.3,
        )
    axes[1, 1].axhline(0, color=COLORS["gray"], ls=":", lw=1.0)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks([0.5, 2, 8], ["0.5", "2", "8"])
    axes[1, 1].set_xlabel(r"sample coefficient $\alpha$")
    axes[1, 1].set_ylabel(r"OOD penalty $R_{OOD}-R_{IID}$")
    axes[1, 1].grid(True)
    titles = [
        "Learning response",
        "Held-out risk",
        "Data-limited generalization",
        "Distribution-shift check",
    ]
    for label, axis, title in zip("ABCD", axes.flat, titles, strict=True):
        add_panel_label(axis, label)
        axis.set_title(title, loc="left", color=COLORS["navy"], fontweight="bold")
    handles = [
        Line2D([0], [0], color=SERIES_COLORS[index], marker=MARKERS[index], label=rf"$d={size}$")
        for index, size in enumerate(sorted(grouped(records)))
    ]
    axes[0, 0].legend(handles=handles, loc="lower right", ncol=1, frameon=False)
    figure.suptitle(
        "GPU reference: 45 runs with 95% outer-seed intervals and raw seeds",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["navy"],
    )
    save(figure, destination)


def plot_reference_mechanisms(records: list[dict[str, Any]], destination: Path) -> None:
    """Separate causal interventions from finite-size phase diagnostics."""

    figure, axes = new_figure(2, 2)
    for series_index, (size, rows) in enumerate(grouped(records).items()):
        x = np.asarray([float(row["control"]) for row in rows])
        for metric, style, label in (
            ("attention_causal_effect", "--", "attention"),
            ("mlp_causal_effect", "-", "MLP"),
        ):
            values = [estimate(row, metric) for row in rows]
            axes[0, 0].errorbar(
                x,
                [item.mean for item in values],
                yerr=[item.ci95 for item in values],
                color=SERIES_COLORS[series_index],
                marker=MARKERS[series_index],
                ls=style,
                lw=1.25,
                ms=3.8,
                capsize=2,
                label=label if series_index == 0 else None,
            )
    line_metric(
        axes[0, 1],
        records,
        "attention_entropy",
        r"attention entropy $H_A$",
        reference=math.log(2),
    )
    line_metric(axes[1, 0], records, "susceptibility", r"outer-seed response $\chi$", raw=True)
    line_metric(
        axes[1, 1],
        records,
        "binder_cumulant",
        r"Binder cumulant $U_4$",
        reference=2 / 3,
    )
    axes[0, 0].axhline(0, color=COLORS["gray"], ls=":", lw=1.0)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks([0.5, 2, 8], ["0.5", "2", "8"])
    axes[0, 0].set_xlabel(r"sample coefficient $\alpha$")
    axes[0, 0].set_ylabel(r"bounded risk effect $\eta$")
    axes[0, 0].grid(True)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    size_handles = [
        Line2D([0], [0], color=SERIES_COLORS[index], marker=MARKERS[index], label=rf"$d={size}$")
        for index, size in enumerate(sorted(grouped(records)))
    ]
    axes[0, 1].legend(handles=size_handles, frameon=False, ncol=1, loc="lower right")
    titles = [
        "Matched module interventions",
        r"No attention specialization ($\log 2$ reference)",
        "No interior susceptibility peak",
        r"No Binder crossing ($2/3$ reference)",
    ]
    for label, axis, title in zip("ABCD", axes.flat, titles, strict=True):
        add_panel_label(axis, label)
        axis.set_title(title, loc="left", color=COLORS["navy"], fontweight="bold")
    figure.suptitle(
        "Mechanism and phase diagnostics answer different questions",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["navy"],
    )
    save(figure, destination)


def plot_reference_verdict(records: list[dict[str, Any]], destination: Path) -> None:
    """Audit control identifiability and summarize the evidence verdict."""

    figure, axes = new_figure(1, 2, width_ratios=[1.08, 0.92])
    left, right = axes
    sizes = sorted(grouped(records))
    controls = sorted({float(row["control"]) for row in records})
    examples = np.zeros((len(sizes), len(controls)))
    order = np.zeros_like(examples)
    for row in records:
        i = sizes.index(int(row["size"]))
        j = controls.index(float(row["control"]))
        examples[i, j] = estimate(row, "train_examples").mean
        order[i, j] = estimate(row, "semantic_order").mean
    image = left.imshow(order, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.2, order.max()))
    for i in range(len(sizes)):
        for j in range(len(controls)):
            left.text(
                j,
                i,
                f"m={order[i, j]:.3f}\nn={examples[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if order[i, j] > 0.11 else COLORS["navy"],
            )
    left.set_xticks(range(len(controls)), [f"{value:g}" for value in controls])
    left.set_yticks(range(len(sizes)), [str(value) for value in sizes])
    left.set_xlabel(r"requested control $\alpha$")
    left.set_ylabel(r"width $d$")
    left.set_title("Effective control grid", loc="left", fontweight="bold", color=COLORS["navy"])
    add_panel_label(left, "A")
    figure.colorbar(image, ax=left, shrink=0.72, label="semantic order")
    left.add_patch(
        plt.Rectangle((-0.48, -0.48), 1.0, 0.96, fill=False, edgecolor=COLORS["red"], lw=1.8)
    )
    left.text(
        0.02,
        -0.23,
        "duplicated effective point",
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["red"],
        fontweight="bold",
    )

    right.axis("off")
    right.set_title(
        "Evidence vector and decision", loc="left", fontweight="bold", color=COLORS["navy"]
    )
    add_panel_label(right, "B")
    checks = [
        ("Semantic probe", "available", COLORS["teal"]),
        ("Outer-seed ensemble", "five independent", COLORS["teal"]),
        ("Effective controls", "one collapsed pair", COLORS["gold"]),
        ("Interior response peak", "absent", COLORS["red"]),
        ("Binder crossing", "absent", COLORS["red"]),
        ("Attention specialization", "absent", COLORS["red"]),
        ("Largest-size holdout", "not registered", COLORS["gold"]),
    ]
    for index, (check, value, color) in enumerate(checks):
        y = 0.84 - index * 0.10
        right.add_patch(plt.Circle((0.07, y), 0.018, color=color))
        right.text(0.12, y + 0.014, check, va="center", fontsize=7.4, color=COLORS["navy"])
        right.text(0.12, y - 0.022, value, va="center", fontsize=6.6, color=COLORS["gray"])
    box(
        right,
        (0.07, 0.035),
        0.86,
        0.12,
        "Verdict: finite-width learning response",
        "Evidence grade C; no phase boundary or universality claim",
        COLORS["red"],
        title_size=8.0,
    )
    save(figure, destination)


_ESTIMATE_PATTERN = re.compile(
    r"\\providecommand\{\\(?P<name>[A-Za-z]+)\}\{\$(?P<mean>-?[0-9.]+)" r"\\pm (?P<ci>[0-9.]+)\$\}"
)


def load_confirmation_macros(path: Path) -> dict[str, Estimate]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("name"): Estimate(float(match.group("mean")), float(match.group("ci")), ())
        for match in _ESTIMATE_PATTERN.finditer(text)
    }


def bars(
    axis: Any,
    labels: list[str],
    values: list[Estimate],
    ylabel: str,
    *,
    colors: list[str] | None = None,
) -> None:
    x = np.arange(len(labels))
    palette = colors or [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(labels))]
    axis.bar(
        x,
        [value.mean for value in values],
        yerr=[value.ci95 for value in values],
        color=palette,
        edgecolor="white",
        linewidth=0.6,
        capsize=2.5,
        error_kw={"lw": 0.9},
    )
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y")
    axis.set_axisbelow(True)


def plot_confirmation_summary(macros: dict[str, Estimate], destination: Path) -> None:
    """Synthesize the frozen 12-seed confirmation without dense legends."""

    required = [
        "ConfirmMLPLinearOrder",
        "ConfirmMLPGELUOrder",
        "ConfirmMLPGEGLUOrder",
        "ConfirmMLPSwiGLUOrder",
        "ConfirmMLPLinearCausalEffect",
        "ConfirmMLPGELUCausalEffect",
        "ConfirmMLPGEGLUCausalEffect",
        "ConfirmMLPSwiGLUCausalEffect",
        "ConfirmSGDMError",
        "ConfirmAdamWError",
        "ConfirmMuonError",
        "ConfirmSOAPError",
        "ConfirmLinearDataScalingError",
        "ConfirmThreeHalfDataScalingError",
        "ConfirmQuadraticDataScalingError",
        "ConfirmTinyStoriesBitsPerByte",
        "ConfirmSimpleStoriesBitsPerByte",
        "ConfirmFineWebEduBitsPerByte",
        "ConfirmDolmaBitsPerByte",
    ]
    missing = [name for name in required if name not in macros]
    if missing:
        raise ValueError(f"missing confirmation macros: {', '.join(missing)}")
    figure, axes = new_figure(2, 2)
    mlp_labels = ["Linear", "GELU", "GEGLU", "SwiGLU"]
    x = np.arange(len(mlp_labels))
    width = 0.36
    order_values = [
        macros[f"ConfirmMLP{name}Order"] for name in ("Linear", "GELU", "GEGLU", "SwiGLU")
    ]
    causal_values = [
        macros[f"ConfirmMLP{name}CausalEffect"] for name in ("Linear", "GELU", "GEGLU", "SwiGLU")
    ]
    axes[0, 0].bar(
        x - width / 2,
        [value.mean for value in order_values],
        width,
        yerr=[value.ci95 for value in order_values],
        label="task order",
        color=COLORS["blue"],
        capsize=2,
    )
    axes[0, 0].bar(
        x + width / 2,
        [value.mean for value in causal_values],
        width,
        yerr=[value.ci95 for value in causal_values],
        label="MLP effect",
        color=COLORS["orange"],
        capsize=2,
    )
    axes[0, 0].set_xticks(x, mlp_labels, rotation=18, ha="right")
    axes[0, 0].set_ylabel("dimensionless estimate")
    axes[0, 0].grid(True, axis="y")
    axes[0, 0].legend(frameon=False, ncol=2)
    bars(
        axes[0, 1],
        ["SGD-M", "AdamW", "Muon", "SOAP"],
        [
            macros[name]
            for name in (
                "ConfirmSGDMError",
                "ConfirmAdamWError",
                "ConfirmMuonError",
                "ConfirmSOAPError",
            )
        ],
        r"normalized IID risk $R_{IID}$",
    )
    bars(
        axes[1, 0],
        [r"$n\propto d$", r"$n\propto d^{3/2}$", r"$n\propto d^2$"],
        [
            macros["ConfirmLinearDataScalingError"],
            macros["ConfirmThreeHalfDataScalingError"],
            macros["ConfirmQuadraticDataScalingError"],
        ],
        r"normalized IID risk $R_{IID}$",
    )
    bars(
        axes[1, 1],
        ["TinyStories", "SimpleStories", "FineWeb-Edu", "Dolma"],
        [
            macros["ConfirmTinyStoriesBitsPerByte"],
            macros["ConfirmSimpleStoriesBitsPerByte"],
            macros["ConfirmFineWebEduBitsPerByte"],
            macros["ConfirmDolmaBitsPerByte"],
        ],
        "test bits per byte",
    )
    titles = [
        "Feed-forward mechanism",
        "Optimizer geometry",
        "Data scaling path",
        "Natural-data endpoints",
    ]
    for label, axis, title in zip("ABCD", axes.flat, titles, strict=True):
        add_panel_label(axis, label)
        axis.set_title(title, loc="left", fontweight="bold", color=COLORS["navy"])
    figure.suptitle(
        "Frozen 12-seed confirmations: separate estimands, common visual grammar",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["navy"],
    )
    save(figure, destination)


def plot_coverage_map(destination: Path) -> None:
    """Distinguish implementation coverage from evidential depth."""

    figure, axis = new_figure(layout=None)
    figure.subplots_adjust(left=0.18, right=0.82, bottom=0.22, top=0.82)
    domains = ["Transformer", "Diffusion", "Reinforcement", "Multi-agent"]
    columns = [
        "semantics",
        "finite size",
        "mechanism",
        "dynamics",
        "holdout",
        "natural",
    ]
    levels = np.asarray(
        [
            [3, 2, 3, 3, 1, 2],
            [2, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 0],
            [2, 1, 1, 1, 1, 0],
        ]
    )
    cmap = mpl.colors.ListedColormap(["#ECEFF1", "#F2D6A2", "#9FC8CB", "#376F87"])
    image = axis.imshow(levels, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    labels = {
        0: "open",
        1: "model\nvalidation",
        2: "five-seed\nmeasurement",
        3: "frozen\nconfirmation",
    }
    for i in range(levels.shape[0]):
        for j in range(levels.shape[1]):
            level = int(levels[i, j])
            axis.text(
                j,
                i,
                labels[level],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if level == 3 else COLORS["navy"],
                fontweight="bold" if level in (0, 3) else "normal",
            )
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(domains)), domains)
    axis.tick_params(axis="x", labelrotation=25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.tick_params(length=0)
    axis.set_title(
        "Coverage depth is domain specific",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=COLORS["navy"],
        pad=16,
    )
    cbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3], shrink=0.74, pad=0.03)
    cbar.ax.set_yticklabels(["open", "model validation", "measurement", "confirmation"])
    axis.text(
        0,
        -0.20,
        "Artifact status is neither a performance score nor evidence of shared universality.",
        transform=axis.transAxes,
        fontsize=7.4,
        color=COLORS["red"],
        fontweight="bold",
    )
    save(figure, destination)


def write_reference_macros(records: list[dict[str, Any]], destination: Path) -> None:
    by_key = {(int(row["size"]), float(row["control"])): row for row in records}

    def formatted(size: int, control: float, metric: str) -> str:
        value = estimate(by_key[(size, control)], metric)
        return f"${value.mean:.3f}\\pm {value.ci95:.3f}$"

    duplicate = (
        estimate(by_key[(16, 0.5)], "train_examples").mean
        == estimate(by_key[(16, 2.0)], "train_examples").mean
    )
    lines = [
        "% Generated from the strict public GPU reference aggregate; do not edit.",
        "\\providecommand{\\ReferenceRuns}{45}",
        "\\providecommand{\\ReferenceSeeds}{5}",
        "\\providecommand{\\ReferenceSizes}{3}",
        "\\providecommand{\\ReferenceControls}{3}",
        f"\\providecommand{{\\ReferenceLargestOrder}}{{{formatted(64, 8.0, 'semantic_order')}}}",
        f"\\providecommand{{\\ReferenceLargestRisk}}{{{formatted(64, 8.0, 'normalized_test_risk')}}}",
        f"\\providecommand{{\\ReferenceLowDataGap}}{{{formatted(64, 0.5, 'normalized_generalization_gap')}}}",
        f"\\providecommand{{\\ReferenceHighDataGap}}{{{formatted(64, 8.0, 'normalized_generalization_gap')}}}",
        f"\\providecommand{{\\ReferenceLargestMLPEffect}}{{{formatted(64, 0.5, 'mlp_causal_effect')}}}",
        f"\\providecommand{{\\ReferenceLargestAttentionEffect}}{{{formatted(64, 8.0, 'attention_causal_effect')}}}",
        f"\\providecommand{{\\ReferenceCollapsedControl}}{{{'1' if duplicate else '0'}}}",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="strict GPU aggregate JSON")
    parser.add_argument(
        "--confirmation-macros",
        required=True,
        type=Path,
        help="generated frozen-confirmation TeX macros",
    )
    parser.add_argument("--output", required=True, type=Path, help="figure output directory")
    parser.add_argument("--macros", required=True, type=Path, help="reference-result TeX output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    records = load_reference(args.reference)
    confirmation = load_confirmation_macros(args.confirmation_macros)
    figures = {
        "figure1_atlas_taxonomy.pdf": lambda path: plot_atlas_taxonomy(path),
        "figure2_outcome_taxonomy.pdf": lambda path: plot_outcome_taxonomy(path),
        "figure3_observable_map.pdf": lambda path: plot_observable_map(path),
        "figure4_phase_decision.pdf": lambda path: plot_phase_decision(path),
        "figure5_reference_response.pdf": lambda path: plot_reference_response(records, path),
        "figure6_reference_mechanisms.pdf": lambda path: plot_reference_mechanisms(records, path),
        "figure7_reference_verdict.pdf": lambda path: plot_reference_verdict(records, path),
        "figure8_confirmation_summary.pdf": lambda path: plot_confirmation_summary(
            confirmation, path
        ),
        "figure9_coverage_map.pdf": lambda path: plot_coverage_map(path),
    }
    for name, callback in figures.items():
        callback(args.output / name)
    write_reference_macros(records, args.macros)
    manifest = {
        "figure_size_inches": list(FIGSIZE),
        "figures": sorted(figures),
        "reference_conditions": len(records),
    }
    (args.output / "generated_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
