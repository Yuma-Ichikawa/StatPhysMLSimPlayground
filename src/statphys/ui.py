"""Guided, portable entry points and self-contained scientific reports."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


STUDY_KINDS = {
    "order_parameters": "Replica sweep with uncertainty-aware order parameters",
    "phase_diagram": "Two-control sweep with finite-size visualisation",
    "online": "Online linear learning compared with its ODE",
    "replica": "Regularized linear regression compared with a replica solver",
    "ready_made": "A named exploratory study from the built-in catalogue",
}


def catalog() -> list[dict[str, str]]:
    """Return stable user-facing workflows rather than implementation namespaces."""
    return [{"kind": name, "description": description} for name, description in STUDY_KINDS.items()]


def study_template(kind: str = "order_parameters") -> str:
    if kind not in STUDY_KINDS:
        raise ValueError(f"unknown study kind: {kind}")
    return (
        "# A portable StatPhys study. Paths are relative to where you run it.\n"
        "[study]\n"
        'name = "my_statphys_study"\n'
        f'kind = "{kind}"\n'
        'preset = "random_mlp"\n'
        "alphas = [0.5, 1.0, 2.0, 4.0]\n"
        "replicas = 4\n"
        "seed = 0\n\n"
        "[evidence]\n"
        'tier = "exploratory" # exploratory, confirmatory, or finite_size\n'
        'claim = "response shift"\n\n'
        "[output]\n"
        'directory = "statphys_results"\n'
    )


def write_study_template(path: str | Path, kind: str = "order_parameters") -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing study file: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(study_template(kind), encoding="utf-8")
    return destination


def load_study(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("study TOML must contain a table")
    return payload


def validate_study(path: str | Path) -> dict[str, Any]:
    """Validate the public study schema without importing ML dependencies."""
    payload = load_study(path)
    study = payload.get("study")
    if not isinstance(study, dict):
        raise ValueError("study.toml requires a [study] table")
    kind = study.get("kind")
    if kind not in STUDY_KINDS:
        choices = ", ".join(sorted(STUDY_KINDS))
        raise ValueError(f"study.kind must be one of: {choices}")
    name = study.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("study.name must be a non-empty string")
    if kind in {"order_parameters", "phase_diagram"}:
        alphas = study.get("alphas")
        if (
            not isinstance(alphas, list)
            or not alphas
            or not all(float(value) > 0 for value in alphas)
        ):
            raise ValueError("study.alphas must contain one or more positive values")
    replicas = study.get("replicas", 4)
    if not isinstance(replicas, int) or replicas < 2:
        raise ValueError("study.replicas must be an integer of at least two")
    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("[evidence] must be a table when supplied")
    tier = evidence.get("tier", "exploratory")
    if tier not in {"exploratory", "confirmatory", "finite_size"}:
        raise ValueError("evidence.tier must be exploratory, confirmatory, or finite_size")
    return {
        "valid": True,
        "kind": kind,
        "name": name,
        "evidence_tier": tier,
        "allowed_wording": "phase transition" if tier == "finite_size" else "response shift",
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.name
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _summary(values: list[float]) -> dict[str, float]:
    """Return a portable mean and normal-approximation CI from outer seeds."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("an observable has no finite outer-seed values")
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return {"mean": mean, "ci95": 0.0}
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return {"mean": mean, "ci95": 1.96 * math.sqrt(variance / len(finite))}


def _condition_table(result: Mapping[str, Any], variant: str) -> list[dict[str, Any]]:
    """Normalize built-in result layouts to the public condition-table schema."""
    records = result.get("records")
    controls = result.get("x_values")
    if isinstance(records, dict) and isinstance(controls, list):
        conditions: list[dict[str, Any]] = []
        for index, control in enumerate(controls):
            metrics: dict[str, dict[str, float]] = {}
            for name, replicas in records.items():
                if not isinstance(replicas, list):
                    continue
                values = [
                    float(replica[index])
                    for replica in replicas
                    if isinstance(replica, list)
                    and index < len(replica)
                    and isinstance(replica[index], (int, float))
                ]
                if values:
                    metrics[str(name)] = _summary(values)
            if metrics:
                conditions.append(
                    {
                        "domain": "teacher_student",
                        "variant": variant,
                        "series": "independent seeds",
                        "control": float(control),
                        "metrics": metrics,
                    }
                )
        return conditions

    grids = result.get("grids")
    alphas = result.get("alphas")
    parameter_values = result.get("param_values")
    parameter_name = str(result.get("param_name", "control"))
    if (
        not isinstance(grids, dict)
        or not isinstance(alphas, list)
        or not isinstance(parameter_values, list)
    ):
        return []
    conditions = []
    for row, parameter in enumerate(parameter_values):
        for column, alpha in enumerate(alphas):
            metrics = {
                str(name): {"mean": float(grid[row][column]), "ci95": 0.0}
                for name, grid in grids.items()
                if isinstance(grid, list)
                and row < len(grid)
                and isinstance(grid[row], list)
                and column < len(grid[row])
                and isinstance(grid[row][column], (int, float))
            }
            if metrics:
                conditions.append(
                    {
                        "domain": "teacher_student",
                        "variant": variant,
                        "series": f"{parameter_name}={parameter}",
                        "control": float(alpha),
                        "metrics": metrics,
                    }
                )
    return conditions


def _portable_result(result: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Add a condition table and seed count without discarding raw records."""
    portable = dict(result)
    conditions = _condition_table(portable, variant)
    if conditions:
        portable["conditions"] = conditions
    records = portable.get("records")
    if isinstance(records, dict) and records:
        first: Any = next(iter(records.values()), [])
        if isinstance(first, list):
            portable["outer_seed_count"] = len(first)
    return portable


def run_study(path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Run a validated study and save a portable JSON result artifact."""
    validation = validate_study(path)
    source = Path(path)
    payload = load_study(source)
    study = payload["study"]
    output = payload.get("output", {})
    destination_root = Path(output_dir or output.get("directory", "statphys_results"))
    destination_root.mkdir(parents=True, exist_ok=True)
    snapshot = destination_root / "study.toml"
    source_text = source.read_text(encoding="utf-8")
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != source_text:
        raise FileExistsError("output directory already belongs to a different study")
    if not snapshot.exists():
        snapshot.write_text(source_text, encoding="utf-8")
    kind = validation["kind"]
    alphas = [float(value) for value in study.get("alphas", [0.5, 1.0, 2.0, 4.0])]
    replicas = int(study.get("replicas", 4))
    preset = str(study.get("preset", "random_mlp"))

    import statphys

    if kind == "order_parameters":
        result = statphys.quick_order_parameters(
            preset, alphas=alphas, n_replicas=replicas, plot=False, verbose=False
        )
        scientific_result = _portable_result(result.to_dict(), preset)
    elif kind == "phase_diagram":
        parameter = str(study.get("parameter", "sparsity"))
        controls = [float(value) for value in study.get("controls", [0.25, 0.5, 0.75])]
        result = statphys.quick_phase_diagram(
            preset,
            parameter,
            controls,
            alphas=alphas,
            n_replicas=replicas,
            plot=False,
            verbose=False,
        )
        scientific_result = _portable_result(result.to_dict(), preset)
    elif kind == "online":
        result = statphys.quick_online(n_seeds=replicas, plot=False, verbose=False)
        scientific_result = _portable_result(result.to_dict(), "online")
    elif kind == "replica":
        result = statphys.quick_replica(n_seeds=replicas, alphas=alphas, plot=False, verbose=False)
        scientific_result = _portable_result(result.to_dict(), "replica")
    else:
        from statphys.experiment.studies import run_study as run_ready_made

        run_ready_made(str(study.get("name", preset)), out_dir=destination_root, quick=True)
        scientific_result = {"status": "completed", "kind": "ready_made"}

    artifact = {
        "schema_version": "2.0",
        "study": {
            "name": validation["name"],
            "kind": kind,
            "evidence_tier": validation["evidence_tier"],
            "allowed_wording": validation["allowed_wording"],
        },
        "result": scientific_result,
    }
    destination = destination_root / "result.json"
    destination.write_text(
        json.dumps(artifact, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    return {
        "schema_version": payload.get("schema_version", "unknown"),
        "top_level_keys": sorted(result) if isinstance(result, dict) else [],
        "study": payload.get("study", {}),
        "has_conditions": isinstance(result, dict) and bool(result.get("conditions")),
        "has_dynamics": isinstance(result, dict) and bool(result.get("dynamics")),
    }


def _metric_means(payload: Mapping[str, Any]) -> dict[str, float]:
    result = payload.get("result", payload)
    conditions = result.get("conditions", []) if isinstance(result, dict) else []
    if not conditions and isinstance(result, dict):
        conditions = _condition_table(result, str(result.get("variant", "result")))
    values: dict[str, list[float]] = {}
    for condition in conditions:
        for name, summary in condition.get("metrics", {}).items():
            if isinstance(summary, dict) and isinstance(summary.get("mean"), (int, float)):
                values.setdefault(name, []).append(float(summary["mean"]))
    return {name: float(sum(items) / len(items)) for name, items in values.items() if items}


def compare_artifacts(left: str | Path, right: str | Path) -> dict[str, Any]:
    first = json.loads(Path(left).read_text(encoding="utf-8"))
    second = json.loads(Path(right).read_text(encoding="utf-8"))
    left_means, right_means = _metric_means(first), _metric_means(second)
    names = sorted(set(left_means) | set(right_means))
    return {
        "metrics": [
            {
                "metric": name,
                "left_mean": left_means.get(name),
                "right_mean": right_means.get(name),
                "difference": (
                    None
                    if name not in left_means or name not in right_means
                    else right_means[name] - left_means[name]
                ),
            }
            for name in names
        ]
    }


def _condition_series(
    payload: Mapping[str, Any],
) -> tuple[str | None, dict[str, list[tuple[float, float, float]]]]:
    result = payload.get("result", payload)
    conditions = result.get("conditions", []) if isinstance(result, dict) else []
    if not conditions and isinstance(result, dict):
        conditions = _condition_table(result, str(result.get("variant", "result")))
    if not conditions:
        return None, {}
    preferred = (
        "semantic_order",
        "signed_order",
        "order_parameter",
        "m_hat",
        "generalization_error",
    )
    available = set().union(*(set(row.get("metrics", {})) for row in conditions))
    metric = next((name for name in preferred if name in available), next(iter(available), None))
    if metric is None:
        return None, {}
    series: dict[str, list[tuple[float, float, float]]] = {}
    for row in conditions:
        summary = row.get("metrics", {}).get(metric)
        if not isinstance(summary, dict) or not isinstance(summary.get("mean"), (int, float)):
            continue
        error = float(summary.get("ci95", 0.0))
        if "ci95_low" in summary and "ci95_high" in summary:
            error = 0.5 * abs(float(summary["ci95_high"]) - float(summary["ci95_low"]))
        label = str(row.get("series", f"size={row.get('size', 'all')}"))
        series.setdefault(label, []).append(
            (float(row.get("control", 0.0)), float(summary["mean"]), error)
        )
    return metric, {name: sorted(rows) for name, rows in series.items()}


def _chart_svg(metric: str | None, series: Mapping[str, list[tuple[float, float, float]]]) -> str:
    if not series:
        return '<div class="empty">No condition-level curve is available in this artifact.</div>'
    values = [value for rows in series.values() for point in rows for value in point[:2]]
    xs = values[::2]
    ys = values[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_pad = max((x1 - x0) * 0.08, 1e-6)
    y_pad = max((y1 - y0) * 0.12, 1e-6)
    x0, x1, y0, y1 = x0 - x_pad, x1 + x_pad, y0 - y_pad, y1 + y_pad
    width, height, left, bottom = 760, 330, 58, 282
    colors = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#0891b2")

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            left + (x - x0) / (x1 - x0) * (width - left - 24),
            bottom - (y - y0) / (y1 - y0) * (bottom - 24),
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(metric or "observable")} curve">',
        f'<line x1="{left}" y1="24" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - 24}" y2="{bottom}" class="axis"/>',
        f'<text x="{width / 2:.1f}" y="322" class="label">control parameter</text>',
        f'<text x="16" y="{height / 2:.1f}" class="label" transform="rotate(-90 16 {height / 2:.1f})">{html.escape(metric or "observable")}</text>',
    ]
    for index, (label, rows) in enumerate(series.items()):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{point(x, y)[0]:.2f},{point(x, y)[1]:.2f}" for x, y, _ in rows)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>'
        )
        for x, y, error in rows:
            px, py = point(x, y)
            _, low = point(x, y - error)
            _, high = point(x, y + error)
            parts.append(
                f'<line x1="{px:.2f}" y1="{low:.2f}" x2="{px:.2f}" y2="{high:.2f}" stroke="{color}"/>'
            )
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="{color}"/>')
        parts.append(
            f'<text x="{width - 190}" y="{44 + index * 20}" fill="{color}" class="legend">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_report(artifact_path: str | Path, output_path: str | Path) -> Path:
    """Render a self-contained HTML evidence dashboard with no local URLs."""
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    metric, series = _condition_series(payload)
    study = payload.get("study", {}) if isinstance(payload, dict) else {}
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    seed_count = (
        result.get("outer_seed_count", result.get("seed_count", "not recorded"))
        if isinstance(result, dict)
        else "not recorded"
    )
    status = (
        result.get("theory_status", result.get("status", "unclassified"))
        if isinstance(result, dict)
        else "unclassified"
    )
    wording = (
        study.get("allowed_wording", "response shift")
        if isinstance(study, dict)
        else "response shift"
    )
    chart = _chart_svg(metric, series)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>StatPhys analysis report</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#f6f8fb;color:#172033;margin:0}} main{{max-width:900px;margin:auto;padding:28px 18px 44px}} h1{{margin:0}} .sub{{color:#526071}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}} .card{{background:white;border:1px solid #dce3ed;border-radius:10px;padding:14px;box-shadow:0 1px 3px #1720330d}} .label{{font-size:12px;fill:#526071}} .legend{{font-size:13px}} .axis{{stroke:#637083;stroke-width:1}} svg{{width:100%;height:auto;background:white;border:1px solid #dce3ed;border-radius:10px}} .empty{{padding:42px;background:white;border:1px dashed #aab5c4;border-radius:10px;color:#526071}} code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}</style></head>
<body><main><h1>StatPhys Analysis Report</h1><p class="sub">Interactive-ready, self-contained evidence view. It contains no local-server address or source path.</p>
<section class="grid"><div class="card"><strong>Study</strong><br>{html.escape(str(study.get("name", "unnamed")))}</div><div class="card"><strong>Outer seeds</strong><br>{html.escape(str(seed_count))}</div><div class="card"><strong>Theory status</strong><br>{html.escape(str(status))}</div><div class="card"><strong>Allowed wording</strong><br>{html.escape(str(wording))}</div></section>
<section><h2>Phase explorer</h2><p class="sub">Curve points show reported means; vertical bars show recorded or seed-derived 95% uncertainty where available.</p>{chart}</section>
<section class="card"><h2>Evidence panel</h2><p>Observable: <code>{html.escape(metric or "not available")}</code>. Statistical quantities from independent seed ensembles must be labelled separately from checkpoint-time variation. Unclassified theory is rendered as provisional evidence and should not be described as an exact result.</p></section>
</main></body></html>"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def doctor() -> dict[str, Any]:
    """Check optional capabilities without exposing host-specific details."""
    checks: dict[str, bool] = {}
    for name in ("numpy", "torch", "scipy", "matplotlib", "pandas"):
        try:
            __import__(name)
            checks[name] = True
        except ImportError:
            checks[name] = False
    return {"ok": all(checks.values()), "dependencies": checks}
