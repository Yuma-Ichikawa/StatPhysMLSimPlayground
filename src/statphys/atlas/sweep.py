"""Declarative TOML sweeps and immutable JSONL run manifests."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import RunSpec, SeedPlan, run_spec_from_dict


def _set_dotted(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cursor = mapping
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"grid path {dotted!r} crosses non-table key {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def cartesian_specs(
    base: Mapping[str, Any],
    grid: Mapping[str, Iterable[Any]],
    *,
    replicas: int = 1,
    root_seed: int = 0,
) -> list[RunSpec]:
    """Expand a dotted-key Cartesian grid in deterministic lexical order."""

    if replicas < 1:
        raise ValueError("replicas must be positive")
    axes = sorted(grid)
    choices = [list(grid[name]) for name in axes]
    if any(not values for values in choices):
        raise ValueError("every sweep axis must contain at least one value")
    products = itertools.product(*choices) if axes else [()]
    result: list[RunSpec] = []
    seen: set[str] = set()
    for values in products:
        expanded = deepcopy(dict(base))
        for name, value in zip(axes, values):
            _set_dotted(expanded, name, value)
        base_spec = run_spec_from_dict(expanded)
        for replica in range(replicas):
            # A large odd stride avoids accidental overlap with small hand-set seeds.
            seeds = replace(base_spec.seeds, root=root_seed + replica * 1_000_003)
            spec = replace(base_spec, replica=replica, seeds=seeds)
            if spec.run_id not in seen:
                result.append(spec)
                seen.add(spec.run_id)
    return result


def load_sweep(path: str | Path) -> tuple[list[RunSpec], dict[str, Any]]:
    """Load and validate one portable atlas TOML file."""

    import tomllib

    source = Path(path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    meta = dict(raw.get("sweep", {}))
    base = dict(raw.get("base", {}))
    grid = dict(raw.get("grid", {}))
    replicas = int(meta.get("replicas", 1))
    root_seed = int(meta.get("root_seed", base.get("seeds", {}).get("root", 0)))
    specs = cartesian_specs(base, grid, replicas=replicas, root_seed=root_seed)
    metadata = {
        "source": str(source),
        "name": meta.get("name", source.stem),
        "description": meta.get("description", ""),
        "stage": meta.get("stage", "unspecified"),
        "replicas": replicas,
        "root_seed": root_seed,
        "n_runs": len(specs),
        "resources": dict(raw.get("resources", {})),
        "cluster": dict(raw.get("cluster", {})),
    }
    return specs, metadata


def write_manifest(path: str | Path, specs: Iterable[RunSpec], metadata: Mapping[str, Any]) -> Path:
    """Write a self-describing JSONL manifest (header followed by runs)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "metadata", **dict(metadata)}, sort_keys=True)]
    for index, spec in enumerate(specs):
        lines.append(
            json.dumps(
                {"kind": "run", "index": index, "run_id": spec.run_id, "spec": spec.to_dict()},
                sort_keys=True,
            )
        )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_manifest(path: str | Path) -> tuple[list[RunSpec], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    specs: list[RunSpec] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "metadata":
            metadata = {key: value for key, value in record.items() if key != "kind"}
        elif record.get("kind") == "run":
            spec = run_spec_from_dict(record["spec"])
            expected = record.get("run_id")
            if expected is not None and spec.run_id != expected:
                raise ValueError(f"content hash mismatch on manifest line {line_number}")
            specs.append(spec)
        else:
            raise ValueError(f"unknown manifest record on line {line_number}")
    return specs, metadata


def refine_numeric_axis(
    values: Iterable[float], scores: Iterable[float], *, top_intervals: int = 2
) -> list[float]:
    """Add midpoints around the strongest adjacent changes.

    This is intentionally a proposal function: new state points are written to
    a new manifest, so adaptive exploration cannot silently alter a registered
    confirmatory sweep.
    """

    points = sorted(zip(values, scores), key=lambda pair: pair[0])
    if len(points) < 2:
        return [float(point[0]) for point in points]
    intervals = sorted(
        (
            (abs(right[1] - left[1]), index, 0.5 * (left[0] + right[0]))
            for index, (left, right) in enumerate(zip(points[:-1], points[1:]))
        ),
        reverse=True,
    )
    additions = {midpoint for _, _, midpoint in intervals[:top_intervals]}
    return sorted({float(value) for value, _ in points} | additions)

