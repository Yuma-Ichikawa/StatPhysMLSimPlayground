"""Portable schemas for nested-disorder predictive experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    domain: str
    variant: str
    size: int
    control: float
    secondary: float
    seed: int
    inner_replicates: int
    stage: str
    holdout: bool
    parameters: dict[str, Any]
    task_id: str = ""

    def finalized(self) -> "Task":
        payload = asdict(self)
        payload.pop("task_id")
        digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        return Task(**payload, task_id=f"predictive-{self.domain}-{digest}-s{self.seed}")


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    study: str
    tasks: tuple[Task, ...]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "study": self.study,
            "tasks": [asdict(task) for task in self.tasks],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        payload = json.loads(Path(path).read_text())
        return cls(
            schema_version=str(payload["schema_version"]),
            study=str(payload["study"]),
            tasks=tuple(Task(**task) for task in payload["tasks"]),
        )
