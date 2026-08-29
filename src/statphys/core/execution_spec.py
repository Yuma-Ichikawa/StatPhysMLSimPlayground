"""Execution attempts kept separate from scientific and protocol identities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any


def content_id(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


@dataclass(frozen=True)
class ExecutionSpec:
    """One hardware/software attempt of a registered scientific protocol."""

    device_type: str
    precision: str
    attempt: int = 1
    package_lock_hash: str | None = None
    container_digest: str | None = None
    source_revision: str | None = None
    deterministic_settings: Mapping[str, Any] = field(default_factory=dict)
    resource_request: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device_type.strip() or not self.precision.strip():
            raise ValueError("device_type and precision must be non-empty")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        object.__setattr__(
            self, "deterministic_settings", MappingProxyType(dict(self.deterministic_settings))
        )
        object.__setattr__(self, "resource_request", MappingProxyType(dict(self.resource_request)))

    @property
    def execution_id(self) -> str:
        return content_id(self.to_dict(), prefix="execution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "precision": self.precision,
            "attempt": self.attempt,
            "package_lock_hash": self.package_lock_hash,
            "container_digest": self.container_digest,
            "source_revision": self.source_revision,
            "deterministic_settings": dict(self.deterministic_settings),
            "resource_request": dict(self.resource_request),
        }


@dataclass(frozen=True)
class IdentityBundle:
    system_id: str
    protocol_id: str
    execution: ExecutionSpec

    @property
    def execution_id(self) -> str:
        return self.execution.execution_id


__all__ = ["ExecutionSpec", "IdentityBundle", "content_id"]
