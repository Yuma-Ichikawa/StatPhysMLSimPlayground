"""Solvable-to-realistic phase-continuation experiments.

Every paper condition is represented by exactly five independent seeds and
every domain emits the same scalar phase card plus domain-specific arrays.
"""

from .schema import (
    Domain,
    Manifest,
    REQUIRED_SEED_COUNT,
    TaskSpec,
    expand_config,
    read_manifest,
    write_manifest,
)

__all__ = [
    "Domain",
    "Manifest",
    "REQUIRED_SEED_COUNT",
    "TaskSpec",
    "expand_config",
    "read_manifest",
    "write_manifest",
]
