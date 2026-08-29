"""Validate relative documentation links and portable public documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_LOOPBACK = "(?:local" + "host|127\\.0\\.0\\.1)"
_LOCAL_ADDRESS = re.compile(r"(?i)(?:https?://)?" + _LOOPBACK + r"(?::\d+)?")
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])/" + "(?:home|mnt|tmp|var)/")
_FILE_URL = "file" + "://"


def _is_external(destination: str) -> bool:
    return destination.startswith(("#", "http://", "https://", "mailto:"))


def validate_documentation(root: Path) -> list[str]:
    """Return all broken-link and portability violations in public Markdown."""
    errors: list[str] = []
    for document in sorted((root / "docs").rglob("*.md")):
        text = document.read_text(encoding="utf-8")
        if _LOCAL_ADDRESS.search(text) or _FILE_URL in text or _LOCAL_PATH.search(text):
            errors.append(
                f"{document.relative_to(root)} contains a local address or absolute host path"
            )
        for raw_destination in _LINK.findall(text):
            destination = unquote(raw_destination.split(maxsplit=1)[0]).split("#", maxsplit=1)[0]
            if not destination or _is_external(destination):
                continue
            candidate = document.parent / destination
            if not candidate.is_file():
                errors.append(f"{document.relative_to(root)} links to missing {destination!r}")
    return errors


def main() -> int:
    """Validate project documentation from any current working directory."""
    root = Path(__file__).resolve().parents[1]
    errors = validate_documentation(root)
    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    count = len(list((root / "docs").rglob("*.md")))
    print(f"Validated {count} Markdown documents: relative links and portability checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
