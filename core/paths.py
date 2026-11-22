"""Deterministic path validation. Every file the agent touches goes through here."""
from __future__ import annotations

import re
from pathlib import Path


class PathValidationError(ValueError):
    pass


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_resolve(base_dir: Path | str, relative: Path | str) -> Path:
    """Resolve ``relative`` inside ``base_dir`` and refuse anything that escapes it.

    Rejects absolute paths, ``..`` traversal and symlinks pointing outside the base.
    """
    base = Path(base_dir).resolve()
    rel = Path(relative)
    if rel.is_absolute():
        raise PathValidationError(f"Absolute paths are not allowed: {relative}")
    if any(part == ".." for part in rel.parts):
        raise PathValidationError(f"Path traversal is not allowed: {relative}")
    candidate = (base / rel).resolve()
    if candidate != base and base not in candidate.parents:
        raise PathValidationError(f"Path escapes the allowed directory: {relative}")
    return candidate


def safe_filename(name: str) -> str:
    """Validate a single file name (no directories, no hidden files)."""
    if not _SAFE_NAME.match(name or ""):
        raise PathValidationError(f"Unsafe file name: {name!r}")
    return name
