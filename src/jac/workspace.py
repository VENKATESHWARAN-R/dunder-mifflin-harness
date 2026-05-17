"""Workspace discovery — where JAC reads user-scoped and project-scoped files.

Resolves:
  - user scope:    `~/.jac/` (override via `$JAC_HOME`)
  - project scope: the nearest ancestor containing `.agents/`, or failing
    that the nearest git root (`.git/`). `project_dir` is then
    `project_root / ".agents"` whether or not that directory exists yet.

Project discovery is bounded — walk up the cwd's parents until a marker is
found or the filesystem root is hit. No `pyproject.toml` fallback; we keep
the lookup small and predictable until a real use case demands more.

This module imports nothing from `jac.*` — see the dependency-direction
matrix in `docs/reference/PHILOSOPHY.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved JAC workspace paths.

    `project_root` is the directory that holds either `.agents/` or `.git/`;
    `project_dir` is the `.agents` subdirectory underneath it (the on-disk
    home for project-scope personas, skills, MCP server stubs). Both are
    `None` when run outside any recognisable project.
    """

    user_dir: Path
    project_root: Path | None = None

    @property
    def settings_path(self) -> Path:
        return self.user_dir / "settings.json"

    @property
    def project_dir(self) -> Path | None:
        return self.project_root / ".agents" if self.project_root else None


def default_user_dir() -> Path:
    """Resolve the user-global JAC directory, honouring `JAC_HOME`."""
    override = os.environ.get("JAC_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".jac"


def discover_project_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default cwd) looking for `.agents/` or `.git/`.

    Returns the first ancestor that has either marker, or `None` if neither
    is found before the filesystem root.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".agents").is_dir() or (candidate / ".git").exists():
            return candidate
    return None


def discover_workspace(start: Path | None = None) -> Workspace:
    """Return the resolved workspace. Directories may not yet exist."""
    return Workspace(
        user_dir=default_user_dir(),
        project_root=discover_project_root(start),
    )
