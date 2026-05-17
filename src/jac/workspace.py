"""Workspace discovery — where JAC reads user-scoped files from.

M1 Slice B scope: resolve the user-global directory (`~/.jac/` by default,
overridable via `JAC_HOME`) and expose the canonical path for
`settings.json`. Project-scope (`<repo>/.agents/`) discovery lands when the
runtime/CLI slice needs it.

This module imports nothing from `jac.*` — see the dependency-direction
matrix in `docs/reference/PHILOSOPHY.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved user-scope JAC paths."""

    user_dir: Path

    @property
    def settings_path(self) -> Path:
        return self.user_dir / "settings.json"


def default_user_dir() -> Path:
    """Resolve the user-global JAC directory, honouring `JAC_HOME`."""
    override = os.environ.get("JAC_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".jac"


def discover_workspace() -> Workspace:
    """Return the resolved workspace. The directory may not yet exist."""
    return Workspace(user_dir=default_user_dir())
