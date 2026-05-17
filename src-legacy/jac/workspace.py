"""Workspace discovery and layered environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


_LOADED_ENV_SOURCES: dict[str, Path] = {}
_LOADED_ENV_VALUES: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved workspace paths for a JAC invocation."""

    cwd: Path
    user_dir: Path
    project_root: Path | None
    project_dir: Path | None
    project_instruction_path: Path | None

    @property
    def user_env_path(self) -> Path:
        return self.user_dir / ".env"

    @property
    def project_env_path(self) -> Path | None:
        if self.project_dir is None:
            return None
        return self.project_dir / ".env.local"

    @property
    def state_db_path(self) -> Path:
        if self.project_dir is not None:
            return self.project_dir / "state.db"
        digest = sha256(self.cwd.resolve().as_posix().encode("utf-8")).hexdigest()[:16]
        return self.user_dir / "runs" / digest / "state.db"

    @property
    def mode_label(self) -> str:
        return "project" if self.project_root is not None else "no-project"


def default_user_dir() -> Path:
    """Return the user-global JAC configuration directory."""
    return Path(os.getenv("JAC_CONFIG_DIR", Path.home() / ".jac")).expanduser()


def discover_workspace(
    cwd: Path | None = None,
    *,
    user_dir: Path | None = None,
) -> Workspace:
    """Discover the project workspace, if any, from ``cwd`` upward."""
    resolved_cwd = (cwd or Path.cwd()).resolve()
    resolved_user_dir = (user_dir or default_user_dir()).expanduser()

    for directory in (resolved_cwd, *resolved_cwd.parents):
        project_dir = directory / ".agents"
        top_level_agents = directory / "AGENTS.md"
        if project_dir.is_dir():
            instruction_path = (
                top_level_agents
                if top_level_agents.is_file()
                else project_dir / "AGENTS.md"
            )
            return Workspace(
                cwd=resolved_cwd,
                user_dir=resolved_user_dir,
                project_root=directory,
                project_dir=project_dir,
                project_instruction_path=instruction_path,
            )
        if top_level_agents.is_file():
            return Workspace(
                cwd=resolved_cwd,
                user_dir=resolved_user_dir,
                project_root=directory,
                project_dir=project_dir,
                project_instruction_path=top_level_agents,
            )

    return Workspace(
        cwd=resolved_cwd,
        user_dir=resolved_user_dir,
        project_root=None,
        project_dir=None,
        project_instruction_path=None,
    )


def env_file_paths(workspace: Workspace) -> list[Path]:
    """Return dotenv files in increasing precedence order."""
    paths = [workspace.user_env_path]
    if workspace.project_env_path is not None:
        paths.append(workspace.project_env_path)
    return paths


def load_workspace_env(
    cwd: Path | None = None,
    *,
    user_dir: Path | None = None,
) -> dict[str, Path]:
    """Load JAC dotenv files without overriding inherited process env vars.

    The user-global file is loaded first and the project-local file second.
    Values inherited by the process stay authoritative.
    """
    workspace = discover_workspace(cwd, user_dir=user_dir)
    protected_keys = {
        key
        for key in os.environ
        if key not in _LOADED_ENV_VALUES or os.environ[key] != _LOADED_ENV_VALUES[key]
    }
    loaded: dict[str, Path] = {}

    for path in env_file_paths(workspace):
        for key, value in parse_dotenv(path).items():
            if key in protected_keys:
                continue
            os.environ[key] = value
            _LOADED_ENV_SOURCES[key] = path
            _LOADED_ENV_VALUES[key] = value
            loaded[key] = path

    return loaded


def env_source(name: str) -> str | None:
    """Return a non-secret source label for an environment variable."""
    if name in _LOADED_ENV_SOURCES and os.getenv(name):
        return str(_LOADED_ENV_SOURCES[name])
    if os.getenv(name):
        return "process"
    return None


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a small dotenv file.

    Supports ``KEY=value``, optional ``export``, comments, and simple quoted
    values. Multiline values are intentionally out of scope for JAC config.
    """
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_dotenv_value(value.strip())
    return values


def _strip_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
