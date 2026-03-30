"""
Configuration system with layered loading.

Load order (later overrides earlier):
1. Built-in defaults (Pydantic field defaults)
2. User-level TOML: ~/.pygemini/settings.toml
3. Project-level TOML: .pygemini/settings.toml
4. Environment variables (GEMINI_API_KEY, PYGEMINI_MODEL, etc.)
5. CLI flag overrides (passed as kwargs)
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ApprovalMode = Literal["interactive", "auto_edit", "yolo"]
SandboxMode = Literal["none", "docker"]

# ---------------------------------------------------------------------------
# Default list of core tool names (all built-in tools)
# ---------------------------------------------------------------------------

DEFAULT_CORE_TOOLS: list[str] = [
    "read_file",
    "read_many_files",
    "write_file",
    "edit_file",
    "list_directory",
    "run_shell_command",
    "web_fetch",
    "google_web_search",
    "save_memory",
    "ask_user",
]

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class HookConfig(BaseModel):
    """Configuration for a lifecycle hook command."""

    command: str
    timeout: int = 10


# ---------------------------------------------------------------------------
# Main Config model
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """
    Pydantic settings model. Mirrors Gemini CLI's settings.json structure.

    Load order (later overrides earlier):
    1. Default settings (built-in)
    2. ~/.pygemini/settings.toml (user-level)
    3. .pygemini/settings.toml (project-level)
    4. Environment variables (GEMINI_API_KEY, PYGEMINI_MODEL, etc.)
    5. CLI flags (--model, --sandbox, etc.)
    """

    # API
    api_key: str | None = None
    model: str = "gemini-3.1-flash-lite-preview"
    fallback_model: str = "gemini-2.5-flash"

    # Tools
    core_tools: list[str] = list(DEFAULT_CORE_TOOLS)
    allowed_tools: list[str] = []
    excluded_tools: list[str] = []

    # Safety
    approval_mode: ApprovalMode = "interactive"
    sandbox: SandboxMode = "none"

    # Context
    gemini_md_paths: list[str] = []
    include_directories: list[str] = []

    # Session
    checkpointing_enabled: bool = False

    # MCP
    mcp_servers: dict[str, MCPServerConfig] = {}

    # Hooks
    hooks_enabled: bool = True
    hooks: dict[str, list[HookConfig]] = {}

    # UI
    theme: str = "default"

    @property
    def config_dir(self) -> Path:
        """Return the pygemini config directory, respecting PYGEMINI_HOME."""
        return Path(os.environ.get("PYGEMINI_HOME", "~/.pygemini")).expanduser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_config_dir(config: Config | None = None) -> Path:
    """Create the config directory if it doesn't exist and return its path."""
    if config is not None:
        path = config.config_dir
    else:
        path = Path(os.environ.get("PYGEMINI_HOME", "~/.pygemini")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict, or {} if missing."""
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _env_overrides() -> dict[str, Any]:
    """Collect config values from recognised environment variables."""
    overrides: dict[str, Any] = {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is not None:
        overrides["api_key"] = api_key

    model = os.environ.get("PYGEMINI_MODEL")
    if model is not None:
        overrides["model"] = model

    sandbox = os.environ.get("PYGEMINI_SANDBOX")
    if sandbox is not None:
        overrides["sandbox"] = sandbox

    approval_mode = os.environ.get("PYGEMINI_APPROVAL_MODE")
    if approval_mode is not None:
        overrides["approval_mode"] = approval_mode

    return overrides


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(**cli_overrides: Any) -> Config:
    """
    Build a Config with layered loading.

    Order (later wins):
    1. Pydantic defaults
    2. User-level TOML (~/.pygemini/settings.toml)
    3. Project-level TOML (.pygemini/settings.toml)
    4. Environment variables
    5. CLI keyword overrides
    """
    config_home = Path(
        os.environ.get("PYGEMINI_HOME", "~/.pygemini")
    ).expanduser()

    # Layer 2 — user-level TOML
    merged: dict[str, Any] = _load_toml(config_home / "settings.toml")

    # Layer 3 — project-level TOML
    project_toml = _load_toml(Path(".pygemini/settings.toml"))
    merged.update(project_toml)

    # Layer 4 — environment variables
    merged.update(_env_overrides())

    # Layer 5 — CLI overrides (drop None values so they don't clobber)
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value

    return Config(**merged)
