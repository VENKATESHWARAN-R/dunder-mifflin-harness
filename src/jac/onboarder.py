"""Deterministic workspace onboarding helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jac.config import Settings, credential_candidates_for_model
from jac.workspace import Workspace, discover_workspace, env_source


DEFAULT_MODEL = "gateway/google-vertex:gemini-3.1-flash-lite-preview"
DEFAULT_APPROVAL_MODE = "interactive"
DEFAULT_WORKFLOW_MODE = "feature_by_feature"


@dataclass(frozen=True, slots=True)
class InitResult:
    """Files and directories touched by an init command."""

    workspace: Workspace
    created: list[Path]
    updated: list[Path]


def init_global_workspace(
    *,
    user_dir: Path,
    model: str = DEFAULT_MODEL,
    gateway_api_key: str | None = None,
) -> InitResult:
    """Create or update the user-global JAC workspace."""
    workspace = discover_workspace(user_dir=user_dir)
    created: list[Path] = []
    updated: list[Path] = []

    for directory in [
        user_dir,
        user_dir / "agents",
        user_dir / "skills",
        user_dir / "mcp",
        user_dir / "history",
    ]:
        if _ensure_dir(directory):
            created.append(directory)

    settings_path = user_dir / "settings.json"
    if _write_json_if_missing(settings_path, _default_settings()):
        created.append(settings_path)

    env_values = {
        "JAC_MODEL": model,
        "PYDANTIC_AI_GATEWAY_API_KEY": gateway_api_key or "",
    }
    _upsert_dotenv(user_dir / ".env", env_values, created=created, updated=updated)

    return InitResult(workspace=workspace, created=created, updated=updated)


def init_project_workspace(
    *,
    cwd: Path,
    create_env_local: bool = False,
    model: str = DEFAULT_MODEL,
) -> InitResult:
    """Create or update the project-local JAC workspace."""
    discovered = discover_workspace(cwd)
    project_root = discovered.project_root or cwd.resolve()
    project_dir = project_root / ".agents"
    created: list[Path] = []
    updated: list[Path] = []

    for directory in [
        project_dir,
        project_dir / "agents",
        project_dir / "skills",
        project_dir / "mcp",
    ]:
        if _ensure_dir(directory):
            created.append(directory)

    settings_path = project_dir / "settings.json"
    if _write_json_if_missing(settings_path, _default_settings()):
        created.append(settings_path)

    agents_path = project_dir / "AGENTS.md"
    if not agents_path.exists():
        agents_path.write_text("# Project Instructions\n", encoding="utf-8")
        created.append(agents_path)

    if create_env_local:
        _upsert_dotenv(
            project_dir / ".env.local",
            {"JAC_MODEL": model, "PYDANTIC_AI_GATEWAY_API_KEY": ""},
            created=created,
            updated=updated,
        )

    _ensure_gitignore_entries(
        project_root / ".gitignore",
        [
            ".agents/.env.local",
            ".agents/settings.local.json",
            ".agents/state.db",
            ".agents/logs/",
        ],
        created=created,
        updated=updated,
    )

    workspace = discover_workspace(project_root)
    return InitResult(workspace=workspace, created=created, updated=updated)


def doctor_report(settings: Settings, *, cwd: Path | None = None) -> str:
    """Return a non-secret diagnostic report for the current workspace."""
    workspace = discover_workspace(cwd)
    model = settings.model
    candidates = credential_candidates_for_model(model)
    lines = [
        "JAC Doctor",
        f"- mode: {workspace.mode_label}",
        f"- cwd: {workspace.cwd}",
        f"- user config: {workspace.user_dir}",
        f"- state db: {workspace.state_db_path}",
        f"- model: {model}",
    ]

    if workspace.project_root is not None:
        lines.append(f"- project root: {workspace.project_root}")
        lines.append(f"- project config: {workspace.project_dir}")
    else:
        lines.append("- project root: (none)")

    if not candidates:
        lines.append("- credentials: not required for selected model")
    else:
        for name in candidates:
            source = env_source(name)
            status = f"set via {source}" if source else "missing"
            lines.append(f"- credential {name}: {status}")

    return "\n".join(lines)


def _default_settings() -> dict[str, object]:
    return {
        "default_tier": "worker",
        "default_approval_mode": DEFAULT_APPROVAL_MODE,
        "default_workflow_mode": DEFAULT_WORKFLOW_MODE,
        "model_overrides": {
            "scout": "gateway/google-vertex:gemini-3.1-flash-lite-preview",
            "worker": "gateway/google-vertex:gemini-3.1-flash-lite-preview",
            "architect": "gateway/google-vertex:gemini-3.1-flash-lite-preview",
        },
        "telemetry": {"enabled": False},
    }


def _ensure_dir(path: Path) -> bool:
    if path.exists():
        return False
    path.mkdir(parents=True, exist_ok=True)
    return True


def _write_json_if_missing(path: Path, payload: dict[str, object]) -> bool:
    if path.exists():
        return False
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def _upsert_dotenv(
    path: Path,
    values: dict[str, str],
    *,
    created: list[Path],
    updated: list[Path],
) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    existing_keys = {
        line.split("=", 1)[0].strip()
        for line in existing_lines
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }

    new_lines = list(existing_lines)
    changed = False
    for key, value in values.items():
        if key in existing_keys:
            continue
        new_lines.append(f"{key}={value}")
        changed = True

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        created.append(path)
    elif changed:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        updated.append(path)


def _ensure_gitignore_entries(
    path: Path,
    entries: list[str],
    *,
    created: list[Path],
    updated: list[Path],
) -> None:
    existed = path.exists()
    existing = path.read_text(encoding="utf-8").splitlines() if existed else []
    missing = [entry for entry in entries if entry not in existing]
    if not missing:
        return

    new_lines = list(existing)
    if new_lines and new_lines[-1] != "":
        new_lines.append("")
    new_lines.extend(missing)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    (updated if existed else created).append(path)
