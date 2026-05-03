"""Deterministic workspace onboarding helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jac.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TIER,
    Settings,
    credential_candidates_for_model,
    default_model_tiers,
    profile_env_name,
    provider_definition,
)
from jac.state.seeder import check_workspace_files
from jac.workspace import Workspace, discover_workspace, env_source


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
    provider: str = DEFAULT_PROVIDER,
    model_tiers: dict[str, list[str]] | None = None,
    env_values: dict[str, str] | None = None,
    model: str = DEFAULT_MODEL,
    gateway_api_key: str | None = None,
    profile: str = "default",
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
    selected_tiers = model_tiers or default_model_tiers(provider)
    _write_json(
        settings_path,
        _default_settings(
            provider=provider,
            model_tiers=selected_tiers,
            active_profile=profile,
        ),
        created=created,
        updated=updated,
    )

    selected_env = env_values or _default_env_values(provider)
    selected_env.setdefault("JAC_MODEL", _first_worker_model(selected_tiers) or model)
    selected_env.update(_profile_env_values(profile, provider, selected_env))
    if gateway_api_key is not None:
        selected_env["PYDANTIC_AI_GATEWAY_API_KEY"] = gateway_api_key
        selected_env[profile_env_name(profile, "PYDANTIC_AI_GATEWAY_API_KEY")] = (
            gateway_api_key
        )
    _upsert_dotenv(user_dir / ".env", selected_env, created=created, updated=updated)

    return InitResult(workspace=workspace, created=created, updated=updated)


def configure_global_profile(
    *,
    user_dir: Path,
    profile: str,
    provider: str,
    model_tiers: dict[str, list[str]] | None = None,
    env_values: dict[str, str] | None = None,
    activate: bool = False,
) -> InitResult:
    """Create or update a named user-global model profile."""
    workspace = discover_workspace(user_dir=user_dir)
    created: list[Path] = []
    updated: list[Path] = []

    if _ensure_dir(user_dir):
        created.append(user_dir)

    selected_tiers = model_tiers or default_model_tiers(provider)
    settings_path = user_dir / "settings.json"
    payload = _read_json_object(settings_path)
    profiles = _object_dict(payload.get("profiles"))
    profiles[profile] = _profile_settings(provider=provider, model_tiers=selected_tiers)
    payload["profiles"] = profiles
    if activate:
        payload["active_profile"] = profile
        payload.update(_profile_settings(provider=provider, model_tiers=selected_tiers))
    elif "active_profile" not in payload:
        payload["active_profile"] = profile
        payload.update(_profile_settings(provider=provider, model_tiers=selected_tiers))

    _write_json(settings_path, payload, created=created, updated=updated)

    selected_env = env_values or _default_env_values(provider)
    _upsert_dotenv(
        user_dir / ".env",
        _profile_env_values(profile, provider, selected_env),
        created=created,
        updated=updated,
    )

    return InitResult(workspace=workspace, created=created, updated=updated)


def activate_global_profile(*, user_dir: Path, profile: str) -> InitResult:
    """Make an existing user-global model profile active."""
    workspace = discover_workspace(user_dir=user_dir)
    created: list[Path] = []
    updated: list[Path] = []
    settings_path = user_dir / "settings.json"
    payload = _read_json_object(settings_path)
    profiles = _object_dict(payload.get("profiles"))
    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' is not configured.")
    selected_raw = profiles[profile]
    if not isinstance(selected_raw, dict):
        raise ValueError(f"Profile '{profile}' is invalid.")
    selected = _object_dict(selected_raw)

    payload["active_profile"] = profile
    payload.update(
        {
            key: selected[key]
            for key in (
                "default_provider",
                "default_tier",
                "model_tiers",
                "model_overrides",
            )
            if key in selected
        }
    )
    _write_json(settings_path, payload, created=created, updated=updated)
    return InitResult(workspace=workspace, created=created, updated=updated)


def init_project_workspace(
    *,
    cwd: Path,
    create_env_local: bool = False,
    provider: str = DEFAULT_PROVIDER,
    model_tiers: dict[str, list[str]] | None = None,
    env_values: dict[str, str] | None = None,
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
    selected_tiers = model_tiers or default_model_tiers(provider)
    _write_json(
        settings_path,
        _default_settings(provider=provider, model_tiers=selected_tiers),
        created=created,
        updated=updated,
    )

    agents_path = project_dir / "AGENTS.md"
    if not agents_path.exists():
        agents_path.write_text("# Project Instructions\n", encoding="utf-8")
        created.append(agents_path)

    if create_env_local:
        selected_env = env_values or _default_env_values(provider)
        selected_env.setdefault(
            "JAC_MODEL", _first_worker_model(selected_tiers) or model
        )
        _upsert_dotenv(
            project_dir / ".env.local",
            selected_env,
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
    selection = settings.resolve_model_selection()
    model = selection.model_ref
    candidates = credential_candidates_for_model(model, provider=selection.provider)
    lines = [
        "JAC Doctor",
        f"- mode: {workspace.mode_label}",
        f"- cwd: {workspace.cwd}",
        f"- user config: {workspace.user_dir}",
        f"- state db: {workspace.state_db_path}",
        f"- active profile: {settings.active_profile or '(none)'}",
        f"- provider: {selection.provider}",
        f"- default tier: {settings.default_tier}",
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
            source = _first_env_source(settings.candidate_env_names(name))
            status = f"set via {source}" if source else "missing"
            lines.append(f"- credential {name}: {status}")

    for item in provider_definition(selection.provider).env:
        if item.required or item.name in candidates:
            continue
        source = _first_env_source(settings.candidate_env_names(item.name))
        status = (
            f"set via {source}"
            if source
            else "using default"
            if item.default
            else "not set"
        )
        lines.append(f"- optional {item.name}: {status}")

    for warning in check_workspace_files(workspace):
        lines.append(f"- warning: {warning}")

    return "\n".join(lines)


def _first_env_source(names: tuple[str, ...]) -> str | None:
    for name in names:
        if source := env_source(name):
            return source
    return None


def _default_settings(
    *,
    provider: str = DEFAULT_PROVIDER,
    model_tiers: dict[str, list[str]] | None = None,
    active_profile: str = "default",
) -> dict[str, object]:
    selected_tiers = model_tiers or default_model_tiers(provider)
    profile_settings = _profile_settings(provider=provider, model_tiers=selected_tiers)
    return {
        "active_profile": active_profile,
        "profiles": {active_profile: profile_settings},
        "default_provider": provider,
        "default_tier": "worker",
        "default_approval_mode": DEFAULT_APPROVAL_MODE,
        "default_workflow_mode": DEFAULT_WORKFLOW_MODE,
        "model_tiers": selected_tiers,
        "model_overrides": _legacy_model_overrides(selected_tiers),
        "telemetry": {"enabled": False},
    }


def _ensure_dir(path: Path) -> bool:
    if path.exists():
        return False
    path.mkdir(parents=True, exist_ok=True)
    return True


def _write_json(
    path: Path,
    payload: dict[str, object],
    *,
    created: list[Path],
    updated: list[Path],
) -> None:
    rendered = json.dumps(payload, indent=2) + "\n"
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")
        created.append(path)
        return
    if path.read_text(encoding="utf-8") == rendered:
        return
    path.write_text(rendered, encoding="utf-8")
    updated.append(path)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return _default_settings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Settings file must contain a JSON object: {path}")
    return payload


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _upsert_dotenv(
    path: Path,
    values: dict[str, str],
    *,
    created: list[Path],
    updated: list[Path],
) -> None:
    existing_lines = (
        path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    )
    existing_keys = {
        line.split("=", 1)[0].strip()
        for line in existing_lines
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    existing_values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in existing_lines
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }

    new_lines = list(existing_lines)
    changed = False
    for key, value in values.items():
        if key in existing_keys:
            if not value or existing_values.get(key) == value:
                continue
            new_lines = _replace_dotenv_value(new_lines, key, value)
            changed = True
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


def _default_env_values(provider: str) -> dict[str, str]:
    definition = provider_definition(provider)
    return {item.name: item.default for item in definition.env}


def _profile_settings(
    *,
    provider: str,
    model_tiers: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "default_provider": provider,
        "default_tier": DEFAULT_TIER,
        "model_tiers": model_tiers,
        "model_overrides": _legacy_model_overrides(model_tiers),
    }


def _profile_env_values(
    profile: str,
    provider: str,
    values: dict[str, str],
) -> dict[str, str]:
    definition = provider_definition(provider)
    profile_values: dict[str, str] = {}
    for item in definition.env:
        value = values.get(item.name, item.default)
        profile_values[profile_env_name(profile, item.name)] = value
    return profile_values


def _first_worker_model(model_tiers: dict[str, list[str]]) -> str | None:
    for tier in (DEFAULT_TIER, "scout", "architect"):
        models = model_tiers.get(tier)
        if models:
            return models[0]
    return None


def _legacy_model_overrides(model_tiers: dict[str, list[str]]) -> dict[str, str]:
    return {
        tier: models[0]
        for tier, models in model_tiers.items()
        if tier in {"scout", "worker", "architect"} and models
    }


def _replace_dotenv_value(lines: list[str], key: str, value: str) -> list[str]:
    replaced: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                replaced.append(f"{key}={value}")
                continue
        replaced.append(line)
    return replaced
