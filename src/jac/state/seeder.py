"""Workspace file → DB seeding pass (C2).

Files on disk are the source of truth; the DB is the resolved index.
On every harness boot this module walks both workspace scopes, upserts
skills/mcp_servers rows, and garbage-collects rows whose source file is gone.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from jac.state.db import StateStore
from jac.workspace import Workspace


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SeedResult:
    upserted_skills: list[str] = field(default_factory=list)
    upserted_mcp_servers: list[str] = field(default_factory=list)
    deleted_skills: list[str] = field(default_factory=list)
    deleted_mcp_servers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal entry containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SkillEntry:
    name: str
    description: str
    domain: str
    version: str
    content: str
    scope: str
    path: Path


@dataclass(slots=True)
class _McpEntry:
    name: str
    description: str
    transport: str
    config: str
    scope: str
    path: Path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def seed_workspace(workspace: Workspace, state: StateStore) -> SeedResult:
    """Walk both workspace scopes, upsert skill/MCP rows, GC stale rows.

    Project scope entries override user scope entries with the same name.
    Duplicate names within one scope are a configuration error; the affected
    entries are skipped and their names added to ``SeedResult.errors``.
    """
    result = SeedResult()

    # Collect entries; project scope runs second so it overwrites user for same name.
    skill_map: dict[str, _SkillEntry] = {}
    mcp_map: dict[str, _McpEntry] = {}

    _collect_skills(workspace.user_dir / "skills", "user", skill_map, result.errors)
    _collect_mcp(workspace.user_dir / "mcp", "user", mcp_map, result.errors)

    if workspace.project_dir is not None:
        _collect_skills(
            workspace.project_dir / "skills", "project", skill_map, result.errors
        )
        _collect_mcp(
            workspace.project_dir / "mcp", "project", mcp_map, result.errors
        )

    # Upsert into DB.
    for entry in skill_map.values():
        await state.skills.upsert(
            name=entry.name,
            description=entry.description,
            domain=entry.domain,
            content=entry.content,
            version=entry.version,
            source_scope=entry.scope,
            source_path=str(entry.path),
        )
        result.upserted_skills.append(entry.name)

    for entry in mcp_map.values():
        await state.mcp_servers.upsert(
            name=entry.name,
            description=entry.description,
            transport=entry.transport,
            config=entry.config,
            source_scope=entry.scope,
            source_path=str(entry.path),
        )
        result.upserted_mcp_servers.append(entry.name)

    # GC: delete rows whose source file no longer exists.
    for scope in ("user", "project"):
        skill_paths = {str(e.path) for e in skill_map.values() if e.scope == scope}
        mcp_paths = {str(e.path) for e in mcp_map.values() if e.scope == scope}
        result.deleted_skills += await state.skills.delete_stale(scope, skill_paths)
        result.deleted_mcp_servers += await state.mcp_servers.delete_stale(
            scope, mcp_paths
        )

    return result


# ---------------------------------------------------------------------------
# Workspace diagnostics (no DB required)
# ---------------------------------------------------------------------------


def check_workspace_files(workspace: Workspace) -> list[str]:
    """Return configuration warnings without touching the DB.

    Reports duplicate skill/MCP names within each scope and missing
    .gitignore entries for project workspaces.
    """
    errors: list[str] = []
    _check_duplicates_in_dir(
        workspace.user_dir / "skills", "user", "skill", _parse_skill_name, errors
    )
    _check_duplicates_in_dir(
        workspace.user_dir / "mcp", "user", "MCP server", _parse_mcp_name, errors
    )
    if workspace.project_dir is not None:
        _check_duplicates_in_dir(
            workspace.project_dir / "skills",
            "project",
            "skill",
            _parse_skill_name,
            errors,
        )
        _check_duplicates_in_dir(
            workspace.project_dir / "mcp",
            "project",
            "MCP server",
            _parse_mcp_name,
            errors,
        )
    if workspace.project_root is not None:
        errors += _check_gitignore(workspace.project_root)
    return errors


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _collect_skills(
    directory: Path,
    scope: str,
    skill_map: dict[str, _SkillEntry],
    errors: list[str],
) -> None:
    if not directory.is_dir():
        return
    seen_in_scope: dict[str, Path] = {}
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            meta, content = _parse_frontmatter(text)
        except Exception as exc:
            errors.append(f"skill {path.name}: parse error — {exc}")
            continue

        name = meta.get("name", "").strip()
        if not name:
            errors.append(f"skill {path.name}: missing 'name' in frontmatter")
            continue

        if name in seen_in_scope:
            errors.append(
                f"skill '{name}': duplicate in {scope} scope "
                f"({seen_in_scope[name].name} and {path.name})"
            )
            skill_map.pop(name, None)
            continue

        seen_in_scope[name] = path
        skill_map[name] = _SkillEntry(
            name=name,
            description=meta.get("description", ""),
            domain=meta.get("domain", "general"),
            version=meta.get("version", "1.0"),
            content=content,
            scope=scope,
            path=path,
        )


def _collect_mcp(
    directory: Path,
    scope: str,
    mcp_map: dict[str, _McpEntry],
    errors: list[str],
) -> None:
    if not directory.is_dir():
        return
    seen_in_scope: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"mcp {path.name}: parse error — {exc}")
            continue

        if not isinstance(data, dict):
            errors.append(f"mcp {path.name}: expected a JSON object")
            continue

        name = data.get("name", "").strip()
        if not name:
            errors.append(f"mcp {path.name}: missing 'name' field")
            continue

        if name in seen_in_scope:
            errors.append(
                f"MCP server '{name}': duplicate in {scope} scope "
                f"({seen_in_scope[name].name} and {path.name})"
            )
            mcp_map.pop(name, None)
            continue

        seen_in_scope[name] = path
        mcp_map[name] = _McpEntry(
            name=name,
            description=data.get("description", ""),
            transport=data.get("transport", "stdio"),
            config=json.dumps(data.get("config", {})),
            scope=scope,
            path=path,
        )


# ---------------------------------------------------------------------------
# Doctor-only helpers (no DB, read-only)
# ---------------------------------------------------------------------------


def _check_duplicates_in_dir(
    directory: Path,
    scope: str,
    kind: str,
    name_fn: Callable[[Path], str | None],
    errors: list[str],
) -> None:
    if not directory.is_dir():
        return
    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        name = name_fn(path)
        if name is None:
            continue
        if name in seen:
            errors.append(
                f"{kind} '{name}': duplicate in {scope} scope "
                f"({seen[name].name} and {path.name})"
            )
        else:
            seen[name] = path


def _parse_skill_name(path: Path) -> str | None:
    if path.suffix != ".md":
        return None
    try:
        meta, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))
        return meta.get("name", "").strip() or None
    except Exception:
        return None


def _parse_mcp_name(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("name", "").strip() or None
    except Exception:
        pass
    return None


def _check_gitignore(project_root: Path) -> list[str]:
    required = [
        ".agents/.env.local",
        ".agents/settings.local.json",
        ".agents/state.db",
        ".agents/logs/",
    ]
    gitignore = project_root / ".gitignore"
    if not gitignore.is_file():
        return [f"missing .gitignore entry: {entry}" for entry in required]
    existing = set(gitignore.read_text(encoding="utf-8").splitlines())
    return [
        f"missing .gitignore entry: {entry}"
        for entry in required
        if entry not in existing
    ]


# ---------------------------------------------------------------------------
# Minimal YAML frontmatter parser
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse ``---`` frontmatter from markdown.  Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" not in line or line.startswith("#"):
            continue
        key, _, raw_value = line.partition(":")
        value = raw_value.strip().strip("\"'")
        meta[key.strip()] = value
    return meta, body
