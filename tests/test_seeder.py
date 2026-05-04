"""Tests for the workspace file → DB seeding pass (C2)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jac.state import open_state_store
from jac.state.seeder import check_workspace_files, seed_workspace
from jac.workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _make_workspace(
    tmp_path: Path,
    *,
    with_project: bool = True,
) -> tuple[Workspace, Path, Path | None]:
    user_dir = tmp_path / ".jac"
    user_dir.mkdir()
    (user_dir / "skills").mkdir()
    (user_dir / "mcp").mkdir()
    project_dir: Path | None = None
    if with_project:
        project_dir = tmp_path / ".agents"
        project_dir.mkdir()
        (project_dir / "skills").mkdir()
        (project_dir / "mcp").mkdir()
    workspace = Workspace(
        cwd=tmp_path,
        user_dir=user_dir,
        project_root=tmp_path if with_project else None,
        project_dir=project_dir,
        project_instruction_path=None,
    )
    return workspace, user_dir, project_dir


def _write_skill(
    directory: Path, filename: str, name: str, domain: str = "general"
) -> Path:
    path = directory / filename
    path.write_text(
        f"---\nname: {name}\ndomain: {domain}\nversion: 1.0\ndescription: A test skill\n---\n\nSkill body for {name}.\n",
        encoding="utf-8",
    )
    return path


def _write_mcp(directory: Path, filename: str, name: str) -> Path:
    path = directory / filename
    path.write_text(
        json.dumps(
            {
                "name": name,
                "description": f"MCP server {name}",
                "transport": "stdio",
                "config": {"command": "npx", "args": [f"@{name}/mcp"]},
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Skill upsert tests
# ---------------------------------------------------------------------------


def test_seed_skill_from_user_scope(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    _write_skill(user_dir / "skills", "react.md", "react-conventions", "frontend")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        row = await state.skills.get_by_name("react-conventions")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert "react-conventions" in result.upserted_skills
    assert row is not None
    assert row.name == "react-conventions"
    assert row.domain == "frontend"
    assert row.source_scope == "user"
    assert row.source_path is not None
    assert row.source_path.endswith("react.md")
    assert "Skill body for react-conventions" in row.content


def test_seed_skill_from_project_scope(tmp_path: Path) -> None:
    workspace, _, project_dir = _make_workspace(tmp_path)
    assert project_dir is not None
    _write_skill(project_dir / "skills", "ts.md", "typescript-style", "backend")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        row = await state.skills.get_by_name("typescript-style")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert "typescript-style" in result.upserted_skills
    assert row is not None
    assert row.source_scope == "project"


def test_project_skill_overrides_user_skill(tmp_path: Path) -> None:
    workspace, user_dir, project_dir = _make_workspace(tmp_path)
    assert project_dir is not None
    _write_skill(user_dir / "skills", "shared.md", "shared-skill", "general")
    _write_skill(project_dir / "skills", "shared.md", "shared-skill", "backend")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        row = await state.skills.get_by_name("shared-skill")
        skills = await state.skills.list_all()
        await state.close()
        return result, row, skills

    result, row, skills = _run(scenario())
    assert result.errors == []
    assert row is not None
    assert row.domain == "backend"
    assert row.source_scope == "project"
    assert len(skills) == 1


def test_duplicate_skill_in_same_scope_skipped(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    _write_skill(user_dir / "skills", "a.md", "dup-skill")
    _write_skill(user_dir / "skills", "b.md", "dup-skill")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        skills = await state.skills.list_all()
        await state.close()
        return result, skills

    result, skills = _run(scenario())
    assert any("dup-skill" in e for e in result.errors)
    assert len(skills) == 0


def test_seed_is_idempotent(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    _write_skill(user_dir / "skills", "react.md", "react-conventions")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        await seed_workspace(workspace, state)
        await seed_workspace(workspace, state)
        skills = await state.skills.list_all()
        await state.close()
        return skills

    skills = _run(scenario())
    assert len(skills) == 1


# ---------------------------------------------------------------------------
# MCP server upsert tests
# ---------------------------------------------------------------------------


def test_seed_mcp_from_user_scope(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    _write_mcp(user_dir / "mcp", "playwright.json", "playwright")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        row = await state.mcp_servers.get_by_name("playwright")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert "playwright" in result.upserted_mcp_servers
    assert row is not None
    assert row.transport == "stdio"
    assert row.source_scope == "user"
    assert json.loads(row.config)["command"] == "npx"


def test_project_mcp_overrides_user_mcp(tmp_path: Path) -> None:
    workspace, user_dir, project_dir = _make_workspace(tmp_path)
    assert project_dir is not None
    _write_mcp(user_dir / "mcp", "pw.json", "playwright")
    project_mcp = project_dir / "mcp" / "pw.json"
    project_mcp.write_text(
        json.dumps(
            {
                "name": "playwright",
                "description": "Project playwright",
                "transport": "http",
                "config": {"url": "http://localhost:8080"},
            }
        ),
        encoding="utf-8",
    )

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        result = await seed_workspace(workspace, state)
        row = await state.mcp_servers.get_by_name("playwright")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert result.errors == []
    assert row is not None
    assert row.transport == "http"
    assert row.source_scope == "project"


# ---------------------------------------------------------------------------
# Garbage collection tests
# ---------------------------------------------------------------------------


def test_stale_skill_is_garbage_collected(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    skill_file = _write_skill(user_dir / "skills", "old.md", "old-skill")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        await seed_workspace(workspace, state)
        # Delete file, then re-seed — row should be GC'd.
        skill_file.unlink()
        result = await seed_workspace(workspace, state)
        row = await state.skills.get_by_name("old-skill")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert "old-skill" in result.deleted_skills
    assert row is None


def test_stale_mcp_is_garbage_collected(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    mcp_file = _write_mcp(user_dir / "mcp", "old.json", "old-server")

    async def scenario():
        state = await open_state_store(tmp_path / "state.db")
        await seed_workspace(workspace, state)
        mcp_file.unlink()
        result = await seed_workspace(workspace, state)
        row = await state.mcp_servers.get_by_name("old-server")
        await state.close()
        return result, row

    result, row = _run(scenario())
    assert "old-server" in result.deleted_mcp_servers
    assert row is None


# ---------------------------------------------------------------------------
# check_workspace_files (doctor) tests
# ---------------------------------------------------------------------------


def test_check_workspace_files_no_issues(tmp_path: Path) -> None:
    workspace, user_dir, project_dir = _make_workspace(tmp_path)
    assert project_dir is not None
    _write_skill(user_dir / "skills", "react.md", "react-conventions")
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(
        ".agents/.env.local\n.agents/settings.local.json\n.agents/state.db\n.agents/logs/\n",
        encoding="utf-8",
    )
    warnings = check_workspace_files(workspace)
    assert warnings == []


def test_check_workspace_files_detects_duplicates(tmp_path: Path) -> None:
    workspace, user_dir, _ = _make_workspace(tmp_path, with_project=False)
    _write_skill(user_dir / "skills", "a.md", "dup")
    _write_skill(user_dir / "skills", "b.md", "dup")
    warnings = check_workspace_files(workspace)
    assert any("dup" in w for w in warnings)


def test_check_workspace_files_missing_gitignore_entries(tmp_path: Path) -> None:
    workspace, _, _ = _make_workspace(tmp_path)
    warnings = check_workspace_files(workspace)
    assert any(".agents/state.db" in w for w in warnings)


def test_check_workspace_files_partial_gitignore(tmp_path: Path) -> None:
    workspace, _, _ = _make_workspace(tmp_path)
    (tmp_path / ".gitignore").write_text(".agents/state.db\n", encoding="utf-8")
    warnings = check_workspace_files(workspace)
    assert any(".env.local" in w for w in warnings)
    assert not any("state.db" in w for w in warnings)
