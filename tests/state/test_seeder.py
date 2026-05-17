"""Workspace seeding: upsert + GC + scope precedence."""

from __future__ import annotations

import json
from pathlib import Path

from jac.state import StateStore, seed_workspace
from jac.workspace import Workspace


def _write_skill(directory: Path, name: str, body: str = "skill body") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: a skill\ndomain: general\nversion: 1.0\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def _write_mcp(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "description": "a server",
                "transport": "stdio",
                "config": {"command": "echo", "args": ["hi"]},
            }
        ),
        encoding="utf-8",
    )
    return path


async def test_seeder_upserts_user_scope(state: StateStore, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    _write_skill(user_dir / "skills", "react-patterns")
    _write_mcp(user_dir / "mcp", "playwright")
    workspace = Workspace(user_dir=user_dir)

    result = await seed_workspace(workspace, state)

    assert result.upserted_skills == ["react-patterns"]
    assert result.upserted_mcp_servers == ["playwright"]
    assert not result.errors


async def test_project_scope_overrides_user(state: StateStore, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "project"
    _write_skill(user_dir / "skills", "shared", body="user version")
    _write_skill(proj_dir / ".agents" / "skills", "shared", body="project version")
    workspace = Workspace(user_dir=user_dir, project_root=proj_dir)

    await seed_workspace(workspace, state)
    skill = await state.skills.get_by_name("shared")
    assert skill is not None
    assert "project version" in skill.content
    assert skill.source_scope == "project"


async def test_seeder_gc_removes_orphan(state: StateStore, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    path = _write_skill(user_dir / "skills", "ephemeral")
    workspace = Workspace(user_dir=user_dir)

    first = await seed_workspace(workspace, state)
    assert first.upserted_skills == ["ephemeral"]

    path.unlink()
    second = await seed_workspace(workspace, state)
    assert second.deleted_skills == ["ephemeral"]
    assert await state.skills.get_by_name("ephemeral") is None


async def test_duplicate_name_within_scope_is_error(
    state: StateStore, tmp_path: Path
) -> None:
    user_dir = tmp_path / "user"
    skills_dir = user_dir / "skills"
    skills_dir.mkdir(parents=True)
    # Two files declaring the same `name` in the same scope.
    (skills_dir / "a.md").write_text(
        "---\nname: clash\ndomain: general\n---\nfirst\n", encoding="utf-8"
    )
    (skills_dir / "b.md").write_text(
        "---\nname: clash\ndomain: general\n---\nsecond\n", encoding="utf-8"
    )
    workspace = Workspace(user_dir=user_dir)

    result = await seed_workspace(workspace, state)
    assert any("clash" in err for err in result.errors)
    assert await state.skills.get_by_name("clash") is None
