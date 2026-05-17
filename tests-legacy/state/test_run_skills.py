"""Tests for RunSkillsRepo."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_create_and_list_active(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            skill = await store.skills.upsert(
                name="pytest",
                description="d",
                domain="testing",
                content="Use pytest for tests.",
                source_scope="seeded",
                source_path=None,
            )
            await store.run_skills.create(run_id="run-1", skill_id=skill.skill_id)
            rows = await store.run_skills.list_active_for_run("run-1")
            return rows
        finally:
            await store.close()

    rows = _run(scenario())
    assert len(rows) == 1
    assert rows[0].run_id == "run-1"


def test_list_active_with_details(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            skill = await store.skills.upsert(
                name="typescript",
                description="d",
                domain="frontend",
                content="Prefer strict mode.",
                source_scope="seeded",
                source_path=None,
            )
            await store.run_skills.create(run_id="run-1", skill_id=skill.skill_id)
            rows = await store.run_skills.list_active_for_run_with_details("run-1")
            return rows
        finally:
            await store.close()

    rows = _run(scenario())
    assert len(rows) == 1
    assert rows[0].name == "typescript"
    assert rows[0].domain == "frontend"
    assert rows[0].content == "Prefer strict mode."


def test_list_active_respects_agent_role(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            skill = await store.skills.upsert(
                name="scoped-skill",
                description="d",
                domain="general",
                content="Scoped.",
                source_scope="seeded",
                source_path=None,
            )
            await store.run_skills.create(
                run_id="run-1", skill_id=skill.skill_id, agent_role="planner"
            )
            chat_rows = await store.run_skills.list_active_for_run(
                "run-1", agent_role="chat"
            )
            all_rows = await store.run_skills.list_active_for_run("run-1")
            planner_rows = await store.run_skills.list_active_for_run(
                "run-1", agent_role="planner"
            )
            return chat_rows, all_rows, planner_rows
        finally:
            await store.close()

    chat_rows, all_rows, planner_rows = _run(scenario())
    assert len(chat_rows) == 0
    assert len(all_rows) == 1
    assert len(planner_rows) == 1


def test_toggle(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            skill = await store.skills.upsert(
                name="toggle-skill",
                description="d",
                domain="general",
                content="Toggle me.",
                source_scope="seeded",
                source_path=None,
            )
            row = await store.run_skills.create(run_id="run-1", skill_id=skill.skill_id)
            assert row.enabled == 1
            updated = await store.run_skills.toggle(row.id, enabled=0)
            assert updated is not None
            assert updated.enabled == 0
            active = await store.run_skills.list_active_for_run("run-1")
            return active
        finally:
            await store.close()

    active = _run(scenario())
    assert len(active) == 0


def test_ordering_by_domain_and_name(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            z_skill = await store.skills.upsert(
                name="z-skill",
                description="d",
                domain="zzz",
                content="Zzz.",
                source_scope="seeded",
                source_path=None,
            )
            python_skill = await store.skills.upsert(
                name="python-skill",
                description="d",
                domain="python",
                content="Python.",
                source_scope="seeded",
                source_path=None,
            )
            general_skill = await store.skills.upsert(
                name="general-skill",
                description="d",
                domain="general",
                content="General.",
                source_scope="seeded",
                source_path=None,
            )
            # Insert in reverse order to prove sorting works
            await store.run_skills.create(run_id="run-1", skill_id=z_skill.skill_id)
            await store.run_skills.create(
                run_id="run-1", skill_id=python_skill.skill_id
            )
            await store.run_skills.create(
                run_id="run-1", skill_id=general_skill.skill_id
            )
            rows = await store.run_skills.list_active_for_run_with_details("run-1")
            return [r.name for r in rows]
        finally:
            await store.close()

    names = _run(scenario())
    # ORDER BY domain ASC, name ASC  => general < python < zzz
    assert names == ["general-skill", "python-skill", "z-skill"]
