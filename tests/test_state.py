"""Unit tests for the SQLite state package."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_open_creates_db_and_runs_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"

    async def scenario() -> tuple[set[str], str]:
        store = await open_state_store(db_path)
        try:
            cursor = await store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row["name"] for row in await cursor.fetchall()}
            await cursor.close()
            cursor = await store.connection.execute(
                "SELECT value FROM schema_meta WHERE key='version'"
            )
            row = await cursor.fetchone()
            await cursor.close()
            assert row is not None
            return tables, row["value"]
        finally:
            await store.close()

    tables, version = _run(scenario())

    expected = {
        "schema_meta",
        "runs",
        "tasks",
        "attempts",
        "agent_configs",
        "context_store",
        "messages",
        "mcp_servers",
        "skills",
        "run_mcp_servers",
        "run_skills",
        "agent_teams",
        "agent_instances",
        "agent_messages",
    }
    assert expected.issubset(tables)
    assert version == "1.1"
    assert db_path.exists()


def test_runs_repo_create_and_get(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            created = await store.runs.create(run_id="run-abc", prompt="hello world")
            fetched = await store.runs.get("run-abc")
            return created, fetched
        finally:
            await store.close()

    created, fetched = _run(scenario())
    assert fetched is not None
    assert fetched.run_id == "run-abc"
    assert fetched.prompt == "hello world"
    assert fetched.status == "running"
    assert fetched.workflow_mode == "feature_by_feature"
    assert created.created_at == fetched.created_at


def test_runs_repo_update_status(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-x", prompt="p")
            await store.runs.update_status("run-x", "done")
            return await store.runs.get("run-x")
        finally:
            await store.close()

    row = _run(scenario())
    assert row is not None
    assert row.status == "done"


def test_messages_repo_append_and_list(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            await store.messages.append("run-1", "user", "hi")
            await store.messages.append("run-1", "assistant", "hello")
            await store.messages.append("run-1", "user", "again")
            return await store.messages.list_for_run("run-1")
        finally:
            await store.close()

    rows = _run(scenario())
    assert [r.role for r in rows] == ["user", "assistant", "user"]
    assert [r.content for r in rows] == ["hi", "hello", "again"]


def test_messages_count_for_run(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-c", prompt="p")
            await store.messages.append("run-c", "user", "u1")
            await store.messages.append("run-c", "assistant", "a1")
            return await store.messages.count_for_run("run-c")
        finally:
            await store.close()

    assert _run(scenario()) == 2


def test_open_is_idempotent(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        await store.runs.create(run_id="run-keep", prompt="p")
        await store.close()
        store2 = await open_state_store(tmp_path / "state.db")
        try:
            row = await store2.runs.get("run-keep")
            cursor = await store2.connection.execute(
                "SELECT value FROM schema_meta WHERE key='version'"
            )
            version_row = await cursor.fetchone()
            await cursor.close()
            assert version_row is not None
            return row, version_row["value"]
        finally:
            await store2.close()

    row, version = _run(scenario())
    assert row is not None
    assert version == "1.1"


def test_foreign_key_enforced(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.messages.append("does-not-exist", "user", "orphan")
        finally:
            await store.close()

    with pytest.raises(aiosqlite.IntegrityError):
        _run(scenario())
