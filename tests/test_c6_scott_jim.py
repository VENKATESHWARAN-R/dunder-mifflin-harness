"""C6 — Scott + Jim persistence and migration tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jac.agents import ensure_builder_config, ensure_manager_config
from jac.agents.personas import PERSONAS
from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_manager_seed_idempotent(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            first = await ensure_manager_config(store, "r1")
            second = await ensure_manager_config(store, "r1")
            assert first.config_id == second.config_id
            cur = await store.connection.execute(
                "SELECT COUNT(*) AS c FROM agent_configs WHERE run_id = ? AND role = ?",
                ("r1", "manager"),
            )
            row = await cur.fetchone()
            await cur.close()
            assert row is not None
            assert row["c"] == 1
        finally:
            await store.close()

    _run(scenario())


def test_builder_seed_idempotent(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            first = await ensure_builder_config(store, "r1")
            second = await ensure_builder_config(store, "r1")
            assert first.config_id == second.config_id
            cur = await store.connection.execute(
                "SELECT COUNT(*) AS c FROM agent_configs WHERE run_id = ? AND role = ?",
                ("r1", "builder"),
            )
            row = await cur.fetchone()
            await cur.close()
            assert row is not None
            assert row["c"] == 1
        finally:
            await store.close()

    _run(scenario())


def test_agent_config_row_has_persona(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            await ensure_manager_config(store, "r1")
            row = await store.agent_configs.get_by_run_and_role("r1", "manager")
            assert row is not None
            p = PERSONAS["manager"]
            assert row.persona == p.persona
            assert row.display_name == p.display_name
        finally:
            await store.close()

    _run(scenario())


def test_attempt_create_and_update(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            att = await store.attempts.create(
                run_id="r1",
                role="manager",
                model="test-model",
                tier="worker",
            )
            assert att.status == "running"
            await store.attempts.update_status(att.attempt_id, "passed")
            cur = await store.connection.execute(
                "SELECT status FROM attempts WHERE attempt_id = ?",
                (att.attempt_id,),
            )
            r = await cur.fetchone()
            await cur.close()
            assert r is not None
            assert r["status"] == "passed"
        finally:
            await store.close()

    _run(scenario())


def test_attempt_parent_child(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            scott = await store.attempts.create(
                run_id="r1",
                role="manager",
                model="m1",
                tier="worker",
            )
            jim = await store.attempts.create(
                run_id="r1",
                role="builder",
                model="m2",
                tier="worker",
                parent_attempt_id=scott.attempt_id,
            )
            assert jim.parent_attempt_id == scott.attempt_id
            cur = await store.connection.execute(
                "SELECT parent_attempt_id FROM attempts WHERE attempt_id = ?",
                (jim.attempt_id,),
            )
            r = await cur.fetchone()
            await cur.close()
            assert r is not None
            assert r["parent_attempt_id"] == scott.attempt_id
        finally:
            await store.close()

    _run(scenario())


def test_migration_002_applies(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            cur = await store.connection.execute("PRAGMA table_info(agent_configs)")
            cols = {row["name"] for row in await cur.fetchall()}
            await cur.close()
            cur = await store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='attempts'"
            )
            att = await cur.fetchone()
            await cur.close()
            return cols, att is not None
        finally:
            await store.close()

    cols, has_attempts = _run(scenario())
    assert "persona" in cols
    assert "display_name" in cols
    assert has_attempts
