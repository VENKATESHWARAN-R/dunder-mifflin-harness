"""C6 — Scott + Jim persistence and migration tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.usage import RunUsage

from jac.agents import ensure_builder_config, ensure_manager_config
from jac.agents.tools import make_summon_jim_tool
from jac.agents.personas import PERSONAS
from jac.config import Settings
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.runtime.events import EventBus
from jac.runtime.session import SessionState
from jac.state import open_state_store
from jac.tools.cache import ToolResultCache


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


def test_summon_jim_uses_active_approval_policy(monkeypatch, tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            run_id = "r1"
            await store.runs.create(run_id=run_id, prompt="build")
            await ensure_manager_config(store, run_id)
            await ensure_builder_config(store, run_id)

            settings = Settings()
            session = SessionState()
            session.run_id = run_id
            events = EventBus()
            policy = ApprovalPolicy(mode=ApprovalMode.AUTO_EDIT)
            captured: dict[str, object] = {}

            class FakeResult:
                output = "jim-complete"

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _task: str, *, usage=None):
                    if usage is not None:
                        usage.incr(
                            RunUsage(
                                input_tokens=5,
                                output_tokens=1,
                                requests=1,
                                tool_calls=0,
                            )
                        )
                    return FakeResult()

            async def fake_loader(**kwargs):
                captured["approval_policy"] = kwargs.get("approval_policy")
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)

            summon_jim = make_summon_jim_tool(
                store,
                settings,
                session,
                events,
                approval_policy=policy,
                tool_result_cache=ToolResultCache(),
                summariser=lambda _c, _h, _ctx=None: asyncio.sleep(0, result="summary"),
            )
            ctx = SimpleNamespace(usage=RunUsage())
            output = await summon_jim(ctx, "write tests")
            assert output == "jim-complete"
            assert captured["approval_policy"] is policy
        finally:
            await store.close()

    _run(scenario())


def test_summon_jim_records_parent_child_attempt_and_pass(
    monkeypatch, tmp_path: Path
) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            run_id = "r1"
            await store.runs.create(run_id=run_id, prompt="build")
            await ensure_manager_config(store, run_id)
            await ensure_builder_config(store, run_id)

            settings = Settings()
            session = SessionState()
            session.run_id = run_id
            parent = await store.attempts.create(
                run_id=run_id,
                role="manager",
                model="test-model",
                tier="worker",
            )
            session.active_attempt_id = parent.attempt_id
            events = EventBus()
            policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

            class FakeResult:
                output = "complete"

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _task: str, *, usage=None):
                    if usage is not None:
                        usage.incr(
                            RunUsage(
                                input_tokens=5,
                                output_tokens=1,
                                requests=1,
                                tool_calls=0,
                            )
                        )
                    return FakeResult()

            async def fake_loader(**_kwargs):
                return FakeAgent()

            # patch imported symbol target used by make_summon_jim_tool
            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            summon_jim = make_summon_jim_tool(
                store,
                settings,
                session,
                events,
                approval_policy=policy,
                tool_result_cache=ToolResultCache(),
                summariser=lambda _c, _h, _ctx=None: asyncio.sleep(0, result="summary"),
            )
            ctx = SimpleNamespace(usage=RunUsage())
            output = await summon_jim(ctx, "write tests")
            assert output == "complete"

            cur = await store.connection.execute(
                "SELECT parent_attempt_id, status FROM attempts WHERE role = 'builder'"
            )
            row = await cur.fetchone()
            await cur.close()
            assert row is not None
            assert row["parent_attempt_id"] == parent.attempt_id
            assert row["status"] == "passed"
        finally:
            await store.close()

    _run(scenario())


def test_summon_jim_marks_failed_attempt_on_error(monkeypatch, tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            run_id = "r1"
            await store.runs.create(run_id=run_id, prompt="build")
            await ensure_manager_config(store, run_id)
            await ensure_builder_config(store, run_id)
            settings = Settings()
            session = SessionState()
            session.run_id = run_id
            events = EventBus()
            policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _task: str, *, usage=None):
                    raise RuntimeError("jim failed")

            async def fake_loader(**_kwargs):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            summon_jim = make_summon_jim_tool(
                store,
                settings,
                session,
                events,
                approval_policy=policy,
                tool_result_cache=ToolResultCache(),
                summariser=lambda _c, _h, _ctx=None: asyncio.sleep(0, result="summary"),
            )
            ctx = SimpleNamespace(usage=RunUsage())
            try:
                await summon_jim(ctx, "break")
            except RuntimeError:
                pass

            cur = await store.connection.execute(
                "SELECT status FROM attempts WHERE role = 'builder' ORDER BY created_at DESC LIMIT 1"
            )
            row = await cur.fetchone()
            await cur.close()
            assert row is not None
            assert row["status"] == "failed"
        finally:
            await store.close()

    _run(scenario())
