"""C7 usage tracking — attempts repo, model specs, session counters."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jac.runtime.events import EventBus, SessionUsageUpdated
from jac.runtime.model_specs import spec_for
from jac.runtime.session import SessionState
from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_update_usage_persists_columns(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            a = await store.attempts.create(
                run_id="r1",
                role="manager",
                model="anthropic:claude-sonnet-4-6",
                tier="worker",
            )
            await store.attempts.update_usage(
                a.attempt_id,
                tokens_in=100,
                tokens_out=20,
                requests=3,
                tool_calls=2,
                duration_ms=500,
            )
            rows = await store.attempts.list_for_run("r1")
            assert len(rows) == 1
            r = rows[0]
            assert r.tokens_in == 100
            assert r.tokens_out == 20
            assert r.requests == 3
            assert r.tool_calls == 2
            assert r.duration_ms == 500
        finally:
            await store.close()

    _run(scenario())


def test_tree_for_run_parent_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            parent = await store.attempts.create(
                run_id="r1",
                role="manager",
                model="m",
                tier="worker",
            )
            child = await store.attempts.create(
                run_id="r1",
                role="builder",
                model="m2",
                tier="worker",
                parent_attempt_id=parent.attempt_id,
            )
            tree = await store.attempts.tree_for_run("r1")
            assert len(tree) == 1
            root = tree[0]
            assert root.row.attempt_id == parent.attempt_id
            assert len(root.children) == 1
            assert root.children[0].row.attempt_id == child.attempt_id
        finally:
            await store.close()

    _run(scenario())


def test_totals_for_run_groups(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = await open_state_store(tmp_path / "s.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            a1 = await store.attempts.create(
                run_id="r1", role="manager", model="m", tier="worker"
            )
            a2 = await store.attempts.create(
                run_id="r1", role="builder", model="m", tier="architect"
            )
            await store.attempts.update_usage(
                a1.attempt_id,
                tokens_in=10,
                tokens_out=2,
                requests=1,
                tool_calls=1,
                duration_ms=1,
            )
            await store.attempts.update_usage(
                a2.attempt_id,
                tokens_in=5,
                tokens_out=1,
                requests=1,
                tool_calls=0,
                duration_ms=1,
            )
            totals = await store.attempts.totals_for_run("r1")
            assert totals.tokens_in == 15
            assert totals.tokens_out == 3
            assert totals.by_role["manager"][0] == 10
            assert totals.by_role["builder"][0] == 5
            assert totals.by_tier["worker"][0] == 10
            assert totals.by_tier["architect"][0] == 5
        finally:
            await store.close()

    _run(scenario())


def test_spec_for_hit_and_default() -> None:
    s = spec_for("anthropic:claude-sonnet-4-6")
    assert s.max_context == 200_000
    unknown = spec_for("anthropic:claude-totally-fake-model")
    assert unknown.max_context == 200_000


def test_session_usage_updated_emitted_on_clear() -> None:
    async def scenario() -> None:
        events = EventBus()
        seen: list[SessionUsageUpdated] = []

        async def capture(ev: SessionUsageUpdated) -> None:
            seen.append(ev)

        events.on(SessionUsageUpdated, capture)
        session = SessionState()
        session.cumulative_tokens_in = 99
        session.reset_usage_counters()
        assert session.cumulative_tokens_in == 0
        model_ref = "anthropic:claude-sonnet-4-6"
        mx = spec_for(model_ref).max_context
        await events.emit(
            SessionUsageUpdated(
                tokens_in=0,
                tokens_out=0,
                requests=0,
                tool_calls=0,
                last_context_tokens=0,
                context_max=mx,
                context_pct=0.0,
                model=model_ref,
            )
        )
        assert len(seen) == 1
        assert seen[0].tokens_in == 0
        assert seen[0].context_max == mx

    _run(scenario())
