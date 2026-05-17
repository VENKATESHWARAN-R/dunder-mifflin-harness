"""Runs repo CRUD."""

from __future__ import annotations

from jac.state import StateStore


async def test_create_and_get(state: StateStore) -> None:
    row = await state.runs.create("r1", prompt="hello")
    assert row.run_id == "r1"
    assert row.status == "running"
    assert row.total_cost == 0.0
    assert row.total_tokens == 0

    fetched = await state.runs.get("r1")
    assert fetched is not None
    assert fetched.prompt == "hello"


async def test_update_status(state: StateStore) -> None:
    await state.runs.create("r1", prompt="x")
    await state.runs.update_status("r1", "done")
    row = await state.runs.get("r1")
    assert row is not None
    assert row.status == "done"


async def test_list_recent_orders_by_created_at_desc(state: StateStore) -> None:
    await state.runs.create("r1", prompt="first")
    await state.runs.create("r2", prompt="second")
    rows = await state.runs.list_recent(limit=10)
    ids = [r.run_id for r in rows]
    assert set(ids) == {"r1", "r2"}
