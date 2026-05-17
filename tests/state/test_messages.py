"""Messages repo CRUD."""

from __future__ import annotations

from jac.state import StateStore


async def test_append_and_list(state: StateStore, run_id: str) -> None:
    await state.messages.append(run_id, role="user", content="hi")
    await state.messages.append(run_id, role="assistant", content="hello")
    rows = await state.messages.list_for_run(run_id)
    assert [r.role for r in rows] == ["user", "assistant"]
    assert [r.content for r in rows] == ["hi", "hello"]


async def test_count_for_run(state: StateStore, run_id: str) -> None:
    assert await state.messages.count_for_run(run_id) == 0
    await state.messages.append(run_id, role="user", content="hi")
    await state.messages.append(run_id, role="user", content="there")
    assert await state.messages.count_for_run(run_id) == 2


async def test_task_id_nullable(state: StateStore, run_id: str) -> None:
    row = await state.messages.append(run_id, role="user", content="x")
    assert row.task_id is None
