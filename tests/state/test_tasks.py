"""Tasks repo CRUD + auto order_index."""

from __future__ import annotations

from jac.state import StateStore


async def test_create_auto_increments_order_index(
    state: StateStore, run_id: str
) -> None:
    t0 = await state.tasks.create(run_id=run_id, title="first")
    t1 = await state.tasks.create(run_id=run_id, title="second")
    t2 = await state.tasks.create(run_id=run_id, title="third")
    assert (t0.order_index, t1.order_index, t2.order_index) == (0, 1, 2)


async def test_explicit_order_index_preserved(state: StateStore, run_id: str) -> None:
    row = await state.tasks.create(run_id=run_id, title="x", order_index=42)
    assert row.order_index == 42


async def test_list_for_run_orders_by_order_index(
    state: StateStore, run_id: str
) -> None:
    await state.tasks.create(run_id=run_id, title="c", order_index=2)
    await state.tasks.create(run_id=run_id, title="a", order_index=0)
    await state.tasks.create(run_id=run_id, title="b", order_index=1)
    tasks = await state.tasks.list_for_run(run_id)
    assert [t.title for t in tasks] == ["a", "b", "c"]


async def test_list_for_run_filter_by_status(state: StateStore, run_id: str) -> None:
    await state.tasks.create(run_id=run_id, title="open1")
    done = await state.tasks.create(run_id=run_id, title="done1")
    await state.tasks.update_status(done.task_id, "completed")

    open_only = await state.tasks.list_for_run(run_id, statuses=("pending",))
    assert [t.title for t in open_only] == ["open1"]


async def test_update_fields_patches_only_provided(
    state: StateStore, run_id: str
) -> None:
    row = await state.tasks.create(run_id=run_id, title="orig", description="orig desc")
    updated = await state.tasks.update_fields(row.task_id, title="new title")
    assert updated is not None
    assert updated.title == "new title"
    assert updated.description == "orig desc"
    assert updated.status == "pending"
