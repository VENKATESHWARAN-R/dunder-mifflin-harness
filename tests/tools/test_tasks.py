"""Task-CRUD tool tests — exercises the RunContext[ScottDeps] path."""

from __future__ import annotations

from pydantic_ai import RunContext

from jac.tools.tasks import add_task, complete_task, list_tasks, update_task
from jac.tools.types import ScottDeps, ToolStatus


async def test_add_task_writes_through_repo(task_ctx: RunContext[ScottDeps]) -> None:
    result = await add_task(task_ctx, title="write tests", description="for slice 3")
    assert result.status == ToolStatus.OK
    assert result.task is not None
    assert result.task.title == "write tests"
    assert result.task.status == "pending"

    rows = await task_ctx.deps.tasks_repo.list_for_run(task_ctx.deps.run_id)
    assert len(rows) == 1
    assert rows[0].title == "write tests"


async def test_list_tasks_preserves_insertion_order(
    task_ctx: RunContext[ScottDeps],
) -> None:
    await add_task(task_ctx, title="first")
    await add_task(task_ctx, title="second")
    await add_task(task_ctx, title="third")
    result = await list_tasks(task_ctx)
    titles = [t.title for t in result.tasks]
    assert titles == ["first", "second", "third"]


async def test_list_tasks_filters_by_status(task_ctx: RunContext[ScottDeps]) -> None:
    a = await add_task(task_ctx, title="a")
    await add_task(task_ctx, title="b")
    assert a.task is not None
    await complete_task(task_ctx, task_id=a.task.task_id)

    pending = await list_tasks(task_ctx, statuses=["pending"])
    completed = await list_tasks(task_ctx, statuses=["completed"])
    assert [t.title for t in pending.tasks] == ["b"]
    assert [t.title for t in completed.tasks] == ["a"]


async def test_update_task_changes_fields(task_ctx: RunContext[ScottDeps]) -> None:
    created = await add_task(task_ctx, title="raw")
    assert created.task is not None
    updated = await update_task(
        task_ctx,
        task_id=created.task.task_id,
        title="renamed",
        status="in_progress",
    )
    assert updated.status == ToolStatus.OK
    assert updated.task is not None
    assert updated.task.title == "renamed"
    assert updated.task.status == "in_progress"


async def test_complete_task_sets_completed(task_ctx: RunContext[ScottDeps]) -> None:
    created = await add_task(task_ctx, title="finish me")
    assert created.task is not None
    result = await complete_task(task_ctx, task_id=created.task.task_id)
    assert result.status == ToolStatus.OK
    assert result.task is not None
    assert result.task.status == "completed"


async def test_update_unknown_task_returns_not_found(
    task_ctx: RunContext[ScottDeps],
) -> None:
    result = await update_task(task_ctx, task_id="does-not-exist", title="x")
    assert result.status == ToolStatus.NOT_FOUND


async def test_complete_unknown_task_returns_not_found(
    task_ctx: RunContext[ScottDeps],
) -> None:
    result = await complete_task(task_ctx, task_id="does-not-exist")
    assert result.status == ToolStatus.NOT_FOUND
