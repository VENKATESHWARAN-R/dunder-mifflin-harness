"""Task-CRUD agent tools — Scott's context-resilient memory.

Unlike file/shell tools (which are stateless), these need per-run state:
the `run_id` (to scope the writes) and a `TasksRepo` (to write through).
That comes in via Pydantic AI's `RunContext[ScottDeps]` — the runtime
coordinator constructs `ScottDeps` once per run and passes it to
`agent.run(..., deps=...)`.

Each function still carries a `.approval: ToolApprovalMeta` attribute so
the same factory-side approval wrapper gates them uniformly with the
file/shell tools. `list_tasks` is READ_ONLY; the mutating three are LOW
risk (reversible — the task row stays in the DB; nothing destructive
happens).
"""

from __future__ import annotations

from pydantic_ai import RunContext

from jac.tools.types import (
    RiskLevel,
    ScottDeps,
    TaskInfo,
    TaskListResult,
    TaskResult,
    ToolApprovalMeta,
    ToolStatus,
)


def _row_to_info(row: object) -> TaskInfo:
    """Project a `TasksRepo.TaskRow` onto the flat `TaskInfo` agents see."""
    return TaskInfo(
        task_id=getattr(row, "task_id"),
        title=getattr(row, "title"),
        description=getattr(row, "description"),
        status=getattr(row, "status"),
        order_index=getattr(row, "order_index"),
    )


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


async def add_task(
    ctx: RunContext[ScottDeps],
    title: str,
    description: str = "",
) -> TaskResult:
    """Append a new task to the current run's task list."""
    deps = ctx.deps
    try:
        row = await deps.tasks_repo.create(
            run_id=deps.run_id,
            title=title,
            description=description,
            status="pending",
        )
    except Exception as exc:  # noqa: BLE001 — tools must not raise
        return TaskResult(status=ToolStatus.ERROR, error=str(exc))
    return TaskResult(task=_row_to_info(row))


setattr(
    add_task,
    "approval",
    ToolApprovalMeta(
        category="task",
        risk_level=RiskLevel.LOW,
        reversible=True,
        description_fn=lambda title="", **_: f"Add task: `{title}`",
    ),
)


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


async def update_task(
    ctx: RunContext[ScottDeps],
    task_id: str,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> TaskResult:
    """Update one or more fields on an existing task."""
    deps = ctx.deps
    try:
        row = await deps.tasks_repo.update_fields(
            task_id,
            title=title,
            description=description,
            status=status,
        )
    except Exception as exc:  # noqa: BLE001
        return TaskResult(status=ToolStatus.ERROR, error=str(exc))
    if row is None:
        return TaskResult(
            status=ToolStatus.NOT_FOUND,
            error=f"task not found: {task_id}",
        )
    return TaskResult(task=_row_to_info(row))


setattr(
    update_task,
    "approval",
    ToolApprovalMeta(
        category="task",
        risk_level=RiskLevel.LOW,
        reversible=True,
        description_fn=lambda task_id="", **_: f"Update task `{task_id}`",
    ),
)


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------


async def complete_task(
    ctx: RunContext[ScottDeps],
    task_id: str,
) -> TaskResult:
    """Mark a task as completed."""
    deps = ctx.deps
    try:
        row = await deps.tasks_repo.update_fields(task_id, status="completed")
    except Exception as exc:  # noqa: BLE001
        return TaskResult(status=ToolStatus.ERROR, error=str(exc))
    if row is None:
        return TaskResult(
            status=ToolStatus.NOT_FOUND,
            error=f"task not found: {task_id}",
        )
    return TaskResult(task=_row_to_info(row))


setattr(
    complete_task,
    "approval",
    ToolApprovalMeta(
        category="task",
        risk_level=RiskLevel.LOW,
        reversible=True,
        description_fn=lambda task_id="", **_: f"Complete task `{task_id}`",
    ),
)


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


async def list_tasks(
    ctx: RunContext[ScottDeps],
    statuses: list[str] | None = None,
) -> TaskListResult:
    """List tasks for the current run, optionally filtered by status."""
    deps = ctx.deps
    try:
        rows = await deps.tasks_repo.list_for_run(
            deps.run_id,
            statuses=tuple(statuses) if statuses else None,
        )
    except Exception as exc:  # noqa: BLE001
        return TaskListResult(status=ToolStatus.ERROR, error=str(exc))
    return TaskListResult(tasks=[_row_to_info(r) for r in rows])


setattr(
    list_tasks,
    "approval",
    ToolApprovalMeta(
        category="task",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda statuses=None, **_: (
            "List tasks" + (f" (statuses={statuses})" if statuses else "")
        ),
    ),
)
