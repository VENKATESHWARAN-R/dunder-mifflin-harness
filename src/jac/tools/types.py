"""Common types for agent tool results, approval metadata, and Scott deps.

All tool functions return a subtype of `ToolResult` and never raise. The
`ToolApprovalMeta` attached to each tool via `setattr(fn, "approval", ...)`
tells the factory's approval wrapper how to gate the call.

Stateful tools (task-CRUD) take `RunContext[ScottDeps]` as their first arg
so the per-run `run_id` and `TasksRepo` flow in via Pydantic AI deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Status + risk + approval metadata
# ---------------------------------------------------------------------------


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TRUNCATED = "truncated"


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"  # no side effects; auto-approved in all modes
    LOW = "low"  # reversible local change (edit a file)
    MEDIUM = "medium"  # harder to reverse (create new file)
    HIGH = "high"  # significant side effects (shell command)


class ToolResult(BaseModel):
    """Base protocol all agent tool results must follow."""

    status: ToolStatus = ToolStatus.OK
    warnings: list[str] = []
    error: str | None = None  # populated only when status != OK


class ToolApprovalMeta(BaseModel):
    """Approval metadata attached to each agent tool function.

    Read by the factory's approval wrapper at call time. Tools declare what
    they are; the `ApprovalPolicy` decides what to do about it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    category: str
    """Semantic category: file_read | file_write | shell | task."""

    risk_level: RiskLevel
    reversible: bool
    description_fn: Callable[..., str]
    """Called with the tool's kwargs at call time → one-line action string."""

    timeout_seconds: float | None = None
    """Per-tool timeout; None means use the agent-wide default."""


# ---------------------------------------------------------------------------
# Filesystem result types
# ---------------------------------------------------------------------------


class DirEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int  # bytes; 0 for directories


class GrepMatch(BaseModel):
    file: str
    line_number: int
    line: str
    context_before: list[str] = []
    context_after: list[str] = []


class FileReadResult(ToolResult):
    content: str = ""
    path: str = ""
    lines_total: int = 0
    lines_returned: int = 0
    truncated: bool = False


class FileWriteResult(ToolResult):
    path: str = ""
    bytes_written: int = 0
    created: bool = False  # True = new file, False = overwrote existing


class FileEditResult(ToolResult):
    path: str = ""
    replacements_made: int = 0
    diff: str = ""  # unified diff (old → new); populated on success


class DirectoryListResult(ToolResult):
    path: str = ""
    entries: list[DirEntry] = []


class SearchResult(ToolResult):
    root: str = ""
    pattern: str = ""
    matches: list[str] = []


class GrepResult(ToolResult):
    matches: list[GrepMatch] = []
    searched_files: int = 0
    total_matches: int = 0  # includes matches beyond max_matches cap


# ---------------------------------------------------------------------------
# Shell result types
# ---------------------------------------------------------------------------


class ShellToolResult(ToolResult):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    command: str = ""
    cwd: str = ""
    timed_out: bool = False
    truncated: bool = False
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class BackgroundProcessResult(ToolResult):
    process_id: str = ""
    command: str = ""
    cwd: str = ""


class ProcessOutputResult(ToolResult):
    stdout: str = ""
    stderr: str = ""
    running: bool = False
    exit_code: int | None = None  # None while still running


# ---------------------------------------------------------------------------
# Task-CRUD result types
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """A single task row, surfaced to the agent as a flat record."""

    task_id: str
    title: str
    description: str
    status: str
    order_index: int


class TaskResult(ToolResult):
    """Returned by add_task / update_task / complete_task."""

    task: TaskInfo | None = None


class TaskListResult(ToolResult):
    """Returned by list_tasks."""

    tasks: list[TaskInfo] = []


# ---------------------------------------------------------------------------
# Scott deps (per-run state passed to stateful tools via RunContext)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScottDeps:
    """Per-run state passed to Scott's stateful tools.

    Task-CRUD tools take `RunContext[ScottDeps]` as their first arg; the
    runtime coordinator (Slice 4) constructs this once per run and passes
    it to `agent.run(..., deps=...)`.
    """

    run_id: str
    """The run row this turn belongs to. Required for task-CRUD writes."""

    tasks_repo: Any
    """`jac.state.tasks.TasksRepo` — typed as Any to avoid a state→tools cycle."""


# ---------------------------------------------------------------------------
# Tool registry alias
# ---------------------------------------------------------------------------

ToolFn = Any  # async callable with a `.approval: ToolApprovalMeta` attribute
