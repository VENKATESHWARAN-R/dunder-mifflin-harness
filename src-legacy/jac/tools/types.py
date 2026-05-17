"""Common types for agent tool results and approval metadata."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TRUNCATED = "truncated"  # partial success — output exists but was cut


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"  # no side effects; auto-approved in all modes
    LOW = "low"  # reversible local change (edit a file)
    MEDIUM = "medium"  # harder to reverse (create new file)
    HIGH = "high"  # significant side effects (shell command)


class ToolResult(BaseModel):
    """Base protocol all agent tool results must follow.

    The approval layer, event emitter, and evaluator all read this base.
    Tool-specific subtypes add their payload fields on top.
    """

    status: ToolStatus = ToolStatus.OK
    warnings: list[str] = []
    error: str | None = None  # populated only when status != OK


class ToolApprovalMeta(BaseModel):
    """Approval metadata attached to each agent tool function via setattr(fn, 'approval', ...).

    The approval policy reads this at call time to decide whether to prompt the user.
    Tools declare what they are; the policy decides what to do about it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    category: str
    """Semantic category: file_read | file_write | shell"""

    risk_level: RiskLevel
    """How risky and hard-to-reverse this tool is."""

    reversible: bool
    """Whether the action can be undone (e.g. via git checkout or file restore)."""

    description_fn: Callable[..., str]
    """Called with the tool's kwargs at call time → human-readable action string for approval prompt."""

    timeout_seconds: float | None = None
    """Per-tool timeout enforced by the approval wrapper via asyncio.wait_for.

    None means the tool uses the agent-wide default tool timeout.
    """


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
    context_before: list[str] = []  # lines before the match (when context_lines > 0)
    context_after: list[str] = []  # lines after the match  (when context_lines > 0)


class FileReadResult(ToolResult):
    content: str = ""
    path: str = ""
    lines_total: int = 0
    lines_returned: int = 0
    truncated: bool = False


class FileReadSmartResult(FileReadResult):
    """Result of `read_file_smart` with size-guidance flags."""

    large: bool = False
    metadata_only: bool = False


class FileWriteResult(ToolResult):
    path: str = ""
    bytes_written: int = 0
    created: bool = False  # True = new file, False = overwrote existing


class FileEditResult(ToolResult):
    path: str = ""
    replacements_made: int = 0
    diff: str = ""  # unified diff (old → new); populated on success so callers can emit FileEditPreviewed


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
    """Result for run_shell (blocking). Used by both CLI and agent tools."""

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
    """Result for run_shell_background — non-blocking. Use process_id to poll output."""

    process_id: str = ""
    command: str = ""
    cwd: str = ""


class ProcessEntry(BaseModel):
    process_id: str
    command: str
    started_at: str
    running: bool
    exit_code: int | None = None  # None while still running


class ProcessListResult(ToolResult):
    processes: list[ProcessEntry] = []


class ProcessOutputResult(ToolResult):
    stdout: str = ""
    stderr: str = ""
    running: bool = False
    exit_code: int | None = None  # None while still running


class SummarizedToolResult(ToolResult):
    """Result envelope returned when a large tool output is summarized."""

    summary: str
    summarized: bool = True
    original_tokens: int
    full_result_handle: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Tool registry type
# ---------------------------------------------------------------------------

ToolFn = (
    Any  # async callable with a `.approval: ToolApprovalMeta` attribute set via setattr
)
