"""Approval-gating wrapper for local agent tools.

Wraps every non-READ_ONLY tool so the call goes through the approval
handshake before execution. The wrapper:

- Skips the gate for `RiskLevel.READ_ONLY` tools — they auto-execute.
- Computes a unified diff for `file_write` / `edit_file` calls and stashes
  it in `ApprovalRequest.preview` so Slice 4's EventBus renderer can
  surface a `FileEditPreviewed` event when it lands.
- Asks the `ApprovalPolicy` for an auto-response. If `None`, falls
  through to the supplied `ApprovalCallback` (Slice 3 default: auto-deny).
- Returns a typed denial / redirect `ToolResult` when refused — the model
  sees a coherent tool response, never an exception.

The wrapper accepts `*args, **kwargs` so Pydantic AI tools that take
`ctx: RunContext[ScottDeps]` as their first positional arg (task-CRUD)
work alongside stateless tools that only take kwargs (file/shell).
`functools.wraps` preserves `__wrapped__`, which Pydantic AI uses for
signature introspection.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

from jac.agents.approval_callbacks import ApprovalCallback
from jac.runtime.approvals import (
    ApprovalActionKind,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.runtime.approvals import RiskLevel as ApprovalRiskLevel
from jac.tools.filesystem import PreparedEdit, apply_edit, compute_edit, preview_write
from jac.tools.types import RiskLevel as ToolRiskLevel
from jac.tools.types import (
    ToolApprovalMeta,
    ToolResult,
    ToolStatus,
)


def make_approval_wrapper(
    fn: Any,
    policy: ApprovalPolicy,
    approval_callback: ApprovalCallback,
) -> Any:
    """Return an approval-gated wrapper around `fn`.

    `fn` must carry a `.approval: ToolApprovalMeta` attribute. The returned
    coroutine has the same signature as `fn` (via `__wrapped__`) so
    Pydantic AI can introspect it.
    """
    meta: ToolApprovalMeta = fn.approval
    return_type = _resolve_return_type(fn)

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if meta.risk_level == ToolRiskLevel.READ_ONLY:
            return await _call_with_timeout(
                fn(*args, **kwargs), meta.timeout_seconds, return_type
            )

        prepared_edit: PreparedEdit | None = None
        prospective_diff: str | None = None

        if meta.category == "file_write":
            prepared_edit, prospective_diff, early_error = _prepare_file_write(
                fn.__name__, kwargs
            )
            if early_error is not None:
                return early_error

        request = ApprovalRequest(
            summary=_safe_description(meta, kwargs),
            action_kind=_action_kind_for(meta.category),
            risk=_risk_for(meta.risk_level),
            tool_name=fn.__name__,
            details=_details_for(fn.__name__, kwargs),
            preview=prospective_diff,
        )
        response = policy.auto_response_for(request)
        if response is None:
            response = await approval_callback(request)
        policy.record_response(request, response)

        if response.decision == ApprovalDecision.REDIRECT:
            return _redirect_result(return_type, response)

        if not response.approved:
            return _denial_result(return_type, response)

        if prepared_edit is not None:
            return apply_edit(prepared_edit)
        return await _call_with_timeout(
            fn(*args, **kwargs), meta.timeout_seconds, return_type
        )

    return wrapper


async def _call_with_timeout(
    coro: Any, timeout: float | None, return_type: type
) -> Any:
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if issubclass(return_type, ToolResult):
            return return_type(
                status=ToolStatus.TIMEOUT,
                error=f"tool timed out after {timeout}s",
            )
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            error=f"tool timed out after {timeout}s",
        )


def _resolve_return_type(fn: Any) -> type:
    """Pick the `ToolResult` subclass to use for denial / timeout results."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ToolResult
    annotation = sig.return_annotation
    if annotation is inspect.Signature.empty:
        return ToolResult
    if isinstance(annotation, type) and issubclass(annotation, ToolResult):
        return annotation
    return ToolResult


def _prepare_file_write(
    tool_name: str, kwargs: dict[str, Any]
) -> tuple[PreparedEdit | None, str | None, ToolResult | None]:
    """Compute the prospective diff for a file_write tool.

    Returns `(prepared_edit, diff_or_None, early_error_or_None)`.

    For `edit_file`, `prepared_edit` is set so the wrapper can call
    `apply_edit` after approval without re-reading the file. When
    `compute_edit` fails (file missing, ambiguous match, etc.), the error
    surfaces as `early_error` so the wrapper short-circuits without
    prompting for an impossible action.
    """
    raw_path = kwargs.get("path")
    if not isinstance(raw_path, str):
        return None, None, None

    if tool_name == "edit_file":
        prepared = compute_edit(
            path=raw_path,
            old_string=kwargs.get("old_string", ""),
            new_string=kwargs.get("new_string", ""),
            replace_all=bool(kwargs.get("replace_all", False)),
        )
        if isinstance(prepared, PreparedEdit):
            return prepared, prepared.diff, None
        return None, None, prepared

    if tool_name == "write_file":
        diff = preview_write(raw_path, kwargs.get("content", ""))
        return None, diff, None

    return None, None, None


def _safe_description(meta: ToolApprovalMeta, kwargs: dict[str, Any]) -> str:
    """Call `description_fn` defensively — if a lambda raises, the user
    still sees a workable summary instead of an internal error."""
    try:
        return meta.description_fn(**kwargs)
    except Exception:  # noqa: BLE001 — description must never break the gate
        return f"{meta.category} action"


def _action_kind_for(category: str) -> ApprovalActionKind:
    if category == "file_read":
        return ApprovalActionKind.TOOL
    if category == "file_write":
        return ApprovalActionKind.FILE_WRITE
    if category == "shell":
        return ApprovalActionKind.SHELL
    return ApprovalActionKind.OTHER


def _risk_for(level: ToolRiskLevel) -> ApprovalRiskLevel:
    if level == ToolRiskLevel.HIGH:
        return ApprovalRiskLevel.HIGH
    if level == ToolRiskLevel.LOW:
        return ApprovalRiskLevel.LOW
    return ApprovalRiskLevel.MEDIUM


def _details_for(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Surface a small subset of args to the renderer/log without dumping
    everything (file content, full env, etc.)."""
    if tool_name in {"write_file", "edit_file", "read_file"}:
        return {"path": kwargs.get("path", "")}
    if tool_name in {"run_shell", "run_shell_background"}:
        return {
            "command": kwargs.get("command", ""),
            "cwd": kwargs.get("cwd", ""),
        }
    if tool_name in {"add_task", "update_task", "complete_task"}:
        return {
            "task_id": kwargs.get("task_id", ""),
            "title": kwargs.get("title", ""),
        }
    return {}


def _denial_result(return_type: type, response: ApprovalResponse) -> Any:
    reason = response.reason or "user denied this action"
    error = f"Action denied by user: {reason}"
    if issubclass(return_type, ToolResult):
        return return_type(status=ToolStatus.PERMISSION_DENIED, error=error)
    return ToolResult(status=ToolStatus.PERMISSION_DENIED, error=error)


def _redirect_result(return_type: type, response: ApprovalResponse) -> Any:
    """Return user feedback as the tool result so the model adjusts and retries."""
    feedback = response.redirect_message or "Please adjust your approach and try again."
    message = f"[User feedback] {feedback}"
    if issubclass(return_type, ToolResult):
        return return_type(status=ToolStatus.PERMISSION_DENIED, error=message)
    return ToolResult(status=ToolStatus.PERMISSION_DENIED, error=message)
