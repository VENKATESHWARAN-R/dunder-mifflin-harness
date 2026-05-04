"""Approval-gating wrapper for local agent tools.

Wraps each non-read-only tool function returned by `_resolve_local_tools` so
that every call goes through the approval handshake before execution. The
wrapper:

- Skips the gate for `RiskLevel.READ_ONLY` tools.
- Computes a prospective diff for `file_write` tools and emits
  `FileEditPreviewed` so the renderer can show it before the prompt fires.
- Builds an `ApprovalRequest`, asks the policy for an auto-response, and
  falls through to `events.request_approval` when no auto-response applies.
- Returns a denial `ToolResult` (with the same return type as the wrapped
  tool) when the user denies — no Python exception, so the model gets a
  coherent tool response.
- Emits `FileEditApplied` after a successful file write so the renderer can
  confirm the edit landed.

The wrapper preserves `__name__`, `__doc__`, and `__annotations__` via
`functools.wraps` so pydantic_ai's signature introspection still works.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any

from jac.runtime.approvals import (
    ApprovalActionKind,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.runtime.approvals import (
    RiskLevel as ApprovalRiskLevel,
)
from jac.runtime.events import EventBus, FileEditApplied, FileEditPreviewed
from jac.tools.filesystem import PreparedEdit, compute_edit, preview_write
from jac.tools.types import (
    RiskLevel as ToolRiskLevel,
)
from jac.tools.types import (
    ToolApprovalMeta,
    ToolResult,
    ToolStatus,
)


def make_approval_wrapper(fn: Any, events: EventBus, policy: ApprovalPolicy) -> Any:
    """Return an approval-gated wrapper around `fn`.

    `fn` must carry a `.approval: ToolApprovalMeta` attribute (set on every
    tool in `TOOL_REGISTRY`). The returned coroutine has the same signature
    as `fn` so pydantic_ai can introspect it.
    """
    meta: ToolApprovalMeta = fn.approval
    return_type = _resolve_return_type(fn)

    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> Any:
        if meta.risk_level == ToolRiskLevel.READ_ONLY:
            return await fn(**kwargs)

        prepared_edit: PreparedEdit | None = None
        prospective_diff: str | None = None
        target_path: Path | None = None

        if meta.category == "file_write":
            prepared_edit, prospective_diff, target_path, early_error = (
                _prepare_file_write(fn.__name__, kwargs)
            )
            if early_error is not None:
                # compute_edit failed (file missing, ambiguous match, etc.).
                # No action to approve — surface the error to the model now.
                return early_error
            if prospective_diff is not None and target_path is not None:
                await events.emit(
                    FileEditPreviewed(path=target_path, diff=prospective_diff)
                )

        request = ApprovalRequest(
            summary=_safe_description(meta, kwargs),
            action_kind=_action_kind_for(meta.category),
            risk=_risk_for(meta.risk_level),
            tool_name=fn.__name__,
            details=_details_for(fn.__name__, kwargs),
        )
        response = policy.auto_response_for(request)
        if response is None:
            response = await events.request_approval(request)
        policy.record_response(request, response)

        if not response.approved:
            return _denial_result(return_type, response, fn.__name__)

        if prepared_edit is not None:
            from jac.tools.filesystem import apply_edit

            result = apply_edit(prepared_edit)
        else:
            result = await fn(**kwargs)

        if (
            meta.category == "file_write"
            and isinstance(result, ToolResult)
            and result.status == ToolStatus.OK
        ):
            applied_path = _path_from_result_or_kwargs(result, kwargs)
            if applied_path is not None:
                await events.emit(FileEditApplied(path=applied_path))

        return result

    return wrapper


def _resolve_return_type(fn: Any) -> type:
    """Pick the ToolResult subclass to return for denial."""
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
) -> tuple[PreparedEdit | None, str | None, Path | None, ToolResult | None]:
    """Compute the prospective diff for a file_write tool.

    Returns (prepared_edit, diff_or_None, path_or_None, early_error_or_None).

    For `edit_file`, `prepared_edit` is set so the wrapper can call
    `apply_edit` after approval (avoids re-reading and re-diffing). When
    `compute_edit` fails — file missing, ambiguous match, etc. — the error
    result is surfaced as `early_error` so the wrapper can short-circuit
    without prompting for an impossible action.

    For `write_file`, only the diff and path are returned; failures (e.g.,
    permission denied) only surface at write time, after approval.
    """
    raw_path = kwargs.get("path")
    if not isinstance(raw_path, str):
        return None, None, None, None
    path = Path(raw_path)

    if tool_name == "edit_file":
        prepared = compute_edit(
            path=raw_path,
            old_string=kwargs.get("old_string", ""),
            new_string=kwargs.get("new_string", ""),
            replace_all=bool(kwargs.get("replace_all", False)),
        )
        if isinstance(prepared, PreparedEdit):
            return prepared, prepared.diff, path, None
        # compute_edit returned a FileEditResult error
        return None, None, None, prepared

    if tool_name == "write_file":
        diff = preview_write(raw_path, kwargs.get("content", ""))
        return None, diff, path, None

    return None, None, path, None


def _safe_description(meta: ToolApprovalMeta, kwargs: dict[str, Any]) -> str:
    """Call `meta.description_fn` defensively — if a tool's lambda raises, the
    user still sees a workable summary instead of an internal error."""
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
    return {}


def _denial_result(
    return_type: type, response: ApprovalResponse, tool_name: str
) -> Any:
    """Build a typed denial response so pydantic_ai gets a valid tool result."""
    reason = response.reason or "user denied this action"
    error = f"Action denied by user: {reason}"
    if issubclass(return_type, ToolResult):
        return return_type(status=ToolStatus.PERMISSION_DENIED, error=error)
    return ToolResult(status=ToolStatus.PERMISSION_DENIED, error=error)


def _path_from_result_or_kwargs(
    result: ToolResult, kwargs: dict[str, Any]
) -> Path | None:
    raw = getattr(result, "path", None) or kwargs.get("path")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return None
