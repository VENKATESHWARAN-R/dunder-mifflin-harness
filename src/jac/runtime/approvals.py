"""Approval policy contracts.

Slice 3 ships only the data types and the small session-scoped
`ApprovalPolicy` decider. There is no `EventBus` here yet — the Tools-slice
factory wires the wrapper to a plain `ApprovalCallback` (see
`jac.agents.approval_callbacks`). Slice 4 (Runtime) will swap in the real
EventBus-backed callback that handshakes via `asyncio.Future` against the
CLI.

The risk enum here (LOW / MEDIUM / HIGH) is the *approval* axis — what the
human is asked to authorize. The `RiskLevel` in `jac.tools.types`
(READ_ONLY / LOW / MEDIUM / HIGH) is the *tool* axis — what the tool itself
declares. The wrapper bridges the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ApprovalMode(StrEnum):
    """How the runtime should handle approval requests."""

    INTERACTIVE = "interactive"
    AUTO_EDIT = "auto-edit"
    YOLO = "yolo"


class ApprovalActionKind(StrEnum):
    """High-level action categories used for rendering and policy."""

    TOOL = "tool"
    SHELL = "shell"
    FILE_EDIT = "file-edit"
    FILE_WRITE = "file-write"
    OTHER = "other"


class RiskLevel(StrEnum):
    """Coarse risk labels surfaced in the approval prompt."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalDecision(StrEnum):
    """Possible answers to an approval request."""

    APPROVE_ONCE = "approve_once"
    DENY = "deny"
    ALLOW_TOOL_FOR_SESSION = "allow_tool_for_session"
    ALLOW_EXACT_FOR_SESSION = "allow_exact_for_session"
    REDIRECT = "redirect"
    """Deny execution but route user feedback back to the model."""


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A concrete action that may need human approval."""

    summary: str
    action_kind: ApprovalActionKind = ApprovalActionKind.OTHER
    details: dict[str, Any] = field(default_factory=dict)
    risk: RiskLevel = RiskLevel.MEDIUM
    allowed_decisions: tuple[ApprovalDecision, ...] = (
        ApprovalDecision.APPROVE_ONCE,
        ApprovalDecision.DENY,
    )
    tool_name: str | None = None
    exact_key: str | None = None
    preview: str | None = None
    """Diff or other preview text for file_write tools. Slice 4's renderer
    will surface this as a `FileEditPreviewed` event."""
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True, slots=True)
class ApprovalResponse:
    """User or policy response to an approval request."""

    request_id: str
    decision: ApprovalDecision
    reason: str | None = None
    redirect_message: str | None = None
    """Set when decision == REDIRECT; surfaced to the model as the tool
    result so it can adjust before retrying."""

    @property
    def approved(self) -> bool:
        return self.decision in {
            ApprovalDecision.APPROVE_ONCE,
            ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
            ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
        }


class ApprovalPolicy:
    """Session-scoped policy decider.

    Returns an `ApprovalResponse` from `auto_response_for` when a request
    can be auto-approved by the current mode or a session allowance.
    Returns `None` when the user must be prompted — at which point the
    wrapper falls through to its `ApprovalCallback`.
    """

    _auto_edit_actions = {
        ApprovalActionKind.FILE_EDIT,
        ApprovalActionKind.FILE_WRITE,
    }

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.INTERACTIVE,
        allowed_tools: set[str] | None = None,
        allowed_exact: set[str] | None = None,
    ) -> None:
        self.mode = mode
        self.allowed_tools = allowed_tools or set()
        self.allowed_exact = allowed_exact or set()

    def auto_response_for(self, request: ApprovalRequest) -> ApprovalResponse | None:
        if self.mode == ApprovalMode.YOLO:
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.APPROVE_ONCE,
                reason="yolo mode",
            )

        if request.tool_name and request.tool_name in self.allowed_tools:
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.APPROVE_ONCE,
                reason=f"tool allowed for session: {request.tool_name}",
            )

        if request.exact_key and request.exact_key in self.allowed_exact:
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.APPROVE_ONCE,
                reason="exact action allowed for session",
            )

        if (
            self.mode == ApprovalMode.AUTO_EDIT
            and request.action_kind in self._auto_edit_actions
        ):
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.APPROVE_ONCE,
                reason="auto-edit mode",
            )

        return None

    def record_response(
        self, request: ApprovalRequest, response: ApprovalResponse
    ) -> None:
        """Persist session allowances implied by a response."""
        if response.decision == ApprovalDecision.ALLOW_TOOL_FOR_SESSION:
            if request.tool_name:
                self.allowed_tools.add(request.tool_name)
        elif response.decision == ApprovalDecision.ALLOW_EXACT_FOR_SESSION:
            if request.exact_key:
                self.allowed_exact.add(request.exact_key)
