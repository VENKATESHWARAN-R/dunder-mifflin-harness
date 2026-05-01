"""Approval policy types shared by CLI and runtime code."""

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
    """Coarse risk labels for human review."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalDecision(StrEnum):
    """Possible answers to an approval request."""

    APPROVE_ONCE = "approve_once"
    DENY = "deny"
    ALLOW_TOOL_FOR_SESSION = "allow_tool_for_session"
    ALLOW_EXACT_FOR_SESSION = "allow_exact_for_session"


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
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True, slots=True)
class ApprovalResponse:
    """User or policy response to an approval request."""

    request_id: str
    decision: ApprovalDecision
    reason: str | None = None

    @property
    def approved(self) -> bool:
        """Whether this response allows execution to proceed."""
        return self.decision in {
            ApprovalDecision.APPROVE_ONCE,
            ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
            ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
        }


class ApprovalPolicy:
    """Session-scoped approval policy.

    This policy is intentionally small. It decides whether a request can be
    auto-approved and records session allowances chosen by the user.
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
        """Return a policy response when the request does not need prompting."""
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
