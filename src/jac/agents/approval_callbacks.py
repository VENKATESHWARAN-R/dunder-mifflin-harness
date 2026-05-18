"""Approval-callback contract used by the wrapper when the policy can't
auto-resolve a request.

Slice 3 ships only the type alias and a deterministic default
(`auto_deny_callback`) that lets tests exercise the wrapper without a live
EventBus. Slice 4 will provide an EventBus-backed callback that handshakes
with the CLI via `asyncio.Future` and surfaces the request to the user.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from jac.runtime.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)

ApprovalCallback = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]
"""Async callable that resolves a request when the policy returns None."""


async def auto_deny_callback(request: ApprovalRequest) -> ApprovalResponse:
    """Default callback for Slice 3 — denies every prompt that reaches it.

    Combined with `ApprovalPolicy(mode=YOLO)` this lets tests run tools
    without prompting; combined with `mode=INTERACTIVE` everything that
    isn't READ_ONLY gets denied, which is a safe pre-Runtime default.
    """
    return ApprovalResponse(
        request_id=request.id,
        decision=ApprovalDecision.DENY,
        reason="no interactive approval callback wired (pre-Runtime default)",
    )
