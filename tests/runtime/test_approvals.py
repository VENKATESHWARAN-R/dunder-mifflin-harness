"""ApprovalPolicy: auto-resolves what the mode permits, returns None otherwise."""

from __future__ import annotations

from jac.runtime.approvals import (
    ApprovalActionKind,
    ApprovalDecision,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    RiskLevel,
)


def _request(
    *,
    tool_name: str = "write_file",
    action_kind: ApprovalActionKind = ApprovalActionKind.FILE_WRITE,
    exact_key: str | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        summary="x",
        action_kind=action_kind,
        risk=RiskLevel.MEDIUM,
        tool_name=tool_name,
        exact_key=exact_key,
    )


def test_yolo_approves_everything() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.YOLO)
    response = policy.auto_response_for(_request())
    assert response is not None and response.approved


def test_interactive_returns_none_for_unallowlisted() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    assert policy.auto_response_for(_request()) is None


def test_auto_edit_approves_file_writes() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO_EDIT)
    response = policy.auto_response_for(
        _request(action_kind=ApprovalActionKind.FILE_WRITE)
    )
    assert response is not None and response.approved


def test_auto_edit_does_not_approve_shell() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO_EDIT)
    assert (
        policy.auto_response_for(
            _request(tool_name="run_shell", action_kind=ApprovalActionKind.SHELL)
        )
        is None
    )


def test_session_tool_allowance_skips_prompt() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE, allowed_tools={"write_file"})
    response = policy.auto_response_for(_request())
    assert response is not None and response.approved


def test_record_allow_tool_for_session_persists() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    request = _request()
    policy.record_response(
        request,
        ApprovalResponse(
            request_id=request.id,
            decision=ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
        ),
    )
    assert "write_file" in policy.allowed_tools

    # Subsequent request for the same tool should auto-approve.
    assert policy.auto_response_for(_request()) is not None


def test_record_allow_exact_for_session_persists() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    request = _request(exact_key="run_shell:ls")
    policy.record_response(
        request,
        ApprovalResponse(
            request_id=request.id,
            decision=ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
        ),
    )
    assert "run_shell:ls" in policy.allowed_exact
