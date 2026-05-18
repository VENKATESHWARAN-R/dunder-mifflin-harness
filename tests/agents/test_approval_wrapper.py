"""make_approval_wrapper: gate behavior on every code path."""

from __future__ import annotations

from pathlib import Path

import pytest

from jac.agents.approval import make_approval_wrapper
from jac.runtime.approvals import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.tools.filesystem import read_file, write_file
from jac.tools.types import ToolStatus


def _approve(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=request.id, decision=ApprovalDecision.APPROVE_ONCE
    )


def _deny(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=request.id, decision=ApprovalDecision.DENY, reason="testing"
    )


def _redirect(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=request.id,
        decision=ApprovalDecision.REDIRECT,
        redirect_message="try a different path",
    )


async def _async_approve(request: ApprovalRequest) -> ApprovalResponse:
    return _approve(request)


async def _async_deny(request: ApprovalRequest) -> ApprovalResponse:
    return _deny(request)


async def _async_redirect(request: ApprovalRequest) -> ApprovalResponse:
    return _redirect(request)


@pytest.mark.asyncio
async def test_read_only_tool_skips_gate(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("hi", encoding="utf-8")
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

    async def boom(_: ApprovalRequest) -> ApprovalResponse:
        raise AssertionError("read-only must not invoke callback")

    wrapped = make_approval_wrapper(read_file, policy, boom)
    result = await wrapped(path=str(target))
    assert result.status == ToolStatus.OK
    assert result.content == "hi"


@pytest.mark.asyncio
async def test_write_tool_executes_on_approval(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    wrapped = make_approval_wrapper(write_file, policy, _async_approve)
    result = await wrapped(path=str(target), content="approved")
    assert result.status == ToolStatus.OK
    assert target.read_text(encoding="utf-8") == "approved"


@pytest.mark.asyncio
async def test_write_tool_denied_returns_permission_denied(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    wrapped = make_approval_wrapper(write_file, policy, _async_deny)
    result = await wrapped(path=str(target), content="should not land")
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert result.error is not None and "denied" in result.error.lower()
    assert not target.exists()


@pytest.mark.asyncio
async def test_redirect_returns_user_feedback_to_model(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    wrapped = make_approval_wrapper(write_file, policy, _async_redirect)
    result = await wrapped(path=str(target), content="x")
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert result.error is not None and "try a different path" in result.error
    assert not target.exists()


@pytest.mark.asyncio
async def test_yolo_skips_callback(tmp_path: Path) -> None:
    target = tmp_path / "y.txt"
    policy = ApprovalPolicy(mode=ApprovalMode.YOLO)

    async def boom(_: ApprovalRequest) -> ApprovalResponse:
        raise AssertionError("yolo must not invoke callback")

    wrapped = make_approval_wrapper(write_file, policy, boom)
    result = await wrapped(path=str(target), content="y")
    assert result.status == ToolStatus.OK


@pytest.mark.asyncio
async def test_edit_file_diff_appears_in_request_preview(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("alpha\n", encoding="utf-8")
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

    captured: list[ApprovalRequest] = []

    async def capture(request: ApprovalRequest) -> ApprovalResponse:
        captured.append(request)
        return _approve(request)

    from jac.tools.filesystem import edit_file

    wrapped = make_approval_wrapper(edit_file, policy, capture)
    result = await wrapped(path=str(target), old_string="alpha", new_string="beta")
    assert result.status == ToolStatus.OK
    assert target.read_text(encoding="utf-8") == "beta\n"

    assert len(captured) == 1
    assert captured[0].preview is not None
    assert "alpha" in captured[0].preview and "beta" in captured[0].preview


@pytest.mark.asyncio
async def test_edit_file_compute_error_short_circuits(tmp_path: Path) -> None:
    """If compute_edit fails (no match), wrapper returns the error without prompting."""
    target = tmp_path / "code.py"
    target.write_text("hello\n", encoding="utf-8")
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

    async def boom(_: ApprovalRequest) -> ApprovalResponse:
        raise AssertionError("compute-edit failure must not prompt")

    from jac.tools.filesystem import edit_file

    wrapped = make_approval_wrapper(edit_file, policy, boom)
    result = await wrapped(path=str(target), old_string="missing", new_string="x")
    assert result.status == ToolStatus.ERROR
