"""Sanity tests for the shared tool types and approval metadata shape."""

from __future__ import annotations

import pytest

from jac.tools.types import (
    RiskLevel,
    ScottDeps,
    ToolApprovalMeta,
    ToolResult,
    ToolStatus,
)


def test_tool_result_defaults() -> None:
    result = ToolResult()
    assert result.status == ToolStatus.OK
    assert result.warnings == []
    assert result.error is None


def test_tool_approval_meta_round_trip() -> None:
    meta = ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda path, **_: f"Read `{path}`",
        timeout_seconds=30.0,
    )
    assert meta.description_fn(path="/tmp/x") == "Read `/tmp/x`"
    assert meta.timeout_seconds == 30.0


def test_scott_deps_is_frozen() -> None:
    import dataclasses

    deps = ScottDeps(run_id="r1", tasks_repo=object())
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(deps, "run_id", "r2")
