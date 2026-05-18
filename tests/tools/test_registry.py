"""Registry shape contract — keys, members, and `.approval` metadata."""

from __future__ import annotations

from jac.tools import TOOL_REGISTRY
from jac.tools.types import RiskLevel, ToolApprovalMeta


def test_registry_keys() -> None:
    assert set(TOOL_REGISTRY.keys()) == {
        "filesystem",
        "filesystem:read",
        "shell",
        "shell:read",
        "tasks",
    }


def test_filesystem_group_members() -> None:
    names = {fn.__name__ for fn in TOOL_REGISTRY["filesystem"]}
    assert names == {
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "grep_files",
    }


def test_filesystem_read_is_subset_of_filesystem() -> None:
    full = {fn.__name__ for fn in TOOL_REGISTRY["filesystem"]}
    read = {fn.__name__ for fn in TOOL_REGISTRY["filesystem:read"]}
    assert read <= full
    assert "write_file" not in read
    assert "edit_file" not in read


def test_shell_group_members() -> None:
    names = {fn.__name__ for fn in TOOL_REGISTRY["shell"]}
    assert names == {"run_shell", "run_shell_background", "read_process_output"}


def test_tasks_group_members() -> None:
    names = {fn.__name__ for fn in TOOL_REGISTRY["tasks"]}
    assert names == {"add_task", "update_task", "complete_task", "list_tasks"}


def test_every_tool_carries_approval_metadata() -> None:
    for group, fns in TOOL_REGISTRY.items():
        for fn in fns:
            meta = getattr(fn, "approval", None)
            assert isinstance(meta, ToolApprovalMeta), f"{group}/{fn.__name__}"
            assert isinstance(meta.risk_level, RiskLevel)
            assert callable(meta.description_fn)
