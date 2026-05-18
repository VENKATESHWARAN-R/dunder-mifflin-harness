"""Tool registry — the single import surface the factory reads.

Keys are the same strings that show up in `agent_configs.allowed_tools`
(plus narrowed `:read` variants). Each group resolves to a list of async
tool functions, every one carrying a `.approval: ToolApprovalMeta`.

MCP servers are addressed separately by `mcp:<server_id>` entries in
`allowed_tools` and are not in this registry — they're resolved against
the `mcp_servers` table at factory time.
"""

from __future__ import annotations

from jac.tools.filesystem import (
    edit_file,
    grep_files,
    list_directory,
    read_file,
    search_files,
    write_file,
)
from jac.tools.shell import read_process_output, run_shell, run_shell_background
from jac.tools.tasks import add_task, complete_task, list_tasks, update_task
from jac.tools.types import ToolFn

TOOL_REGISTRY: dict[str, list[ToolFn]] = {
    "filesystem": [
        read_file,
        write_file,
        edit_file,
        list_directory,
        search_files,
        grep_files,
    ],
    "filesystem:read": [
        read_file,
        list_directory,
        search_files,
        grep_files,
    ],
    "shell": [
        run_shell,
        run_shell_background,
        read_process_output,
    ],
    "shell:read": [
        read_process_output,
    ],
    "tasks": [
        add_task,
        update_task,
        complete_task,
        list_tasks,
    ],
}


__all__ = ["TOOL_REGISTRY"]
