"""Agent tool registry and local tool helpers."""

from jac.tools.filesystem import (
    edit_file,
    grep_files,
    list_directory,
    read_file,
    search_files,
    write_file,
)
from jac.tools.shell import (
    list_processes,
    read_process_output,
    run_shell,
    run_shell_background,
)
from jac.tools.types import ToolFn

# Maps allowed_tools entry names (from agent_configs) to their tool functions.
# config_loader resolves these when building an agent.
#
# Groups:
#   "filesystem"        — full read + write access
#   "filesystem:read"   — read-only subset (planner, evaluator)
#   "shell"             — blocking + background execution + process management
#   "shell:read"        — process inspection only (no execution)
TOOL_REGISTRY: dict[str, list[ToolFn]] = {
    "filesystem": [read_file, write_file, edit_file, list_directory, search_files, grep_files],
    "filesystem:read": [read_file, list_directory, search_files, grep_files],
    "shell": [run_shell, run_shell_background, list_processes, read_process_output],
    "shell:read": [list_processes, read_process_output],
}

__all__ = [
    "TOOL_REGISTRY",
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "grep_files",
    "run_shell",
    "run_shell_background",
    "list_processes",
    "read_process_output",
]
