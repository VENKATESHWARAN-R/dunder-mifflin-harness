"""Filesystem tools — read, write, edit, list."""

from pygemini.tools.base import BaseTool
from pygemini.tools.filesystem.edit_file import EditFileTool
from pygemini.tools.filesystem.list_directory import ListDirectoryTool
from pygemini.tools.filesystem.read_file import ReadFileTool
from pygemini.tools.filesystem.read_many_files import ReadManyFilesTool
from pygemini.tools.filesystem.write_file import WriteFileTool

__all__ = [
    "EditFileTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "ReadManyFilesTool",
    "WriteFileTool",
    "get_filesystem_tools",
]


def get_filesystem_tools() -> list[BaseTool]:
    """Return all filesystem tool instances."""
    return [
        ListDirectoryTool(),
        ReadFileTool(),
        ReadManyFilesTool(),
        WriteFileTool(),
        EditFileTool(),
    ]
