"""System prompt construction for PyGeminiCLI."""

from __future__ import annotations

TOOL_USAGE_GUIDELINES = """\
## Tool Usage Guidelines
- Use tools proactively to help the user accomplish their goals.
- Prefer reading files before modifying them to understand context.
- For file modifications, use edit_file (search-and-replace) when making targeted changes, write_file for creating new files or full rewrites.
- When running shell commands, always consider the working directory and potential side effects.
- If you're unsure about a destructive operation, use ask_user to confirm with the user.
- Chain tool calls logically: read → understand → modify → verify.
"""

BASE_SYSTEM_PROMPT = """\
You are PyGeminiCLI, an AI coding assistant running in the user's terminal.
You help with software engineering tasks by reading files, writing code, running commands, and searching the web.

## Core Behavior
- Be concise and direct in your responses.
- Use tools to interact with the filesystem and execute commands.
- Always read files before modifying them to understand existing code.
- Confirm destructive operations before executing them.
- When you encounter errors, diagnose the root cause rather than retrying blindly.

{tool_guidelines}

## Available Tools
{tool_list}

{context_content}
"""


def build_system_prompt(
    context_content: str = "",
    tool_names: list[str] | None = None,
) -> str:
    """Build the complete system prompt with injected context and tool info.

    Args:
        context_content: Content from GEMINI.md files and memory, pre-formatted.
        tool_names: List of available tool names for reference.

    Returns:
        Complete system prompt string.
    """
    tool_list = ""
    if tool_names:
        tool_list = "Available tools: " + ", ".join(tool_names)

    context_section = ""
    if context_content:
        context_section = f"## Project Context\n{context_content}"

    return BASE_SYSTEM_PROMPT.format(
        tool_guidelines=TOOL_USAGE_GUIDELINES,
        tool_list=tool_list,
        context_content=context_section,
    ).strip()
