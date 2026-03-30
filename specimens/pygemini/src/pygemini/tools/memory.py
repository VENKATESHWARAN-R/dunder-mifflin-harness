"""save_memory tool — persist facts and preferences across sessions."""

from __future__ import annotations

import asyncio

from pygemini.context.memory_store import MemoryStore
from pygemini.tools.base import BaseTool, ToolResult


class SaveMemoryTool(BaseTool):
    """Save a fact, preference, or piece of information to persistent memory.

    Use this to remember things about the user, project, or preferences
    across sessions.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # BaseTool contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save a fact, preference, or piece of information to persistent memory. "
            "Use this to remember things about the user, project, or preferences "
            "across sessions."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or information to remember.",
                },
            },
            "required": ["content"],
        }

    def validate_params(self, params: dict) -> str | None:
        """Require a non-empty content string."""
        content = params.get("content")
        if content is None:
            return "Missing required parameter: content"
        if not isinstance(content, str) or not content.strip():
            return "content must be a non-empty string"
        return None

    def get_description(self, params: dict) -> str:
        """Return a short human-readable summary, truncated to 80 chars."""
        content = params.get("content", "")
        prefix = "Save memory: "
        max_content = 80 - len(prefix)
        truncated = content[:max_content] if len(content) > max_content else content
        return f"{prefix}{truncated}"

    def should_confirm(self, params: dict) -> None:
        """No confirmation needed — saving memory is non-destructive."""
        return None

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        """Persist the memory entry and return a confirmation result."""
        content: str = params["content"]
        self._memory_store.save(content)
        return ToolResult(
            llm_content=f"Memory saved: {content}",
            display_content="[green]Memory saved[/green]",
        )
