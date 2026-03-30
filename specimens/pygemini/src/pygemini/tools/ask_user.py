"""ask_user tool — asks the user a question to get clarification or additional information."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pygemini.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from pygemini.core.events import EventEmitter


class AskUserTool(BaseTool):
    """Ask the user a question to get clarification or additional information."""

    def __init__(self, event_emitter: EventEmitter) -> None:
        """Initialize AskUserTool with an event emitter for user interaction.

        Args:
            event_emitter: EventEmitter instance for communicating with the CLI layer.
        """
        self._event_emitter = event_emitter

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question to get clarification or additional information. "
            "Use when you need more details to complete a task."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
        }

    def validate_params(self, params: dict) -> str | None:
        """Validate that question is present and non-empty."""
        if "question" not in params:
            return "Missing required parameter: question"
        if not isinstance(params["question"], str):
            return "question must be a string"
        if not params["question"].strip():
            return "question must not be empty"
        return None

    def get_description(self, params: dict) -> str:
        """Return a human-readable description of what will happen."""
        question = params.get("question", "?")
        # Truncate to 100 chars as per specification
        if len(question) > 100:
            question = question[:97] + "..."
        return f"Ask user: {question}"

    def should_confirm(self, params: dict) -> None:
        """No confirmation needed for ask_user tool."""
        return None

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        """Execute by asking the user via the event emitter.

        Args:
            params: Must contain 'question' key with the user question.
            abort_signal: Optional abort signal (not used for this tool).

        Returns:
            ToolResult with user response (confirmed/declined).
        """
        question = params["question"]

        # Use the event emitter's request_confirmation to ask the question
        # This provides a yes/no interface for the user
        approved = await self._event_emitter.request_confirmation(
            description=question,
            details={"type": "question"},
        )

        # Build response based on user's answer
        response = "User confirmed" if approved else "User declined"

        return ToolResult(
            llm_content=f"User response: {response}",
            display_content=f"Asked user: {question}\n[cyan]{response}[/cyan]",
        )
