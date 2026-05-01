"""BaseTool abstract base class, ToolResult, and ToolConfirmation dataclasses.

This module defines the contract that every tool in PyGeminiCLI must follow.
Tools are registered with the ToolRegistry and exposed to the Gemini model as
function declarations. The model calls tools by name; the AgentLoop validates
parameters, optionally prompts for user confirmation, then executes.

Key types:
    ToolResult         — Dual-output (LLM content + display content) from execution.
    ToolConfirmation   — Pre-execution confirmation prompt shown to the user.
    BaseTool           — ABC that all concrete tools must subclass.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Dual-output result from tool execution.

    Attributes:
        llm_content: Raw content sent back to the LLM for reasoning.
        display_content: Formatted content shown to the user (Rich markup).
        is_error: Whether this result represents an error.
    """

    llm_content: str
    display_content: str = ""
    is_error: bool = False

    def __repr__(self) -> str:
        error_tag = " ERROR" if self.is_error else ""
        llm_preview = (
            self.llm_content[:60] + "..."
            if len(self.llm_content) > 60
            else self.llm_content
        )
        return f"ToolResult({llm_preview!r}{error_tag})"


@dataclass
class ToolConfirmation:
    """Details shown to user before confirming a tool execution.

    Attributes:
        description: What the tool will do (human-readable).
        details: Structured details (files, commands, etc.) for the confirmation prompt.
    """

    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.details:
            detail_keys = ", ".join(self.details.keys())
            return f"ToolConfirmation({self.description!r}, details=[{detail_keys}])"
        return f"ToolConfirmation({self.description!r})"


class BaseTool(ABC):
    """Abstract base for all tools. Mirrors Gemini CLI's BaseTool contract.

    Every tool must define:
    - What it's called (name)
    - What it does (description for the model)
    - What parameters it accepts (JSON Schema for function declaration)
    - Whether it needs user confirmation
    - How to execute it
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in Gemini API function declarations."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description sent to the model so it knows when/how to use this tool."""
        ...

    @property
    @abstractmethod
    def parameter_schema(self) -> dict:
        """JSON Schema defining accepted parameters."""
        ...

    def validate_params(self, params: dict) -> str | None:
        """Validate parameters. Returns error message or None if valid."""
        return None

    def get_description(self, params: dict) -> str:
        """Human-readable description of what will happen with these specific params."""
        return self.description

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        """Return confirmation details if user approval is needed, else None."""
        return None  # Default: no confirmation needed (read-only tools)

    @abstractmethod
    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        """Execute the tool. Core implementation."""
        ...

    def to_function_declaration(self) -> dict:
        """Export as Gemini API FunctionDeclaration format.

        Returns a dict compatible with the google-genai Python SDK's
        ``types.Tool(function_declarations=[...])`` interface.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameter_schema,
        }
