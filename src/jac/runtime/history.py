"""Filters for Pydantic AI message history."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def filter_tool_noise(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip ToolCallPart/ToolReturnPart from a message list."""
    cleaned: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            user_parts = [
                part for part in message.parts if isinstance(part, UserPromptPart)
            ]
            if user_parts:
                cleaned.append(ModelRequest(parts=user_parts))
            continue
        if isinstance(message, ModelResponse):
            text_parts = [part for part in message.parts if isinstance(part, TextPart)]
            if text_parts:
                cleaned.append(ModelResponse(parts=text_parts))
    return cleaned
