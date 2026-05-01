"""Conversation history management for PyGeminiCLI.

Manages the ordered list of user, model, and tool-result messages in the
format expected by the google-genai Python SDK (``types.Content``).
"""

from __future__ import annotations

from typing import Any

from google.genai import types


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a simple heuristic.

    Uses the rough approximation of 1 token per 4 characters.  This avoids
    a hard dependency on tiktoken (which does not support Gemini's tokenizer)
    and can be upgraded later to use the Gemini ``count_tokens`` API.

    Args:
        text: The text whose token count should be estimated.

    Returns:
        Estimated token count as an integer.
    """
    return len(text) // 4


class ConversationHistory:
    """Ordered conversation history for the Gemini API.

    Messages are stored as ``types.Content`` objects with role="user" or
    role="model". Function call/response pairs are added as model and user
    turns respectively (the SDK convention).
    """

    def __init__(self) -> None:
        self._messages: list[types.Content] = []

    # -- Adding messages -----------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Append a user text message."""
        self._messages.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            )
        )

    def add_model_response(self, parts: list[types.Part]) -> None:
        """Append a model response with the given parts (text and/or function calls)."""
        self._messages.append(
            types.Content(role="model", parts=parts)
        )

    def add_tool_results(
        self,
        function_calls: list[Any],
        function_responses: list[dict[str, Any]],
    ) -> None:
        """Add function call results back into the history.

        The model's function calls are recorded as a model turn, and the
        corresponding responses are recorded as a user turn — this is the
        convention the Gemini API uses for multi-turn function calling.

        Args:
            function_calls: List of FunctionCallData from the model response.
            function_responses: List of dicts with 'name' and 'response' keys.
        """
        # Model turn: the function call parts
        call_parts = []
        for fc in function_calls:
            call_parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=fc.name,
                        args=fc.args,
                    )
                )
            )
        if call_parts:
            self._messages.append(
                types.Content(role="model", parts=call_parts)
            )

        # User turn: the function response parts
        response_parts = []
        for fr in function_responses:
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fr["name"],
                        response=fr["response"],
                    )
                )
            )
        if response_parts:
            self._messages.append(
                types.Content(role="user", parts=response_parts)
            )

    # -- Access ---------------------------------------------------------------

    def get_messages(self) -> list[types.Content]:
        """Return the full message history."""
        return list(self._messages)

    def get_turn_count(self) -> int:
        """Return the number of user turns in the history."""
        return sum(1 for m in self._messages if m.role == "user")

    def clear(self) -> None:
        """Clear all history."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    # -- Token / size helpers -------------------------------------------------

    @property
    def message_count(self) -> int:
        """Total number of messages (all roles) in the history."""
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """Estimated total tokens across all messages.

        Iterates over every part of every message, collects the text
        representation, and delegates to :func:`estimate_tokens`.
        """
        total_chars = 0
        for message in self._messages:
            if message.parts is None:
                continue
            for part in message.parts:
                # Text parts
                if part.text is not None:
                    total_chars += len(part.text)
                # Function-call parts: count the name and serialised args
                if part.function_call is not None:
                    fc = part.function_call
                    if fc.name:
                        total_chars += len(fc.name)
                    if fc.args:
                        total_chars += len(str(fc.args))
                # Function-response parts: count the name and serialised response
                if part.function_response is not None:
                    fr = part.function_response
                    if fr.name:
                        total_chars += len(fr.name)
                    if fr.response:
                        total_chars += len(str(fr.response))
        return estimate_tokens("x" * total_chars)

    # -- Slice / mutation helpers ---------------------------------------------

    def get_messages_range(
        self, start: int, end: int | None = None
    ) -> list[types.Content]:
        """Return a slice of the message history.

        Args:
            start: Inclusive start index (negative indices are supported).
            end: Exclusive end index.  ``None`` means the end of the list.

        Returns:
            A new list containing the requested slice.
        """
        return list(self._messages[start:end])

    def replace_messages(
        self,
        start: int,
        end: int,
        replacement_messages: list[types.Content],
    ) -> None:
        """Replace ``messages[start:end]`` with *replacement_messages*.

        Used by the compressor to substitute a run of old messages with a
        shorter summary message.

        Args:
            start: Inclusive start index of the range to replace.
            end: Exclusive end index of the range to replace.
            replacement_messages: Messages that will occupy the vacated slice.

        Raises:
            IndexError: If *start* or *end* are outside the valid range for
                the current history.
        """
        n = len(self._messages)
        if start < 0 or start > n:
            raise IndexError(
                f"start index {start} is out of range for history of length {n}"
            )
        if end < start or end > n:
            raise IndexError(
                f"end index {end} is out of range for history of length {n} "
                f"(start={start})"
            )
        self._messages[start:end] = replacement_messages

    def get_recent_messages(self, n: int) -> list[types.Content]:
        """Return the last *n* messages.

        Args:
            n: Number of most-recent messages to return.  If *n* is greater
               than the current history length the full history is returned.

        Returns:
            A new list containing up to *n* messages.
        """
        if n <= 0:
            return []
        return list(self._messages[-n:])
