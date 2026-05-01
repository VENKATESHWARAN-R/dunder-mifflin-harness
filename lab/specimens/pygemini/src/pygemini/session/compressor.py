"""Conversation compressor — summarizes older history via LLM.

When conversation history grows beyond a configurable threshold, the
compressor extracts the oldest messages, sends them to a cheaper model
for summarization, and returns a :class:`CompressionResult` that the
caller (:class:`AgentLoop`) can use to replace those messages with a
single summary turn.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from pygemini.core.config import Config
from pygemini.core.history import ConversationHistory

__all__ = [
    "CompressionResult",
    "ConversationCompressor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Result of compressing conversation history."""

    summary: str
    """The compressed summary text."""

    original_message_count: int
    """How many messages were compressed."""

    compressed_token_estimate: int
    """Rough token count of summary (words * 1.3)."""


# ---------------------------------------------------------------------------
# Summarisation prompt
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You are summarizing a conversation between a user and an AI assistant. "
    "Preserve all important details: decisions made, files modified, errors "
    "encountered, user preferences, and task progress. Be concise but comprehensive."
)

_SUMMARY_TEMPLATE = """\
Summarize the following conversation transcript. Keep all key facts, \
decisions, file paths, error messages, and user preferences. Omit \
pleasantries and filler.

Conversation to summarize:
{transcript}"""

# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

_TOOL_OUTPUT_TRUNCATE = 200


class ConversationCompressor:
    """Compresses conversation history by summarizing older messages.

    When history grows too large, the compressor:

    1. Takes the oldest N messages (keeping the most recent ones intact).
    2. Sends them to the LLM with a summarization prompt.
    3. Replaces those messages with a single "summary" message.

    The caller (:class:`AgentLoop`) is responsible for actually modifying
    the history — this class only *produces* the summary.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # Compression thresholds
        self._max_history_messages: int = 50
        self._keep_recent: int = 10
        # Use the cheaper / faster model for summarisation
        self._summary_model: str = config.fallback_model

    # -- public API ----------------------------------------------------------

    def should_compress(self, history: ConversationHistory) -> bool:
        """Return ``True`` if history exceeds the compression threshold."""
        return len(history) > self._max_history_messages

    async def compress(self, history: ConversationHistory) -> CompressionResult:
        """Compress the older portion of the conversation history.

        Steps:
            1. Extract messages to compress (all except the most recent
               ``_keep_recent``).
            2. Format them as a readable conversation transcript.
            3. Send to LLM with a summarization prompt.
            4. Return the :class:`CompressionResult`.

        The caller is responsible for actually modifying the history.
        """
        messages = history.get_messages()
        split_idx = len(messages) - self._keep_recent
        if split_idx <= 0:
            # Nothing meaningful to compress
            return CompressionResult(
                summary="",
                original_message_count=0,
                compressed_token_estimate=0,
            )

        to_compress = messages[:split_idx]
        transcript = self._format_for_summarization(to_compress)
        prompt = self._build_summary_prompt(transcript)

        logger.info(
            "Compressing %d messages (keeping recent %d)",
            len(to_compress),
            self._keep_recent,
        )

        summary = await self._call_llm_for_summary(prompt)

        # Rough token estimate: ~1.3 tokens per word
        token_estimate = int(len(summary.split()) * 1.3)

        return CompressionResult(
            summary=summary,
            original_message_count=len(to_compress),
            compressed_token_estimate=token_estimate,
        )

    # -- internals -----------------------------------------------------------

    def _format_for_summarization(self, messages: list[types.Content]) -> str:
        """Convert history messages to a readable transcript for the LLM."""
        lines: list[str] = []

        for msg in messages:
            role = msg.role  # "user" or "model"
            if not msg.parts:
                continue

            for part in msg.parts:
                # Plain text
                if part.text:
                    label = "User" if role == "user" else "Assistant"
                    lines.append(f"{label}: {part.text}")

                # Function call (model asking to use a tool)
                elif part.function_call:
                    fc = part.function_call
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in (dict(fc.args).items() if fc.args else {}.items())
                    )
                    lines.append(f"Assistant: [Used tool {fc.name}({args_str})]")

                # Function response (tool result coming back)
                elif part.function_response:
                    fr = part.function_response
                    result_str = str(fr.response) if fr.response else ""
                    if len(result_str) > _TOOL_OUTPUT_TRUNCATE:
                        result_str = result_str[:_TOOL_OUTPUT_TRUNCATE] + "..."
                    lines.append(f"Tool({fr.name}): {result_str}")

        return "\n".join(lines)

    def _build_summary_prompt(self, transcript: str) -> str:
        """Build the prompt that asks the LLM to summarize the conversation."""
        return _SUMMARY_TEMPLATE.format(transcript=transcript)

    async def _call_llm_for_summary(self, prompt: str) -> str:
        """Make a single non-streaming API call to get the summary.

        The ``google-genai`` SDK's ``generate_content`` is synchronous, so
        we run it in an executor to avoid blocking the event loop.
        """
        if not self._config.api_key:
            logger.warning("No API key configured; returning fallback summary")
            return "(Summary unavailable — no API key)"

        loop = asyncio.get_running_loop()

        def _sync_call() -> str:
            client = genai.Client(api_key=self._config.api_key)
            response = client.models.generate_content(
                model=self._summary_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SUMMARY_SYSTEM,
                    temperature=0.3,  # deterministic summaries
                ),
            )
            return response.text or ""

        try:
            return await loop.run_in_executor(None, _sync_call)
        except Exception:
            logger.exception("LLM summarization failed; returning fallback")
            return "(Summary unavailable due to an API error)"
