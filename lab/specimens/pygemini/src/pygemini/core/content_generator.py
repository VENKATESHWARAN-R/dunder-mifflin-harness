"""Gemini API wrapper for streaming content generation.

Wraps the ``google-genai`` Python SDK to provide an async streaming
interface that yields :class:`StreamChunk` objects containing either
text fragments or function-call requests extracted from the model
response.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from pygemini.core.config import Config

__all__ = [
    "FunctionCallData",
    "StreamChunk",
    "ContentGenerator",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FunctionCallData:
    """A single function-call request extracted from the model stream."""

    name: str
    args: dict[str, Any]
    id: str


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One chunk yielded while streaming a model response.

    Exactly one of *text* or *function_calls* will be set per chunk.
    """

    text: str | None = None
    function_calls: list[FunctionCallData] | None = None


# ---------------------------------------------------------------------------
# ContentGenerator
# ---------------------------------------------------------------------------


class ContentGenerator:
    """Thin async wrapper around the Gemini ``generate_content_stream`` API.

    Parameters
    ----------
    config:
        Application configuration carrying the API key, model names, etc.
    """

    def __init__(self, config: Config) -> None:
        if not config.api_key:
            raise ValueError(
                "A Gemini API key is required. Set GEMINI_API_KEY or "
                "configure api_key in settings.toml."
            )
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model
        self._fallback_model = config.fallback_model
        logger.debug("ContentGenerator initialised (model=%s)", self._model)

    # -- public API ----------------------------------------------------------

    async def generate_stream(
        self,
        history: list[types.Content],
        tools: list[dict[str, Any]],
        system_instruction: str,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a model response, yielding :class:`StreamChunk` objects.

        Parameters
        ----------
        history:
            Conversation history as a list of ``types.Content`` objects.
        tools:
            Tool declarations (list of dicts with ``function_declarations``).
        system_instruction:
            The system-level instruction prepended to the conversation.

        Yields
        ------
        StreamChunk
            Chunks containing either ``text`` or ``function_calls``.
        """
        gen_config = self._build_config(system_instruction, tools)

        try:
            async for chunk in self._stream(self._model, history, gen_config):
                yield chunk
        except genai.errors.APIError as exc:
            if _is_rate_limit(exc) and self._fallback_model != self._model:
                logger.warning(
                    "Rate-limited on %s — retrying with fallback %s",
                    self._model,
                    self._fallback_model,
                )
                async for chunk in self._stream(
                    self._fallback_model, history, gen_config
                ):
                    yield chunk
            else:
                logger.error("Gemini API error: %s", exc)
                raise

    # -- internals -----------------------------------------------------------

    def _build_config(
        self,
        system_instruction: str,
        tools: list[dict[str, Any]],
    ) -> types.GenerateContentConfig:
        """Construct a ``GenerateContentConfig`` for the API call."""
        tool_objects: list[types.Tool] | None = None
        if tools:
            tool_objects = [
                types.Tool(function_declarations=t["function_declarations"])
                for t in tools
            ]

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tool_objects,
            temperature=1.0,
        )

    async def _stream(
        self,
        model: str,
        history: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Call the SDK's streaming endpoint and translate chunks."""
        response = self._client.models.generate_content_stream(
            model=model,
            contents=history,
            config=config,
        )

        for server_chunk in response:
            if not server_chunk.candidates:
                continue

            parts = server_chunk.candidates[0].content.parts
            if not parts:
                continue

            fn_calls: list[FunctionCallData] = []

            for part in parts:
                if part.text:
                    yield StreamChunk(text=part.text)
                elif part.function_call:
                    fc = part.function_call
                    fn_calls.append(
                        FunctionCallData(
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                            id=uuid.uuid4().hex,
                        )
                    )

            if fn_calls:
                yield StreamChunk(function_calls=fn_calls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_rate_limit(exc: genai.errors.APIError) -> bool:
    """Return ``True`` if the exception indicates a 429 rate-limit error."""
    return getattr(exc, "code", None) == 429
