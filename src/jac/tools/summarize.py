"""Token estimation and Scout summarizer for large tool outputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from jac.config import Settings


def estimate_tokens(text: str) -> int:
    """Estimate token count from word count."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


Summariser = Callable[[str, str], Awaitable[str]]


def _extract_text(response: ModelResponse) -> str:
    text_parts = [part.content for part in response.parts if isinstance(part, TextPart)]
    return "\n".join(text_parts).strip() or "[no summary content returned]"


def build_scout_summariser(settings: Settings) -> Summariser:
    """Build a summarizer that uses the Scout-tier direct model API."""

    async def _summarise(content: str, hint: str) -> str:
        selection = settings.resolve_model_selection(tier="scout")
        prompt = (
            f"Summarise the following {hint} for a coding agent. "
            "Preserve file paths, error messages, command names, and exit codes verbatim. "
            "Drop noise. Aim for <= 400 words.\n\n---\n\n"
            f"{content}"
        )
        try:
            response = await model_request(
                selection.model_ref,
                [ModelRequest(parts=[UserPromptPart(content=prompt)])],
            )
            return _extract_text(response)
        except Exception as exc:  # noqa: BLE001
            return f"[summariser failed: {exc}; raw output omitted]"

    return _summarise
