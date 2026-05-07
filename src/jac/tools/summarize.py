"""Token estimation and Scout summarizer for large tool outputs."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from jac.config import Settings

if TYPE_CHECKING:
    from jac.runtime.session import SessionState
    from jac.state import StateStore


def estimate_tokens(text: str) -> int:
    """Estimate token count from word count."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def _extract_text(response: ModelResponse) -> str:
    text_parts = [part.content for part in response.parts if isinstance(part, TextPart)]
    return "\n".join(text_parts).strip() or "[no summary content returned]"


Summariser = Callable[..., Awaitable[str]]


def build_scout_summariser(
    settings: Settings,
    *,
    state: StateStore | None = None,
    session: SessionState | None = None,
) -> Summariser:
    """Build a summarizer that uses the Scout-tier direct model API.

    When ``state`` and ``session`` are set, each summarization is recorded as a
    ``direct_llm`` attempt and merged into the parent agent's ``ctx.usage`` when
    ``ctx`` is passed from the result-filter wrapper.
    """

    async def _summarise(content: str, hint: str, ctx: Any = None) -> str:
        selection = settings.resolve_model_selection(tier="scout")
        prompt = (
            f"Summarise the following {hint} for a coding agent. "
            "Preserve file paths, error messages, command names, and exit codes verbatim. "
            "Drop noise. Aim for <= 400 words.\n\n---\n\n"
            f"{content}"
        )
        attempt_id: str | None = None
        started = time.perf_counter()
        if state is not None and session is not None:
            row = await state.attempts.create(
                run_id=session.run_id,
                role="summariser",
                model=selection.model_ref,
                tier="scout",
                parent_attempt_id=session.active_attempt_id,
                call_type="direct_llm",
            )
            attempt_id = row.attempt_id
        try:
            response = await model_request(
                selection.model_ref,
                [ModelRequest(parts=[UserPromptPart(content=prompt)])],
            )
            if ctx is not None and getattr(ctx, "usage", None) is not None:
                try:
                    ctx.usage.incr(response.usage)
                except (TypeError, ValueError, AttributeError):
                    pass
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if state is not None and attempt_id is not None:
                u = response.usage
                await state.attempts.update_usage(
                    attempt_id,
                    tokens_in=int(u.input_tokens or 0),
                    tokens_out=int(u.output_tokens or 0),
                    requests=int(u.requests or 0),
                    tool_calls=0,
                    duration_ms=elapsed_ms,
                )
                await state.attempts.update_status(attempt_id, "passed")
            return _extract_text(response)
        except Exception as exc:  # noqa: BLE001
            if state is not None and attempt_id is not None:
                await state.attempts.update_status(attempt_id, "failed")
            return f"[summariser failed: {exc}; raw output omitted]"

    return _summarise
