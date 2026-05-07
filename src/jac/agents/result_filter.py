"""Tool result interception layer for large outputs."""

from __future__ import annotations

import functools
from typing import Any

from jac.tools.cache import ToolResultCache
from jac.tools.summarize import Summariser, estimate_tokens
from jac.tools.types import SummarizedToolResult

SUMMARISE_THRESHOLD_TOKENS = 4_000


def make_result_filter_wrapper(
    fn: Any,
    cache: ToolResultCache,
    summariser: Summariser,
    threshold_tokens: int = SUMMARISE_THRESHOLD_TOKENS,
) -> Any:
    """Wrap a tool so large results are summarized and cached by handle."""

    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> Any:
        result = await fn(**kwargs)
        try:
            serialized = result.model_dump_json()
        except AttributeError:
            return result

        token_estimate = estimate_tokens(serialized)
        if token_estimate <= threshold_tokens:
            return result

        summary = await summariser(serialized, _hint_for(fn.__name__))
        handle = cache.store(serialized)
        return SummarizedToolResult(
            summary=summary,
            summarized=True,
            original_tokens=token_estimate,
            full_result_handle=handle,
            note=(
                f"Summarized by AI from ~{token_estimate} tokens. "
                f"Call fetch_full_result(handle='{handle}') for full output."
            ),
        )

    wrapped_any = wrapper
    wrapped_any.approval = getattr(fn, "approval", None)
    return wrapped_any


def _hint_for(tool_name: str) -> str:
    if tool_name in {"run_shell", "run_shell_background"}:
        return "shell command output"
    if tool_name.startswith("read_file"):
        return "file contents"
    if tool_name.startswith("grep") or tool_name.startswith("search"):
        return "search results"
    return "tool output"
