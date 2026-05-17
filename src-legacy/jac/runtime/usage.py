"""Helpers for reading Pydantic AI usage objects and computing per-attempt deltas."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class UsageDelta:
    tokens_in: int
    tokens_out: int
    requests: int
    tool_calls: int
    duration_ms: int


def snapshot_usage(usage_obj: Any | None) -> tuple[int, int, int, int]:
    """Read input_tokens / output_tokens / requests / tool_calls, or zeros."""
    if usage_obj is None:
        return (0, 0, 0, 0)
    return (
        int(getattr(usage_obj, "input_tokens", 0) or 0),
        int(getattr(usage_obj, "output_tokens", 0) or 0),
        int(getattr(usage_obj, "requests", 0) or 0),
        int(getattr(usage_obj, "tool_calls", 0) or 0),
    )


def vector_sub(
    total: tuple[int, int, int, int], sub: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Component-wise subtraction with floor at zero (usage counters are monotone)."""
    return cast(
        tuple[int, int, int, int],
        tuple(max(0, a - b) for a, b in zip(total, sub, strict=True)),
    )


def sum_child_usage(rows: list[Any]) -> tuple[int, int, int, int]:
    """Sum tokens_in/out/requests/tool_calls from attempt rows or nodes."""
    tin = tout = req = tc = 0
    for r in rows:
        tin += int(getattr(r, "tokens_in", 0) or 0)
        tout += int(getattr(r, "tokens_out", 0) or 0)
        req += int(getattr(r, "requests", 0) or 0)
        tc += int(getattr(r, "tool_calls", 0) or 0)
    return (tin, tout, req, tc)


def usage_delta_from_snapshots(
    before: tuple[int, int, int, int], after_usage: Any | None, started_at: float
) -> UsageDelta:
    """Compute (after - before) tuple-wise and elapsed ms."""
    after = snapshot_usage(after_usage)
    d0, d1, d2, d3 = vector_sub(after, before)
    return UsageDelta(
        tokens_in=d0,
        tokens_out=d1,
        requests=d2,
        tool_calls=d3,
        duration_ms=int((perf_counter() - started_at) * 1000),
    )
