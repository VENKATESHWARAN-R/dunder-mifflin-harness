# C6c — Universal `spawn_minion` + Tool-Layer Hardening

> **Status:** Ready for implementation
> **Last revised:** 2026-05-06
> **Depends on:** C6b (shipped 2026-05-06)
> **Roadmap entry:** [`docs/ROADMAP.md` §C6c](../ROADMAP.md#c6c--universal-spawn_minion--tool-layer-hardening)
> **Design record:** [`lab/brainstorm/2026-05-05-multi-agent-cast-final.md`](../../lab/brainstorm/2026-05-05-multi-agent-cast-final.md)
> **Contract touchpoints:** [`docs/contracts/TOOLS_CONTRACT.md`](../contracts/TOOLS_CONTRACT.md), [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md)

---

## Goals

1. Ship a universal **`spawn_minion`** tool — every native specialist (Scott, Pam, Jim, and future Dwight) can hand a single-shot brief to a depth-1 child agent. Cost rolls up via Pydantic AI's `usage=ctx.usage`.
2. Add **`fetch_full_result`** — per-run handle lookup for tool outputs that were summarised. In-memory only.
3. Add **`read_file_smart`** — pre-flight size check that gives the agent agency over how to ingest a large file (full / large-flag / metadata-only).
4. Wire a **tool result interception layer** — outputs above 4k tokens flow through a Scout summariser before reaching the agent. Original verbatim is cached for `fetch_full_result`.
5. Wire **per-tool timeouts** — agent-wide default of 180s, with overrides for `read_file*` (30s) and `spawn_minion` (caller-controlled, 240s default, 300s cap).
6. Emit **`MinionSpawned` / `MinionReturned`** events. (Renderer integration deferred to C7.)

## Non-goals (later components)

- **Out-of-process / sandboxed minions, deferred-tools backgrounding** — C26+.
- **`shell_exec` 180→300s auto-bump on retry** — needs retry-detection plumbing that doesn't exist until C9. Flagged as a follow-up; default remains 180s for shell.
- **Cost rollup to the `attempts` table.** Pydantic AI's `usage=ctx.usage` aggregates token counts in-memory; persisting them per attempt row is **C7**. Minion `attempts` rows still get `parent_attempt_id` and `call_type='minion'` so the call tree exists when C7 lands.
- **`/explore <path>` slash command.** Brainstorm lists it; defer until C9 lands the rest of the slash surface.
- **MinionSpawned/MinionReturned rendering in the CLI.** Emit-only in C6c; Rich panel work folds into C7's `/cost` UI.
- **Strategy-aware minion templates.** Dropped per the 2026-05-05 brainstorm — universal `spawn_minion` only.
- **Persisted `tool_results_cache` across runs.** Per-run, in-memory; resume starts empty.

---

## Key files and integration points

| File | What it is | C6c touches it? |
|---|---|---|
| `src/jac/tools/types.py` | Shared result types + approval metadata | Yes — add `SummarizedToolResult`, `FileReadSmartResult`; add `timeout_seconds` field on `ToolApprovalMeta` |
| `src/jac/tools/filesystem.py` | Filesystem tools | Yes — add `read_file_smart` |
| `src/jac/tools/__init__.py` | `TOOL_REGISTRY` | Yes — register `"filesystem:smart"` group entry that includes `read_file_smart` |
| `src/jac/tools/summarize.py` | **New file.** Token estimation + Scout summariser callable | Yes — create |
| `src/jac/tools/cache.py` | **New file.** `ToolResultCache` (per-run in-memory store) | Yes — create |
| `src/jac/agents/spawn.py` | **New file.** `make_spawn_minion_tool`, `make_fetch_full_result_tool`, `native_agent_extras` helper | Yes — create |
| `src/jac/agents/result_filter.py` | **New file.** `make_result_filter_wrapper(fn, cache, summariser, threshold_tokens)` | Yes — create |
| `src/jac/agents/base.py` | `config_loader` factory | Yes — accept `tool_result_cache`, `summariser`, thread into local-tool resolution; pass `tool_timeout=180` to `Agent(...)`; surface per-tool `asyncio.wait_for` from `ToolApprovalMeta.timeout_seconds` in the wrapper |
| `src/jac/agents/approval.py` | Approval wrapper | Yes — small change so the wrapper applies `timeout_seconds` via `asyncio.wait_for` *before* the result-filter wrapper runs |
| `src/jac/agents/tools.py` | `make_summon_jim_tool` | Yes — when summoning Jim, build Jim's agent with `native_agent_extras(...)` so Jim also has `spawn_minion` |
| `src/jac/agents/personas.py` | System prompts | Yes — extend Scott / Pam / Jim prompts with the spawn-minion + read_file_smart guidance |
| `src/jac/runtime/coordinator.py` | `RunCoordinator` | Yes — own a `ToolResultCache`, build a `Summariser`, thread both through `build_agent` / `submit_slash_run` / `make_summon_jim_tool`; pre-build native extras |
| `src/jac/runtime/events.py` | Typed events | Yes — add `MinionSpawned`, `MinionReturned` |
| `src/jac/runtime/session.py` | `SessionState` | Yes — small helper `current_role` (computed: most recent specialist active, falls back to `manager`) for parent-role attribution. Optional; spawn closure can capture parent role instead. |
| `docs/contracts/TOOLS_CONTRACT.md` | Locked contract | Yes — document `spawn_minion` signature, `SummarizedToolResult` shape, timeout policy, `ToolApprovalMeta.timeout_seconds`, `read_file_smart` thresholds, depth ≤ 1 invariant |
| `docs/contracts/STATE_SCHEMA.md` | Locked contract | Yes — extend the `attempts.call_type` discriminator note to include `'minion'`; bump revision date |
| `docs/dev/agents-layer.md` | Layer doc | Yes — document the spawn_minion factory, native_agent_extras, summariser, result-filter wrapper |
| `docs/dev/runtime-layer.md` | Layer doc | Yes — document MinionSpawned/MinionReturned and the cache wiring |
| `docs/ROADMAP.md` | Roadmap | Yes — flip C6c to Done with ship notes |
| `pyproject.toml` + `src/jac/__init__.py` | Version | Yes — bump (new tools are user-visible via agent behaviour) |
| `tests/test_c6c_spawn_minion_and_tool_hardening.py` | **New test file** | Yes — create |

No SQL migration required. `agent_configs.is_minion`, `parent_role`, `depth` already exist (C6 migration). `attempts.call_type` is free-text; the `'minion'` value is added by convention and documented in `STATE_SCHEMA.md`.

---

## Step-by-step implementation

### Step 1 — Token estimation + Scout summariser (`src/jac/tools/summarize.py`)

New file. Pure logic, no I/O at module import.

```python
"""Token estimation proxy + Scout summariser used by the result-filter wrapper."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from jac.config import Settings


def estimate_tokens(text: str) -> int:
    """Word-count proxy: tokens ≈ words × 1.3 (good enough for code-shaped text).

    Used for the 4k / 20k thresholds in `read_file_smart` and the result filter.
    Replace with an exact tokenizer later if measurements show drift.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


Summariser = Callable[[str, str], Awaitable[str]]
"""Async callable: (content_to_summarise, hint) -> summary_text.

`hint` is a short cue ("shell command output", "file contents") that biases
the model toward the right summary shape. Implementations must never raise —
on failure, return a degraded summary so the agent always gets a valid string.
"""


def build_scout_summariser(settings: Settings) -> Summariser:
    """Return a Summariser that calls Scout via `pydantic_ai.direct`.

    Lazy-imports `pydantic_ai.direct` so tests can stub the Summariser
    without exercising the real client.
    """
    async def _summarise(content: str, hint: str) -> str:
        from pydantic_ai.direct import model_request

        selection = settings.resolve_model_selection(tier="scout")
        prompt = (
            f"Summarise the following {hint} for a coding agent. "
            f"Preserve file paths, error messages, command names, and exit codes "
            f"verbatim. Drop noise. Aim for ≤ 400 words.\n\n---\n\n{content}"
        )
        try:
            response = await model_request(selection.model_ref, [prompt])
            return _extract_text(response)
        except Exception as exc:  # noqa: BLE001 — summariser must not propagate
            return f"[summariser failed: {exc}; raw output omitted]"

    return _summarise
```

> **`pydantic_ai.direct` API surface.** The brainstorm-validated entry point is `pydantic_ai.direct.model_request_sync`; we use the async sibling `model_request`. Confirm the exact signature against the installed pydantic_ai version during implementation; the brainstorm capability map is the source of truth. Ship this together with the C12-compaction expectation in mind — same primitive, same import.

### Step 2 — Per-run cache (`src/jac/tools/cache.py`)

New file. Tiny dict wrapper; lifetime = a single `RunCoordinator`.

```python
"""Per-run in-memory cache for tool result handles."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class ToolResultCache:
    """Bounded-by-the-run-only store for verbatim tool outputs.

    `fetch_full_result(handle)` looks up here. The cache is reset on resume
    (a stale handle from a prior run yields not-found; the agent re-runs the
    tool if it still wants the data).
    """
    _store: dict[str, str] = field(default_factory=dict)
    max_entries: int = 64
    """Soft cap. When exceeded, oldest entries are evicted FIFO."""

    def store(self, content: str) -> str:
        handle = uuid4().hex[:12]
        if len(self._store) >= self.max_entries:
            oldest = next(iter(self._store))
            self._store.pop(oldest, None)
        self._store[handle] = content
        return handle

    def fetch(self, handle: str) -> str | None:
        return self._store.get(handle)

    def clear(self) -> None:
        self._store.clear()
```

### Step 3 — New result types (`src/jac/tools/types.py`)

Append these models. Do **not** modify the existing `ToolResult` base.

```python
class SummarizedToolResult(BaseModel):
    """Envelope returned by the result-filter wrapper when a tool's output
    exceeds the summarisation threshold.

    The agent sees this in place of the original. The verbatim original is
    stashed in the per-run cache; `fetch_full_result(full_result_handle)`
    retrieves it if the agent decides it needs the raw bytes.
    """
    summary: str
    summarized: bool = True
    original_tokens: int
    full_result_handle: str
    note: str | None = None


class FileReadSmartResult(FileReadResult):
    """Result of `read_file_smart`. Inherits the FileReadResult payload
    (path, content, lines_total, lines_returned, truncated, warnings) and
    adds two flags so the agent can decide what to do next.
    """
    large: bool = False
    """True when the file is in the 4k–20k token range. `content` is full."""
    metadata_only: bool = False
    """True when the file exceeds 20k tokens. `content` is empty; metadata
    is in `lines_total`, `path`, `warnings`. Agent should use `read_lines`,
    `grep`, or `spawn_minion` to ingest selectively."""
```

Also add a `timeout_seconds: float | None = None` field on `ToolApprovalMeta`. Tools that need a non-default cap declare it here; the wrapper applies `asyncio.wait_for` accordingly.

```python
class ToolApprovalMeta(BaseModel):
    ...
    timeout_seconds: float | None = None
    """Per-tool timeout enforced by the wrapper via asyncio.wait_for.

    None means use the agent-wide default (Pydantic AI's tool_timeout).
    Set on read_file* (30s); spawn_minion enforces its own internally.
    """
```

### Step 4 — `read_file_smart` (`src/jac/tools/filesystem.py`)

New tool function. Pure pre-flight + read; no LLM call. Reuses `read_file`'s
existing error paths verbatim by delegating after the size decision.

```python
SMART_FULL_TOKENS = 4_000
SMART_LARGE_TOKENS = 20_000


async def read_file_smart(path: str) -> FileReadSmartResult:
    """Size-aware file read with an agency-preserving size hint.

    Behaviour:
      ≤ 4k tokens   — full content, large=False, metadata_only=False
      4k–20k tokens — full content, large=True, metadata_only=False
      > 20k tokens  — empty content, metadata_only=True, lines_total set,
                      warning hints to use read_lines/grep/spawn_minion
    """
    p = Path(path)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return FileReadSmartResult(
            status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path,
        )
    except OSError as exc:
        return FileReadSmartResult(status=ToolStatus.ERROR, error=str(exc), path=path)

    # Cheap upper-bound: 1 byte ≤ 1 token for code (in practice much less),
    # so file_bytes / 4 is a safe lower bound on token count without reading.
    estimated_tokens_bound = max(1, stat.st_size // 4)
    if estimated_tokens_bound > SMART_LARGE_TOKENS:
        try:
            line_count = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace"))
        except OSError:
            line_count = 0
        return FileReadSmartResult(
            path=path, content="", lines_total=line_count, lines_returned=0,
            truncated=True, large=True, metadata_only=True,
            warnings=[
                f"file is ~{estimated_tokens_bound} tokens; reading the full "
                "file would consume too much context. Use read_lines(path, "
                "start, end), grep_files for targeted lookups, or spawn_minion("
                "'summarise this file', tools=['filesystem:read']) to digest it.",
            ],
        )

    # Full read; classify based on actual word count.
    base = await read_file(path)
    if base.status != ToolStatus.OK:
        return FileReadSmartResult(**base.model_dump())
    actual_tokens = estimate_tokens(base.content)
    return FileReadSmartResult(
        **base.model_dump(),
        large=actual_tokens > SMART_FULL_TOKENS,
        metadata_only=False,
    )

setattr(read_file_smart, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda path, **_: f"Smart-read `{path}`",
    timeout_seconds=30.0,
))
```

Also set `timeout_seconds=30.0` on the existing `read_file`'s approval metadata so the policy is consistent.

> **Why a `read_lines(path, start, end)` reference in the warning when our `read_file` already accepts `start_line/end_line`?** The warning is a hint to the model; `read_file(path, start_line=N, end_line=M)` is exactly the operation we mean. Phrase it that way in the actual warning string to match the tool surface.

### Step 5 — Register `read_file_smart` in `TOOL_REGISTRY` (`src/jac/tools/__init__.py`)

Add it to the existing `"filesystem"` and `"filesystem:read"` groups so any agent with file tools picks it up. No new group entry — it's a peer of `read_file`, not a different category.

```python
TOOL_REGISTRY: dict[str, list[ToolFn]] = {
    "filesystem": [
        read_file, read_file_smart, write_file, edit_file,
        list_directory, search_files, grep_files,
    ],
    "filesystem:read": [
        read_file, read_file_smart, list_directory, search_files, grep_files,
    ],
    ...
}
```

### Step 6 — Result-filter wrapper (`src/jac/agents/result_filter.py`)

New file. The new outermost layer in the local-tool wrapping chain:

```
agent ── result_filter_wrapper ── approval_wrapper ── raw_tool
```

```python
"""Tool result interception: large outputs are summarised through a Scout call."""
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
    """Wrap an approval-gated tool with the result-filter layer.

    Behaviour:
      - Calls the wrapped tool.
      - JSON-serialises the return.
      - If estimated tokens ≤ threshold, returns the original unchanged.
      - Else, summarises via Scout, caches the verbatim string, and returns
        a SummarizedToolResult envelope.

    Result types other than `pydantic.BaseModel` (rare; today every tool
    returns a `ToolResult`) bypass the threshold check — they are returned
    unchanged.
    """
    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> Any:
        result = await fn(**kwargs)
        try:
            serialised = result.model_dump_json()
        except AttributeError:
            return result
        token_estimate = estimate_tokens(serialised)
        if token_estimate <= threshold_tokens:
            return result
        summary = await summariser(serialised, _hint_for(fn.__name__))
        handle = cache.store(serialised)
        return SummarizedToolResult(
            summary=summary,
            summarized=True,
            original_tokens=token_estimate,
            full_result_handle=handle,
            note=(
                f"Summarised by AI — {token_estimate} tokens compressed. "
                f"Call fetch_full_result('{handle}') for the verbatim output."
            ),
        )
    # Preserve the wrapped function's `.approval` so the chain still type-checks.
    wrapper.approval = getattr(fn, "approval", None)
    return wrapper


def _hint_for(tool_name: str) -> str:
    if tool_name in {"run_shell", "run_shell_background"}:
        return "shell command output"
    if tool_name.startswith("read_file"):
        return "file contents"
    if tool_name.startswith("grep") or tool_name.startswith("search"):
        return "search results"
    return "tool output"
```

### Step 7 — Approval wrapper applies per-tool timeout (`src/jac/agents/approval.py`)

Inside `wrapper`, just before each `await fn(...)` site, apply `asyncio.wait_for` when `meta.timeout_seconds is not None`. On timeout, return a typed `ToolResult(status=TIMEOUT, error="tool timed out after Ns")` so the model gets a coherent retry signal instead of an exception.

```python
async def _call_with_timeout(coro, timeout: float | None):
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return ToolResult(status=ToolStatus.TIMEOUT, error=f"tool timed out after {timeout}s")
```

The two call sites are the read-only fast path and the post-approval execution path. Apply at both. Where the wrapper currently calls `apply_edit(prepared_edit)` synchronously, leave it — file-write timeouts aren't part of the C6c policy.

### Step 8 — `spawn_minion` factory (`src/jac/agents/spawn.py`)

New file. Mirrors `make_summon_jim_tool` but parameterised. Uses `RunContext[None]` so `usage=ctx.usage` rolls cost up.

```python
"""Universal spawn_minion + fetch_full_result + native-extras helper."""
from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from jac.tools.cache import ToolResultCache
from jac.tools.summarize import Summariser
from jac.tools.types import (
    RiskLevel, ToolApprovalMeta, ToolResult, ToolStatus,
)


SPAWN_MINION_DEFAULT_TIMEOUT = 240
SPAWN_MINION_HARD_CAP = 300

# Tools we strip from a default tool inheritance for safety.
# `tools=None` means "inherit caller's allowed_tools minus these".
DESTRUCTIVE_TOOL_GROUPS = {"shell", "filesystem"}
"""Caller-side default: minion gets read-only equivalents instead.

A caller can still pass `tools=['shell']` explicitly to grant write/exec —
the bound is "subset of caller's allowed_tools", not "subset of safe tools".
"""

READONLY_FALLBACKS = {
    "shell": "shell:read",
    "filesystem": "filesystem:read",
}


def _safe_default_tools(parent_allowed: list[str]) -> list[str]:
    """When the caller doesn't pass `tools=`, return parent's allowed_tools
    with destructive groups swapped for their read-only equivalents.
    """
    safe: list[str] = []
    for t in parent_allowed:
        if t in DESTRUCTIVE_TOOL_GROUPS:
            safe.append(READONLY_FALLBACKS[t])
        else:
            safe.append(t)
    return safe


def make_spawn_minion_tool(
    *,
    state,
    settings,
    session,
    events,
    approval_policy,
    cache: ToolResultCache,
    summariser: Summariser,
    parent_role: str,
    parent_depth: int,
    parent_allowed_tools: list[str],
):
    """Return a `spawn_minion` tool bound to a specific parent agent."""

    async def spawn_minion(
        ctx: RunContext[None],
        task: str,
        tools: list[str] | None = None,
        tier: Literal["scout", "worker"] = "scout",
        timeout_sec: int = SPAWN_MINION_DEFAULT_TIMEOUT,
    ) -> str:
        """Run a single-shot child agent. Returns its final string output."""
        from jac.agents.base import config_loader
        from jac.agents.spawn import native_agent_extras  # avoid cycle
        from jac.runtime.events import (
            AttemptRecorded, MinionReturned, MinionSpawned,
        )

        if parent_depth >= 1:
            return "spawn_minion refused: minions cannot spawn further minions (depth ≤ 1)."

        timeout_sec = min(max(timeout_sec, 1), SPAWN_MINION_HARD_CAP)

        # Whitelist resolution: explicit tools must be a subset of caller's;
        # implicit (None) inherits caller's minus destructive groups.
        if tools is None:
            allowed_tools = _safe_default_tools(parent_allowed_tools)
        else:
            invalid = sorted(set(tools) - set(parent_allowed_tools))
            if invalid:
                return (
                    f"spawn_minion refused: requested tool(s) {invalid} are not in "
                    f"caller's allowed_tools {parent_allowed_tools}."
                )
            allowed_tools = list(tools)

        minion_role = f"minion:{uuid4().hex[:8]}"

        # Provision minion config row. is_minion=1, depth=parent+1.
        await state.agent_configs.create(
            run_id=session.run_id,
            role=minion_role,
            persona=None,
            display_name=None,
            is_minion=1,
            parent_role=parent_role,
            depth=parent_depth + 1,
            model_tier=tier,
            model_override=None,
            system_prompt=_MINION_SYSTEM_PROMPT,
            allowed_tools=allowed_tools,
        )

        await events.emit(MinionSpawned(
            parent_role=parent_role, minion_role=minion_role,
            task_summary=task[:120], tools=allowed_tools, tier=tier,
            depth=parent_depth + 1,
        ))

        selection = settings.resolve_model_selection(tier=tier)
        attempt = await state.attempts.create(
            run_id=session.run_id, role=minion_role, model=selection.model_ref,
            tier=tier, parent_attempt_id=session.active_attempt_id,
            call_type="minion",
        )
        await events.emit(AttemptRecorded(
            attempt_id=attempt.attempt_id, role=minion_role,
            call_type="minion", parent_attempt_id=attempt.parent_attempt_id,
        ))

        minion_agent = await config_loader(
            state=state, settings=settings, run_id=session.run_id,
            role=minion_role, events=events, approval_policy=approval_policy,
            tool_result_cache=cache, summariser=summariser,
            # NB: no native_agent_extras for minions — depth limit is enforced
            # by *not injecting* spawn_minion at this site.
        )

        started = perf_counter()
        success = False
        try:
            async with minion_agent:
                result = await asyncio.wait_for(
                    minion_agent.run(task, usage=ctx.usage,
                                     usage_limits=UsageLimits()),
                    timeout=timeout_sec,
                )
            output = result.output if isinstance(result.output, str) else str(result.output)
            success = True
            return output
        except asyncio.TimeoutError:
            return f"minion timed out after {timeout_sec}s"
        except Exception as exc:  # noqa: BLE001
            return f"minion failed: {exc}"
        finally:
            duration_ms = int((perf_counter() - started) * 1000)
            await state.attempts.update_status(
                attempt.attempt_id, "passed" if success else "failed",
            )
            await events.emit(MinionReturned(
                parent_role=parent_role, minion_role=minion_role,
                duration_ms=duration_ms, success=success,
            ))

    setattr(spawn_minion, "approval", ToolApprovalMeta(
        category="agent_spawn",
        risk_level=RiskLevel.MEDIUM,
        reversible=False,
        description_fn=lambda task, **_: f"Spawn minion for: {task[:80]}",
        timeout_seconds=None,  # internal asyncio.wait_for handles it
    ))
    return spawn_minion


_MINION_SYSTEM_PROMPT = """\
You are a single-purpose minion in JAC's agentic harness. You were spawned
by a parent agent to handle one focused task. Use only the tools you were
given, do the job, and return a concise final output. Do not call other
agents. Do not make plans. Be terse.
"""
```

`make_fetch_full_result_tool` is much smaller:

```python
def make_fetch_full_result_tool(cache: ToolResultCache):
    async def fetch_full_result(handle: str) -> ToolResult:
        """Return the verbatim original of a previously summarised tool output."""
        content = cache.fetch(handle)
        if content is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error=f"no cached result for handle '{handle}' "
                      "(cache is per-run; resume sessions reset it)",
            )
        # Wrap in a structured shape so the model sees JSON, not free text.
        class _FullResult(ToolResult):
            content: str = ""
        return _FullResult(content=content)

    setattr(fetch_full_result, "approval", ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda handle, **_: f"Fetch full result `{handle}`",
        timeout_seconds=None,
    ))
    return fetch_full_result
```

The native-extras helper bundles the two:

```python
def native_agent_extras(
    *, state, settings, session, events, approval_policy,
    cache: ToolResultCache, summariser: Summariser,
    parent_role: str, parent_depth: int, parent_allowed_tools: list[str],
) -> list:
    """Return the spawn_minion + fetch_full_result tools bound to a parent."""
    return [
        make_spawn_minion_tool(
            state=state, settings=settings, session=session, events=events,
            approval_policy=approval_policy, cache=cache, summariser=summariser,
            parent_role=parent_role, parent_depth=parent_depth,
            parent_allowed_tools=parent_allowed_tools,
        ),
        make_fetch_full_result_tool(cache),
    ]
```

> **Why pass `parent_allowed_tools` rather than read it from `agent_configs` inside the closure?** Two reasons. (1) The whitelist check happens on every `spawn_minion` call; doing a DB round-trip per call is wasteful when the value is fixed for the parent's lifetime. (2) `agent_configs.allowed_tools` is mutated by `/tier` and (later) hot-reload at C20; the closure captures the value as of the parent's build time, which is also the value the parent's tools were resolved against. Consistent.

### Step 9 — Factory accepts cache + summariser, applies tool_timeout (`src/jac/agents/base.py`)

Extend `config_loader`'s signature:

```python
async def config_loader(
    *, state, settings, run_id,
    role: str = "manager",
    output_type=None,
    events=None,
    approval_policy=None,
    model_settings=None,
    extra_tools=None,
    instructions_addendum: str | None = None,
    tool_result_cache: "ToolResultCache | None" = None,
    summariser: "Summariser | None" = None,
    tool_timeout_seconds: float = 180.0,
) -> Agent:
    ...
```

Behaviour:

1. `_resolve_local_tools` keeps wrapping each raw tool with `make_approval_wrapper` exactly as today.
2. **New step:** if `tool_result_cache` and `summariser` are both provided, additionally wrap each approval-wrapped tool with `make_result_filter_wrapper(...)`. If either is `None`, skip — current callers without observability still work.
3. Pass `tool_timeout=tool_timeout_seconds` to `Agent(...)`. Per-tool overrides come from `ToolApprovalMeta.timeout_seconds` enforced inside the approval wrapper (Step 7). Pydantic AI's `tool_timeout` is the backstop for tools that don't declare one.
4. `extra_tools` (which now includes `spawn_minion` + `fetch_full_result` for native specialists) flow through unchanged. `spawn_minion` enforces its own internal timeout, so wrapping it with the result-filter is fine — its return is a short string, well below 4k tokens.

> **`fetch_full_result` and the result filter.** The fetch tool returns the cached verbatim, which is by definition large. We must **not** put `fetch_full_result` through `make_result_filter_wrapper` — that would re-summarise the very thing the user asked for. Easiest gate: tag the tool's `ToolApprovalMeta` with `category="cache_passthrough"` and skip the result-filter wrap for that category. Update `_resolve_local_tools` accordingly. (Apply the same skip for `spawn_minion` returns since its output is already short — keeps the chain simple.)

### Step 10 — Coordinator owns the cache and the summariser (`src/jac/runtime/coordinator.py`)

`__init__` gains:

```python
from jac.tools.cache import ToolResultCache
from jac.tools.summarize import build_scout_summariser

self._tool_result_cache = ToolResultCache()
self._summariser = build_scout_summariser(settings)
```

`build_agent` switches from inline `extra_tools` to `native_agent_extras` for all native specialists:

```python
async def build_agent(self) -> Agent:
    if self.state is None:
        return self._build_fallback_agent()
    from jac.agents import config_loader
    from jac.agents.spawn import native_agent_extras
    from jac.agents.tools import make_summon_jim_tool

    role = self.session.config.role
    cfg = await self.state.agent_configs.get_by_run_and_role(
        self.session.run_id, role,
    )
    parent_allowed_tools = json.loads(cfg.allowed_tools) if cfg else []
    extras = native_agent_extras(
        state=self.state, settings=self.settings, session=self.session,
        events=self.events, approval_policy=self.approval_policy,
        cache=self._tool_result_cache, summariser=self._summariser,
        parent_role=role, parent_depth=0,
        parent_allowed_tools=parent_allowed_tools,
    )
    if role == "manager":
        extras.append(make_summon_jim_tool(
            self.state, self.settings, self.session, self.events,
            self.approval_policy,
        ))

    return await config_loader(
        state=self.state, settings=self.settings, run_id=self.session.run_id,
        role=role, events=self.events, approval_policy=self.approval_policy,
        extra_tools=extras,
        tool_result_cache=self._tool_result_cache,
        summariser=self._summariser,
        model_settings={"temperature": float(
            self.session.config.model_params.get("temperature", "0"),
        )},
    )
```

Apply the same `native_agent_extras` injection in `submit_slash_run` (Pam's `/plan` and Scott's `/init` need spawn_minion). For Scott in `/init` mode this finally unlocks the brainstorm's "spawn parallel module minions" path.

### Step 11 — `make_summon_jim_tool` injects spawn_minion into Jim (`src/jac/agents/tools.py`)

Today this function builds Jim with no extras. Add the same `native_agent_extras` block so Jim can spawn read-and-summarise minions when he hits a sprawling file. Cache + summariser come in as new function arguments — the coordinator already has them.

```python
def make_summon_jim_tool(
    state, settings, session, events, approval_policy,
    *, tool_result_cache, summariser,
):
    async def summon_jim(task: str) -> str:
        from jac.agents.base import config_loader
        from jac.agents.spawn import native_agent_extras
        ...

        jim_cfg = await state.agent_configs.get_by_run_and_role(session.run_id, "builder")
        jim_allowed = json.loads(jim_cfg.allowed_tools) if jim_cfg else []
        extras = native_agent_extras(
            state=state, settings=settings, session=session, events=events,
            approval_policy=approval_policy, cache=tool_result_cache,
            summariser=summariser, parent_role="builder", parent_depth=0,
            parent_allowed_tools=jim_allowed,
        )

        jim_agent = await config_loader(
            ..., extra_tools=extras,
            tool_result_cache=tool_result_cache, summariser=summariser,
        )
        ...
    return summon_jim
```

Update the coordinator to pass `tool_result_cache=self._tool_result_cache, summariser=self._summariser` when calling `make_summon_jim_tool`.

### Step 12 — New events (`src/jac/runtime/events.py`)

```python
@dataclass(frozen=True, slots=True)
class MinionSpawned(RuntimeEvent):
    parent_role: str
    minion_role: str
    task_summary: str
    tools: list[str]
    tier: str
    depth: int


@dataclass(frozen=True, slots=True)
class MinionReturned(RuntimeEvent):
    parent_role: str
    minion_role: str
    duration_ms: int
    success: bool
```

### Step 13 — System-prompt updates (`src/jac/agents/personas.py`)

Append a `MINION GUIDANCE` paragraph to Scott / Pam / Jim. Keep each terse — one paragraph + one example.

```python
SCOTT_SYSTEM_PROMPT += """

MINION GUIDANCE:
- For an unfamiliar API or doc lookup, use spawn_minion('research X', tools=['filesystem:read', 'shell:read']) so your own context stays clean.
- During /init on a multi-module repo, fan out: emit one spawn_minion call per top-level module to summarise concurrently, then stitch their digests into AGENTS.md.
- Prefer read_file_smart for files of unknown size — it tells you when a file is large enough to warrant a minion or a partial read.
"""

JIM_SYSTEM_PROMPT += """

MINION GUIDANCE:
- Before editing a sprawling file (read_file_smart returns metadata_only or large=True), spawn_minion('summarise <path> focusing on <area>', tools=['filesystem:read']) and use the digest to plan your edit.
- Do not spawn minions for trivial reads — small files go through read_file directly.
"""

PAM_SYSTEM_PROMPT += """

MINION GUIDANCE:
- For unfamiliar APIs, frameworks, or external services referenced in the requirement, spawn_minion('research X and report key constraints', tools=['filesystem:read']) so your planning context stays focused on the plan itself.
- Multiple research minions in one turn is fine — Pydantic AI runs them concurrently.
"""
```

### Step 14 — Contract updates

**`docs/contracts/TOOLS_CONTRACT.md`** — replace the "Sub-Agent Tooling Direction (C6c+)" section with concrete spec; document `SummarizedToolResult`, `ToolApprovalMeta.timeout_seconds`, the timeout policy table, `read_file_smart` thresholds, and `spawn_minion(task, tools, tier, timeout_sec) -> str` with depth-≤-1 invariant. Bump `Last revised` to today.

Add a small **Result Interception** subsection after the Return Type section explaining the wrapper chain `raw_tool → approval → result_filter → agent` and the 4k threshold.

**`docs/contracts/STATE_SCHEMA.md`** — extend the `attempts.call_type` discriminator note:

```
call_type ∈ { 'agent', 'direct_llm', 'minion' }
- 'minion'    = single-shot child agent spawned by spawn_minion;
                parent_attempt_id points to the spawning agent's attempt
                (C6c).
```

Bump `Last revised` to today. No SQL change — `call_type` is free-text.

### Step 15 — Layer docs

- `docs/dev/agents-layer.md` — describe `agents/spawn.py`, `agents/result_filter.py`, the `native_agent_extras` helper, the cache lifetime, and depth ≤ 1 enforcement.
- `docs/dev/runtime-layer.md` — describe `RunCoordinator`'s `_tool_result_cache` and `_summariser` ownership, the `MinionSpawned`/`MinionReturned` emit-only events, and that resume starts with an empty cache.

### Step 16 — Version bump

`pyproject.toml` and `src/jac/__init__.py` from `0.3.7 → 0.3.8`. New tools change agent behaviour and surface area; per CLAUDE.md's versioning rule, bump.

### Step 17 — Tests (`tests/test_c6c_spawn_minion_and_tool_hardening.py`)

LLM-free unit tests, modeled after `tests/test_c6_scott_jim.py` and `tests/test_c6b_planner_and_slash_modes.py`.

1. `test_estimate_tokens_word_count_proxy` — known strings produce expected counts (`"hello world"` → 2.6 → 2; an empty string → 0).
2. `test_tool_result_cache_store_and_fetch_roundtrip` — store text, fetch by handle returns it; a missing handle returns None; eviction past `max_entries`.
3. `test_read_file_smart_small_file_returns_full_content` — write a 50-byte file to tmp; assert `large=False`, `metadata_only=False`, content present.
4. `test_read_file_smart_large_file_returns_metadata_only` — write a 200KB file; assert `metadata_only=True`, `content == ""`, `lines_total > 0`, warning string mentions `read_file`/`grep`/`spawn_minion`.
5. `test_read_file_smart_medium_file_flags_large` — write a ~50KB file in the 4k–20k-token band; assert `large=True`, `metadata_only=False`, content present.
6. `test_result_filter_passthrough_below_threshold` — wrap a fake tool that returns a tiny `FileReadResult`; assert wrapper returns the original instance unchanged and the cache stays empty.
7. `test_result_filter_summarises_above_threshold` — wrap a fake tool returning a 50k-character `ShellToolResult`; stub the summariser with a deterministic callable; assert wrapper returns a `SummarizedToolResult`, `original_tokens` is sane, and `cache.fetch(handle)` returns the original JSON.
8. `test_fetch_full_result_returns_cached_verbatim` — pre-populate the cache, build the tool, call it; assert content is verbatim.
9. `test_fetch_full_result_missing_handle_returns_not_found` — call with an unknown handle; assert `ToolStatus.NOT_FOUND`.
10. `test_spawn_minion_refuses_when_parent_is_minion` — build the tool with `parent_depth=1`; call it; assert refusal string.
11. `test_spawn_minion_refuses_tools_outside_caller_whitelist` — build with `parent_allowed_tools=['filesystem:read']`; call with `tools=['shell']`; assert refusal string.
12. `test_spawn_minion_default_tools_strip_destructive_groups` — build with `parent_allowed_tools=['filesystem','shell']`; stub `config_loader` to capture the role string and config row; call with `tools=None`; assert the minion's `allowed_tools` row is `['filesystem:read','shell:read']`.
13. `test_spawn_minion_records_attempt_with_call_type_minion` — stub `config_loader` to return a fake agent whose `run` returns a fixed string; assert an `attempts` row with `call_type='minion'` and `parent_attempt_id` set, status `passed`.
14. `test_spawn_minion_emits_spawned_and_returned_events` — subscribe to events; assert both events fire with expected payloads.
15. `test_spawn_minion_timeout_returns_string` — stub the minion's `.run` to sleep past `timeout_sec`; assert the tool returns "minion timed out after Ns" and emits `MinionReturned(success=False)`.
16. `test_approval_wrapper_per_tool_timeout` — wrap a slow fake tool with `ToolApprovalMeta(timeout_seconds=0.01)`; assert it returns `ToolStatus.TIMEOUT`.
17. `test_native_agent_extras_attaches_two_tools` — call helper; assert the returned list has exactly `[spawn_minion, fetch_full_result]` with proper `.approval` metadata.
18. `test_minion_config_uses_unique_role_each_call` — call `spawn_minion` twice with distinct stubs; assert two `agent_configs` rows with distinct `role` values both prefixed `minion:`.

Smoke (manual, against a real model — read on a debug session):

```bash
uv run jac "Look up what python's contextvars module does and summarise in 5 bullets."  # Scott should spawn a minion
uv run jac "/init"                                                                       # Scott should spawn module minions on a multi-module repo
uv run jac "/plan integrate stripe billing"                                              # Pam should spawn a research minion for stripe
```

Verify in the SQLite DB:
- `agent_configs` has `minion:<...>` rows with `is_minion=1`, `parent_role` set, `depth=1`.
- `attempts` has rows with `call_type='minion'`, `parent_attempt_id` linked to the spawning agent.

---

## Acceptance checks

- [ ] `just test` green (existing + new)
- [ ] `just typecheck` clean
- [ ] `just lint` clean
- [ ] `uv run jac "<prompt that needs a research lookup>"` results in a minion `agent_configs` row + a `call_type='minion'` `attempts` row with `parent_attempt_id` linked to Scott's attempt
- [ ] `uv run jac "/init"` on a multi-module repo emits ≥ 1 `MinionSpawned` event before AGENTS.md is written
- [ ] A 50k-token shell command (e.g. `find / -name '*.py'` clipped) returns a `SummarizedToolResult` to the agent, and `fetch_full_result(handle)` retrieves the verbatim from the cache
- [ ] `read_file_smart` on a 30k-line file returns `metadata_only=True` and the warning text references `read_file`, `grep`, and `spawn_minion`
- [ ] A minion calling `spawn_minion` itself returns the depth-refusal string and does **not** create a second `agent_configs` row
- [ ] `docs/contracts/TOOLS_CONTRACT.md` and `docs/contracts/STATE_SCHEMA.md` updated (revision date bumped)
- [ ] `docs/ROADMAP.md` C6c entry flipped to Done
- [ ] Version bumped in `pyproject.toml` and `src/jac/__init__.py`

---

## Risks and open implementer notes

- **`pydantic_ai.direct` exact API.** The brainstorm capability map names `pydantic_ai.direct.model_request_sync`. Confirm the async signature against the installed version during implementation — the names `model_request` / `model_request_sync` have shifted across recent releases. Cheapest verification: `uv run python -c "from pydantic_ai.direct import model_request; help(model_request)"`.
- **`Agent(tool_timeout=...)` parameter.** Same caveat — verify the kwarg name. If pydantic_ai exposes per-tool timeout only via `Tool(fn, timeout=N)`, fall back to wrapping each tool through `Tool(...)` in `_resolve_local_tools`. Functionally equivalent.
- **`RunContext[None]` import surface.** This is the codebase's first use of `RunContext`. Make sure `pydantic_ai.RunContext` re-exports from the version we pin; otherwise import from `pydantic_ai.tools`. Either is fine; pick whichever the type-checker accepts.
- **Concurrent `spawn_minion` from one model turn.** Pydantic AI auto-schedules multiple tool calls in one model response with `asyncio.create_task`. The closure captures shared `state`, `events`, `cache` — all are async-safe (state is an aiosqlite connection used serially through `await`; events use a future-based bus). No explicit synchronisation needed.
- **Approval prompts for `spawn_minion`.** Risk level `MEDIUM` means yolo-only auto-approval; INTERACTIVE will prompt. That is correct (spawning costs money), but it changes the UX from "Scott silently delegates" to "Scott asks before each spawn". If we want first-spawn-of-a-task to auto-approve under `auto-edit`, surface the policy change in `ApprovalPolicy` rather than hard-coding. Flag for review during implementation; out of scope to redesign the approval policy here.
- **Cache lifetime on `reset_agent`.** `RunCoordinator.reset_agent()` drops the cached agent after a model/tier change. The cache survives — handles remain valid across an agent rebuild. Confirm by grepping for `reset_agent` callers; nothing in C5–C6b touches the cache today, so the implicit "cache lives as long as the run" contract holds.
- **`fetch_full_result` and `result_filter`.** Apply the `category="cache_passthrough"` skip rigorously; otherwise the agent's verbatim retrieval gets re-summarised in a loop. Add a regression test (`test_fetch_full_result_is_not_re_summarised`).
- **Minion summon in chained delegation.** Today Scott → `summon_jim` → Jim, and now Jim can `spawn_minion`. The depth chain is "Scott(0) → Jim(0) → minion(1)". `parent_depth` for spawn-from-Jim is 0 (Jim is a native specialist), so the minion lands at depth 1. Good. If C9 adds Pam → Jim chains, the same logic applies — depth is "depth of native specialist + 1", not "transitive call depth".
- **`call_type='minion'` and existing `_summon_jim` attempts.** `summon_jim` currently writes `call_type='agent'` for Jim's attempt row. Don't change that — Jim is a native specialist, not a minion. Only `spawn_minion`'s child writes `'minion'`.
- **Resume coverage.** Resume rebuilds the coordinator with a fresh `ToolResultCache` and `Summariser`. Old handles in the conversation history dangle. The fetch tool's NOT_FOUND error is the intended user-visible answer; document this in the cache module docstring so future readers don't try to persist it.
- **Approval wrapper interaction with timeout.** `asyncio.wait_for` cancels the inner task on timeout. Approval prompts that are mid-flight (waiting on an interactive future) would also be cancelled, which surfaces as a confusing CLI state. For C6c, `read_file*` (the only tools with non-default timeouts via the wrapper) are READ_ONLY and never prompt — so the cancellation path is dead in practice. Note this in the code so a future tool with both `risk_level=LOW` *and* `timeout_seconds` doesn't ship without thought.
- **Sequential spawn under one parent.** Two `spawn_minion` calls in the same turn produce two `agent_configs` rows with distinct roles. They share `parent_attempt_id` (the parent's active attempt) which is correct for the C7 cost-rollup tree.
