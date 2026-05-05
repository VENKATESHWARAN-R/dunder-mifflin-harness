# Multi-Agent Cast & `spawn_minion` — Final v0 Picture

> **Status:** Draft brainstorm · **Date:** 2026-05-05 · **Supersedes:** [`2026-05-04-manager-specialist-minion-pattern.md`](2026-05-04-manager-specialist-minion-pattern.md) · **Roadmap:** rewrites **C6b**, adds **C6c**, refines **C9**, **C12**, supersedes **C14**

## What this note locks in

JAC's default agent shape is **manager → specialists**, with a universal `spawn_minion` primitive available to every agent. The 2026-05-04 brainstorm proposed an analyst persona (Pam) and a recruiter persona (Holly) — both are dropped. Scott handles intake himself; minion spawning is a deterministic capability rather than a recruiter agent.

Personas are flavor honoring the `dunder-mifflin-play` predecessor — not load-bearing. Workflow nodes don't get personas.

## Final cast (v0)

| Persona | Role | Tier | Purpose |
|---|---|---|---|
| **Michael Scott** | `manager` | Worker | Default agent. Persistent across turns. User-facing. Routes work; handles env probing and conversation directly. |
| **Pam Beesly** | `planner` | Architect | End-to-end planning. Reads requirements, picks dev strategy (feature-by-feature / TDD / spec-driven / agile), produces task list with acceptance criteria, drives the build loop. |
| **Jim Halpert** | `builder` | Worker | Already shipped. Implements one task at a time per Pam's plan. |
| **Dwight Schrute** | `evaluator` | Worker | Grades pass/fail against acceptance criteria; structured fault classification (code vs spec). |

**Renaming note:** the 2026-05-04 note had Pam as `analyst` and "Date Mike" as `planner`. We're collapsing analyst into Scott and renaming Date Mike → Pam (planner). One Pam, one persona, one role.

**No Holly, no minion templates.** Universal `spawn_minion` tool is on every agent (see below). Templates were ruled out: they go unused, get misused for wrong purposes, and most needs are covered by the main agent itself.

## Default flow

```
User → Scott (manager, persistent across turns)
  │
  ├─ trivial reply / chat / explanation     (one Worker call, no delegation)
  │
  ├─ env probe / quick analysis             (Scott runs shell tools himself)
  │
  ├─ /init or "summarize this codebase"     (Scott in init mode; spawns minions per module)
  │
  ├─ /plan or "make a plan for X"           (Pam directly; returns formatted plan)
  │
  └─ /build or "build / implement / fix Y"  (full Pam → Jim → Dwight loop)
        │
        ▼
        Pam (Architect)
          ├─ picks dev strategy from requirements (feature-by-feature / TDD / spec-driven / agile)
          ├─ creates tasks (rows in `tasks` with acceptance criteria)
          └─ drives the per-task loop:
                Jim (Worker) → Dwight (Worker) → pass / fail
```

### Cross-specialist retry escalation

Dwight emits a structured `EvalVerdict` (output_type) with deterministic routing on `fault_type`:

```
Dwight verdict
  ├─ pass                  → next task / done
  ├─ code-fault            → back to Jim with diagnostic. After N=2 Jim attempts, escalate to Pam.
  ├─ spec-fault            → back to Pam to revise the plan section, then Jim re-runs.
  └─ ambiguous             → Dwight asks Jim for clarification; if still ambiguous, Pam arbitrates.
```

**Retry caps:**
- Jim: **N=2** code-fault attempts per task before bouncing to Pam.
- Pam: **M=2** plan revisions per task before failing the task.
- On final failure: Scott surfaces to the user with the failure and asks whether to **escalate the failing role to a bigger model** (next tier up). Pam is already on Architect; for her, escalation means a fresh planning attempt with explicit failure context. Jim/Dwight escalate Worker → Architect.

The fault classifier is **deterministic routing on Dwight's structured output**, not an extra LLM hop.

## `spawn_minion` — universal primitive

Available to **all** agents (Scott, Pam, Jim, Dwight). Each agent's system prompt teaches it when minions are appropriate. Validated against Pydantic AI's agent-as-tool delegation pattern.

### Contract

```python
@agent.tool
async def spawn_minion(
    ctx: RunContext[Deps],
    task: str,                           # natural-language brief, full context inline
    tools: list[str] | None = None,      # whitelist; bounded by caller's tools
    tier: Literal['scout', 'worker'] = 'scout',
    timeout_sec: int = 240,              # hard cap 300 (5 min)
) -> str:
    """Run a single-shot child agent. Returns its final output."""
    minion = await config_loader(
        state=ctx.deps.state, settings=ctx.deps.settings,
        run_id=ctx.deps.run_id, role='minion',
        tier_override=tier, allowed_tools_override=tools,
        parent_role=ctx.deps.calling_role, depth=ctx.deps.depth + 1,
    )
    result = await minion.run(
        task,
        usage=ctx.usage,                            # cost rolls up to parent (Pydantic AI native)
        usage_limits=UsageLimits(...),              # bounded by parent's remaining budget
    )
    return result.output
```

### Invariants

- **Depth ≤ 1.** Minions cannot spawn minions.
- **Tool whitelist bounded by caller.** No privilege escalation.
- **Budget inherited.** Minion `usage_limits` derived from parent's remaining budget.
- **Factory-mediated.** `agents/base.py` is still the only site that calls `Agent(...)`.
- **Tier:** Scout default (matches the user's "primarily scout" guidance); Worker available when classification or summarization needs more reasoning.
- **Timeout:** 3–5 min cap. Pydantic AI's `tool_timeout` handles this; on timeout, "Timed out after N seconds" goes back to the parent agent as a retry signal.

### Parallel spawning

Two patterns ship in v0; the third defers.

| Pattern | Mechanism | When |
|---|---|---|
| **Concurrent spawn within one tool** | `asyncio.gather(*[_spawn(b) for b in briefs])` | `/init` parallel module exploration; planner's parallel research minions. |
| **Concurrent spawn from one model turn** | Pydantic AI auto-schedules multiple tool calls in one model response via `asyncio.create_task`. Free. | Any agent emits 2+ `spawn_minion` calls in one turn. |
| **Fire-and-forget while parent continues** | Pydantic AI **deferred tools**: tool raises `CallDeferred`; agent run terminates with `DeferredToolRequests`; runtime resumes with `DeferredToolResults`. | Defer to **C26+** (out-of-process / sandboxed minions). Overkill for in-process v0. |

### Use-case examples

- **Scott** during `/init`: spawns parallel minions, one per top-level module, to summarise concurrently.
- **Pam** during planning: spawns a research minion for an unfamiliar API, keeping her own context clean.
- **Jim** before editing a sprawling file: spawns a "summarise this 30k-line file" minion.
- **Dwight** when a test suite produces noisy output: spawns a minion to digest the run before grading.

## Tool result interception layer

Above **4k tokens** of output, a tool result is piped through a direct Scout LLM call before reaching the agent.

### Architecture

```
raw_tool → approval_wrapper → result_filter_wrapper → agent
```

`result_filter_wrapper` returns:

```python
class ToolResult(BaseModel):
    summary: str
    summarized: bool
    original_tokens: int | None
    full_result_handle: str | None
    note: str | None     # e.g. "Summarized by AI — 12,400 tokens compressed to 800."
```

A separate tool `fetch_full_result(handle)` returns the original verbatim. **Per-run only**, in-memory cache, discarded at run end.

### Smart file reads

`read_file_smart` is a **separate tool** (not a wrapper around `read_file`). Decision logic before reading:

| File size | Behavior |
|---|---|
| ≤ 4k tokens (or word-count proxy) | Returns full content. |
| 4k–20k tokens | Returns full content + `large=True` flag so the agent's prompt-engineered judgment kicks in (summarise / partial-read / spawn minion). |
| &gt; 20k tokens | Returns metadata only: size, language, top-level structure (grep/tree). Hint: "consider `read_lines(path, start, end)`, `spawn_minion('summarize this file', tools=['read_file'])`, or `grep` first." |

**Token estimation:** word-count proxy is fine for v0 (tokens ≈ words × 1.3 for code). Exact tokenization can be added later if it matters for the threshold accuracy. Flag this as a follow-up.

The agent's system prompt teaches it the decision rule. **No magic auto-summarization on file reads** — gives the agent agency and avoids template-style brittleness.

## Compaction

Auto-compact threshold: **175k tokens** (220k context window − 20k response reserve − margin).

- `/compact` and auto-compact both fire `pydantic_ai.direct.model_request_sync` with a Scout model.
- Output replaces session message history. Preserves: user prompts verbatim, decisions, open todos, current workspace state, `/init`/`/plan` artifacts. Drops: tool noise, completed sub-task chatter.
- Recorded in `attempts` with `call_type='direct_llm'`.
- Mechanic relies on `agents/base.py` using `instructions=` (not `system_prompt=`) — instructions re-apply each run regardless of message history, so compaction safely strips old prompt copies.

## Slash commands — bypass channels

| Command | Routes to | History retention |
|---|---|---|
| `/init` | Scott + init-mode addendum | Custom prompt + final **AGENTS.md** content. Intermediate tool calls dropped. |
| `/plan <task>` | **Pam directly** | Custom prompt + final formatted plan (task list + chosen strategy). |
| `/build <task>` | Pam → Jim → Dwight loop | Per-task outcomes; final summary. |
| `/eval <target>` | Dwight directly | Verdict (`EvalVerdict`). |
| `/compact` | Direct Scout call | Replaces history in place. |
| `/explore <path>` | Scott + spawn_minion | Final exploration report. |

User-facing default chat: always Scott. Slash commands are the power-user bypass; results still thread back into the conversation so Scott has context next turn.

## Dynamic system prompts — mode addendums

Per Pydantic AI's `@agent.system_prompt` decorator. Slash commands set a session mode; the dynamic prompt fragment loads the mode-specific addendum.

```python
@scott.system_prompt
def mode_addendum(ctx: RunContext[Deps]) -> str:
    return MODE_PROMPTS.get(ctx.deps.session.mode, '')
```

The slash command's full custom prompt **stays in history** so Scott (or Pam) has it next turn. Final artifacts (AGENTS.md, plan text) **stay in history** as assistant messages. Intermediate tool calls during a slash-command run are **filtered** before being persisted (drop `ToolCallPart` / `ToolReturnPart` blocks except the final artifact).

### `/init` flow (no template, fully dynamic)

1. `/init` invokes Scott with init-mode addendum (sets `ctx.deps.session.mode = 'init'`).
2. Scott surveys `cwd`: directory tree, file counts, sizes, language hints (shell tools).
3. Scott decides:
   - Small repo → reads inline, writes **AGENTS.md** at project root.
   - Large repo → spawns parallel minions per top-level module via `spawn_minion`; aggregates module digests into AGENTS.md.
4. Scott composes final AGENTS.md.

### Project-level vs user-level instruction files

| Path | Purpose | Authored by |
|---|---|---|
| `./AGENTS.md` | Project-level harness instructions | Generated by `/init`; manually edited |
| `~/.jac/JAC.md` | User-level custom instructions, loaded by all runs | User, manually |

## Workflow strategies (TDD, agile, spec-driven, waterfall)

Same four personas across all modes. Personas play different node roles per workflow. **No new personas per strategy.** Pam picks the strategy at planning time based on the requirements (e.g., test-heavy domain → TDD; tight spec → spec-driven; exploratory → feature-by-feature).

Strategy selection is **Pam's structured output** — part of the plan she emits. Roadmap C22 still owns the alternative-strategy implementations; node interfaces are designed for reuse from day one.

## Pydantic AI capability map (validated)

| Need | Pydantic AI primitive | Notes |
|---|---|---|
| Agent-as-tool delegation | `await sub_agent.run(prompt, usage=ctx.usage, deps=ctx.deps)` | Cost rollup is automatic. |
| Parallel sub-agent runs | `asyncio.gather` inside a tool, OR multiple tool calls in one model response (Pydantic AI auto-schedules). | Confirmed in docs. |
| Tool timeouts | `Agent('model', tool_timeout=180)` agent-wide; `@tool(timeout=5)` per-tool. On timeout the model gets a retry prompt. | Counts toward retry limit. |
| Sequential tool execution | `agent.parallel_tool_call_execution_mode('sequential')` context manager, or `sequential=True` on a tool. | Used for ordered build steps. |
| Dynamic system prompts | `@agent.system_prompt` decorator with `RunContext`. | Powers `/init`, `/plan` mode addendums. |
| Structured outputs | `output_type=` parameter on Agent or run. | `EvalVerdict`, `Plan`, `MinionBrief`. |
| Usage / cost rollup | `usage=ctx.usage` parameter on sub-runs; `result.usage()`. | Maps directly to `attempts` table. |
| Usage caps | `UsageLimits(...)` parameter on `agent.run`. | Budget inheritance for minions. |
| Direct LLM call (no agent) | `pydantic_ai.direct.model_request_sync(...)`. | Compaction + tool result summarization. |
| Long-running deferred work | `raise CallDeferred(...)`, `DeferredToolRequests`, `DeferredToolResults`. | Out-of-process minions at C26+. |
| Streaming graph execution | `agent.iter()` returning `AgentRun`. | Used by hooks (C13) and observers. |

## Roadmap impact

- **C6 (shipped):** unchanged. Scott + Jim + `summon_jim`.
- **C6b (rewritten):** Pam (planner, Architect) + dynamic mode addendums (`@agent.system_prompt`) + `/plan` slash + `/init` (Scott in init mode) + AGENTS.md generation. Drops the prior analyst-Pam framing.
- **C6c (new):** Universal `spawn_minion` (all agents) + Pydantic AI `tool_timeout` policy + tool result interception (4k threshold + `fetch_full_result`) + `read_file_smart`. Tool-layer hardening that unblocks `/init` and Pam's research-minion use case.
- **C9 (rewritten):** Dwight (evaluator, Worker) + `EvalVerdict` structured output + retry loop + code-fault/spec-fault classification + retry caps (Jim N=2, Pam M=2) + escalation to user/Scott on final failure (offer model upgrade) + `/build` slash + `/eval` slash. Single-task autonomy checkpoint preserved.
- **C12 (refined):** compaction at 175k via `pydantic_ai.direct` Scout call; auto + `/compact`; relies on `instructions=` semantics in `agents/base.py`.
- **C14 (superseded):** "Holly Flax / Temp Agency" is **superseded by C6c**. Holly persona dropped; the recruiter slot in `agent_configs.role` is removed from the v0 enum.
- **C26+:** deferred-tools pattern enters when minions move out-of-process (sandboxed/cloud).

## Schema implications (follow-up revision needed)

Flagged here, not yet written into `STATE_SCHEMA.md`. Belongs in a separate revision pass.

| Table | Change | Why |
|---|---|---|
| `agent_configs.role` | enum: `manager | planner | builder | evaluator | minion`. **Drop `analyst`, `recruiter`** from v0. | Four-persona cast + minion role. |
| `agent_configs` | Keep `is_minion`, `parent_role`, `depth` (from prior brainstorm). | Minion lineage. |
| `attempts` | Keep `parent_attempt_id`. Add `call_type` discriminator (`agent | direct_llm | minion`). | Cost rollup tree + Scout summarization audit. |
| New: `tool_results_cache` | In-memory only for v0 (per-run dict). Add table later if persistence across crashes matters. | `fetch_full_result(handle)` lookup. |
| `tasks` | Add `dev_strategy TEXT` (`feature_by_feature | tdd | spec_driven | agile`) populated by Pam's structured output. | Strategy auditing + research data. |
| `tasks` | Confirm `attempt_count` exists for retry-cap enforcement (already in schema for C19). | Retry caps. |

## Tools contract implications (follow-up revision needed)

Flagged for `TOOLS_CONTRACT.md` revision pass.

- New tool result shape: `summary`, `summarized`, `original_tokens`, `full_result_handle`, `note`.
- `tool_timeout` policy: agent-wide default 180s; `spawn_minion` 240s; `read_file` 30s; `shell_exec` 180s with auto-bumped 300s on retry.
- `read_file_smart` tool spec (size thresholds, decision rule).
- `fetch_full_result(handle)` tool spec.
- `spawn_minion(task, tools, tier, timeout_sec)` tool spec.

## Open follow-ups (not blockers for the next implementation pass)

- **Token estimation accuracy** for the 4k / 20k thresholds. Word-count proxy v0; exact tokenizer later if measurements show drift.
- **Mode-addendum prompt content.** Init mode prompt, plan mode prompt, build mode prompt — write before C6b implementation.
- **`EvalVerdict` schema** — fault categorization rubric. Lock during C9 implementation.
- **Pam's strategy-selection rubric.** When does Pam choose TDD vs feature-by-feature? Document in C22's design notes; for C6b she defaults to feature-by-feature.
- **Tool result cache size cap.** Per-run dict, but how big can it grow? Likely bounded by tool-call budget; flag for monitoring.
- **PHILOSOPHY.md addition.** A "Manager–Specialist + Universal Minion" section codifying the pattern. Write after C6b ships, not before.
