# C7 — Usage Tracking & `/usage`

> **Status:** Ready for implementation
> **Last revised:** 2026-05-07
> **Depends on:** C6 (shipped 2026-05-04), C6b (shipped 2026-05-06), C6c (shipped 2026-05-06)
> **Roadmap entry:** [`docs/ROADMAP.md` §C7](../ROADMAP.md#c7--cost-tracking--cost)
> **Contract touchpoints:** [`docs/contracts/EVENT_CONTRACT.md`](../contracts/EVENT_CONTRACT.md), [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md), [`docs/contracts/CLI_DESIGN.md`](../contracts/CLI_DESIGN.md)

---

## Scope change vs. roadmap text

The original C7 was titled "Cost Tracking & `/cost`" and assumed per-call dollar cost rollup. **Dollar cost is dropped from C7** — provider price tables drift independently of our ship cadence and are unreliable to maintain. C7 ships **token usage** tracking only; the slash command becomes `/usage` (alias `/cost` for muscle memory). The `attempts.cost` column stays in the schema (defaults to 0.0) and is left for a possible future component.

Update `docs/ROADMAP.md` §C7 to reflect this scope change as part of this work (see Step 11).

## Goals

1. **Persist per-attempt token usage** into the existing `attempts` table (`tokens_in`, `tokens_out`, `requests`, `tool_calls`, `duration_ms`). Schema columns already exist; today they default to 0 and are never written.
2. **Track direct LLM calls** (Scout summariser in `tools/summarize.py`) as `call_type='direct_llm'` rows so they appear in the call tree.
3. **Cumulative session usage banner** in the prompt_toolkit bottom toolbar — input/output tokens, requests, current context size, % of model max.
4. **`/usage`** slash command — render the run's attempt tree with token totals, per-role and per-tier subtotals.
5. **Extend `/context`** — show last call's input_tokens, % of model max, headroom, and a per-turn growth table.
6. **`/clear` resets** the cumulative session counter (model A from the design discussion).
7. **Replace `CostUpdated` event** outright with `SessionUsageUpdated` carrying structured fields. Contract is unreleased; clean break.

## Non-goals (later components)

- **Dollar cost.** Provider pricing tables are out of scope. Revisit only if a future component justifies the maintenance burden.
- **Per-message context breakdown** (system prompt vs. tool defs vs. history vs. tool results). Requires a tokenizer pass and is **C12** (Context Management Module) territory.
- **Compaction triggers / auto-`/compact`.** Banner shows headroom; user decides. C12 owns automatic action.
- **Cross-session usage history** (e.g. "this week you spent 4M tokens"). Per-run / per-session only.
- **Pricing or context-window data per provider beyond what we ship today.** Cover Anthropic Claude 4.x family at minimum; OpenAI / Gemini specs can be appended to the TOML when C8 lights up tier routing.

---

## Design decisions (locked)

### D1. Per-row token storage = **delta, not cumulative**

Pydantic AI's `usage=ctx.usage` parameter means a child agent's `result.usage()` reflects the **shared** parent usage tracker — i.e. it includes the parent's tokens accumulated before the child ran. Storing `result.usage()` verbatim per attempt would mean parents and children both contain children's tokens, breaking any tree aggregation.

**Rule:** each `attempts` row stores **its own delta**.

- **Sub-agent calls** (`summon_jim`, `spawn_minion`, slash runs spawned from inside another run): snapshot `ctx.usage` before the call, compute delta after, write delta to the child attempt.
- **Root attempts** (Scott's manager run, top-level slash runs): `result.usage()` is the cumulative total. Write `total − Σ direct-children deltas` to the root's row, executed after the run finishes (children's rows already exist and are queryable by `parent_attempt_id`).
- The run's grand total = `SUM(tokens_in)` over all rows for the run. No double counting.

### D2. Context-window source = shipped TOML

Per the project philosophy of avoiding hardcoded dicts unless purely needed: model→max_context lives in **`src/jac/data/model_specs.toml`**, loaded once at runtime and cached.

- Data is external service spec. It changes when providers ship new models, not when our code changes. TOML keeps it diff-friendly and out of Python.
- A future env var (`JAC_MODEL_SPECS_PATH`) can override the path if a deployment needs custom values; not implemented in C7.
- Pydantic AI does not currently expose `max_context` on `Model` instances — verified during research. If that changes upstream, drop the TOML and read from the model object.
- Unknown model → fall back to a conservative `default_max_context` value declared in the same TOML.

### D3. `SessionUsageUpdated` replaces `CostUpdated` outright

Old event: `CostUpdated(summary: str)`. Single string, last-call only.

New event:

```python
@dataclass(frozen=True, slots=True)
class SessionUsageUpdated(RuntimeEvent):
    tokens_in: int            # cumulative session, in
    tokens_out: int           # cumulative session, out
    requests: int             # cumulative requests
    tool_calls: int           # cumulative tool calls
    last_context_tokens: int  # input_tokens of the most recent LLM call = current context size
    context_max: int          # model max for the *current* model
    context_pct: float        # last_context_tokens / context_max
    model: str                # model id of the most recent call
```

Emitted after every `LlmCallCompleted`. Replaces the existing emission sites in `coordinator.submit` / `coordinator.submit_slash_run` / `agents/tools.py`. `SessionState.latest_cost_summary` is removed.

### D4. `/clear` resets cumulative; `/compact` does not

`/clear` is a fresh start — banner counters reset to 0, last_context_tokens reset to 0. `/compact` (C12) keeps the run going and should not reset cumulative. C7 only owns `/clear` behaviour; `/compact` doesn't exist yet but we leave a code comment in the reset helper noting it should *not* call this path.

---

## Key files and integration points

| File | What it is | C7 touches it? |
|---|---|---|
| `src/jac/data/model_specs.toml` | **New file.** Model max_context table | Yes — create |
| `src/jac/runtime/model_specs.py` | **New file.** TOML loader + lookup helpers | Yes — create |
| `src/jac/state/attempts.py` | `attempts` repo | Yes — add `update_usage`, `tree_for_run`, `totals_for_run` |
| `src/jac/state/__init__.py` | Repo exports | Yes — surface new methods if exported |
| `src/jac/runtime/events.py` | Typed events | Yes — replace `CostUpdated` with `SessionUsageUpdated` |
| `src/jac/runtime/session.py` | `SessionState` + `SessionConfig` | Yes — add cumulative counters; remove `latest_cost_summary` |
| `src/jac/runtime/coordinator.py` | `RunCoordinator` | Yes — wire usage capture + delta computation in `submit` and `submit_slash_run`; emit `SessionUsageUpdated`; reset path for `/clear` |
| `src/jac/agents/tools.py` | `make_summon_jim_tool` | Yes — snapshot `ctx.usage` before/after; persist delta to Jim's attempt |
| `src/jac/agents/spawn.py` | `make_spawn_minion_tool` | Yes — same snapshot/delta pattern for minions |
| `src/jac/tools/summarize.py` | Scout direct-LLM summariser | Yes — create a `direct_llm` attempt row, capture `response.usage` |
| `src/jac/cli/input.py` | prompt_toolkit `InputSession` | Yes — extend `_toolbar()` to surface session usage |
| `src/jac/cli/app.py` | `ChatApp` | Yes — register `/usage`, alias `/cost`; rewrite `/clear`; extend `/context`; update toolbar source |
| `src/jac/cli/renderer.py` | Rich renderer | Yes — `render_usage_breakdown(...)`, `render_context_growth(...)`; replace `_on_cost_updated` |
| `src/jac/cli/commands.py` | Slash registry | No (registry already supports aliases) |
| `docs/contracts/EVENT_CONTRACT.md` | Locked | Yes — replace `CostUpdated` with `SessionUsageUpdated`; bump revision date |
| `docs/contracts/STATE_SCHEMA.md` | Locked | Yes — note `attempts` token columns are now populated; add `'direct_llm'` to `call_type` discriminator; bump revision date |
| `docs/contracts/CLI_DESIGN.md` | Locked | Yes — document `/usage` (and `/cost` alias), `/clear` reset semantics, `/context` growth view, toolbar shape; bump revision date |
| `docs/dev/runtime-layer.md` | Layer doc | Yes — usage capture + delta rules + event flow |
| `docs/dev/state-layer.md` | Layer doc | Yes — attempts repo additions |
| `docs/dev/cli-layer.md` | Layer doc | Yes — toolbar usage source, `/usage` renderer, `/context` growth |
| `docs/guide/usage.md` | User guide | Yes — `/usage`, `/context` growth, toolbar walkthrough |
| `docs/guide/configuration.md` | User guide | Yes — note `model_specs.toml` location and override path |
| `docs/ROADMAP.md` | Living | Yes — flip C7 to Done; rewrite the entry to reflect dropped $-cost scope |
| `docs/README.md` | Living index | Yes — only if a new doc row is needed; otherwise skip |
| `README.md` | Repo root | Yes — quickstart should mention `/usage` if `/cost` is referenced |
| `pyproject.toml` + `src/jac/__init__.py` | Version | Yes — bump (user-visible behaviour change: new banner, new commands) |
| `tests/test_usage_tracking.py` | **New file** | Yes — create |
| `tests/test_context_growth.py` | **New file** | Yes — create |
| Existing tests referencing `CostUpdated` / `latest_cost_summary` | various | Yes — migrate to `SessionUsageUpdated` |

**No SQL migration.** All needed columns already exist (migration 002 from C6).

---

## Step-by-step implementation

### Step 1 — Ship `model_specs.toml` + loader

Create `src/jac/data/model_specs.toml`:

```toml
# Model specs shipped with JAC. Update when providers ship new models.
# Schema: [models."<provider>:<model_id>"] keyed entries with `max_context` (input tokens).
# Provider/model id matches Pydantic AI's `model_ref` form used in `runtime/models.py`.

[defaults]
max_context = 200000

[models."anthropic:claude-opus-4-7"]
max_context = 200000

[models."anthropic:claude-sonnet-4-6"]
max_context = 200000

[models."anthropic:claude-haiku-4-5"]
max_context = 200000

[models."anthropic:claude-haiku-4-5-20251001"]
max_context = 200000
```

Add the file to the wheel. In `pyproject.toml`, ensure `[tool.hatch.build.targets.wheel]` (or whatever build backend is in use — verify) includes `src/jac/data/*.toml` via `force-include` or `package-data`.

Create `src/jac/runtime/model_specs.py`:

```python
"""Per-model spec lookup (max context window). Data lives in src/jac/data/model_specs.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class ModelSpec:
    max_context: int


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, ModelSpec], ModelSpec]:
    raw = tomllib.loads(files("jac.data").joinpath("model_specs.toml").read_text())
    default = ModelSpec(max_context=int(raw["defaults"]["max_context"]))
    models = {
        key: ModelSpec(max_context=int(entry["max_context"]))
        for key, entry in (raw.get("models") or {}).items()
    }
    return models, default


def spec_for(model_ref: str) -> ModelSpec:
    """Look up the spec for a model_ref. Falls back to defaults if unknown."""
    models, default = _load()
    return models.get(model_ref, default)
```

Tests: load + lookup hit, lookup miss → default, missing TOML raises (let's not silently degrade).

### Step 2 — Extend `attempts` repo

`src/jac/state/attempts.py` — add the following methods (sketch; use the existing repo conventions for connection handling and parameter binding):

```python
async def update_usage(
    self,
    attempt_id: str,
    *,
    tokens_in: int,
    tokens_out: int,
    requests: int,
    tool_calls: int,
    duration_ms: int,
) -> None:
    """Persist token usage onto an attempt row. Idempotent at the row level."""
    # UPDATE attempts SET tokens_in=?, tokens_out=?, duration_ms=? ... WHERE attempt_id=?
    # Note: requests / tool_calls are not first-class columns — append to the schema only if needed.
```

**Decision for `requests` / `tool_calls`:** the schema doesn't have these columns. Two options:
- (a) Add them in a new migration `003_c7_usage.sql`. Cheap and they're useful for `/usage` rendering.
- (b) Store them only in the in-flight `SessionUsageUpdated` event and don't persist. Tree rendering shows tokens only.

**Pick (a)** — add a migration `003_c7_usage.sql` introducing `requests INTEGER NOT NULL DEFAULT 0` and `tool_calls INTEGER NOT NULL DEFAULT 0` on `attempts`. Update `STATE_SCHEMA.md` accordingly.

```python
async def list_for_run(self, run_id: str) -> list[AttemptRow]:
    """Already exists or trivial — returns rows ordered by created_at."""

async def tree_for_run(self, run_id: str) -> list[AttemptNode]:
    """Build a parent→children tree from list_for_run. Pure Python after the SELECT."""

async def totals_for_run(self, run_id: str) -> RunTotals:
    """Aggregate tokens_in, tokens_out, requests, tool_calls; group by role and tier."""
```

`AttemptNode` and `RunTotals` are small dataclasses local to the repo (or `runtime/models.py` if shared). Keep them dumb data carriers.

### Step 3 — Replace `CostUpdated` with `SessionUsageUpdated`

In `src/jac/runtime/events.py`:

- Delete the `CostUpdated` dataclass.
- Add `SessionUsageUpdated` per D3 above.
- Search the codebase for `CostUpdated` and migrate every reference:
  - `runtime/coordinator.py` emission sites (lines 343, 483 per the audit).
  - `cli/renderer.py` `_on_cost_updated` handler (line 144 per the audit).
  - Any test file referencing `CostUpdated`.

In `src/jac/runtime/session.py`:

- Remove `SessionState.latest_cost_summary`.
- Add cumulative counters:

```python
@dataclass(slots=True)
class SessionState:
    ...
    cumulative_tokens_in: int = 0
    cumulative_tokens_out: int = 0
    cumulative_requests: int = 0
    cumulative_tool_calls: int = 0
    last_context_tokens: int = 0
    last_model: str | None = None

    def reset_usage_counters(self) -> None:
        """Called by /clear. Do NOT call from /compact."""
        self.cumulative_tokens_in = 0
        self.cumulative_tokens_out = 0
        self.cumulative_requests = 0
        self.cumulative_tool_calls = 0
        self.last_context_tokens = 0
        # last_model intentionally preserved — model selection survives /clear.
```

### Step 4 — Wire usage capture at the 4 agent sites

The pattern is the same at each site. **Define a helper** in `runtime/coordinator.py` (or a new `runtime/usage.py` if it grows) so the four call sites stay terse:

```python
@dataclass(frozen=True, slots=True)
class UsageDelta:
    tokens_in: int
    tokens_out: int
    requests: int
    tool_calls: int
    duration_ms: int


def snapshot_usage(usage_obj: Any | None) -> tuple[int, int, int, int]:
    """Read input_tokens / output_tokens / requests / tool_calls from a Pydantic AI Usage,
    or zeros if usage_obj is None (top-level call)."""

def usage_delta(before: tuple[int, int, int, int], after_usage: Any, started_at: float) -> UsageDelta:
    """Compute (after - before) and elapsed ms."""
```

**Apply at each site:**

1. **`coordinator.submit` (Scott's manager run, root attempt)** — `runtime/coordinator.py` ~lines 272–356:
   - Snapshot is `(0, 0, 0, 0)` (no parent ctx).
   - After `result = await self.agent.run(...)` → `total = snapshot_usage(result.usage())`.
   - Children of this attempt already have their own rows written by the time we reach here (they ran during tool execution). Query `attempts.list_children(scott_attempt_id)`, sum their `tokens_in/out`, subtract from total. The remainder is Scott's "own" delta. Persist via `update_usage`.
   - Update `session.cumulative_*` by total (not delta — total is what flowed across the wire).
   - Emit `SessionUsageUpdated` with cumulative + last_context from `total.input_tokens` and `spec_for(session.config.model).max_context`.

2. **`coordinator.submit_slash_run`** (lines ~448–502): same pattern, root attempt. Slash runs (e.g. `/plan`) are isolated runs; they don't have a parent on the call stack but they share the session.

3. **`agents/tools.py make_summon_jim_tool`** (lines ~58–134): Jim is a child of Scott (or whichever caller).
   - Before `result = await jim_agent.run(prompt, usage=ctx.usage, ...)` → snapshot `ctx.usage`.
   - After → compute delta from snapshot. Write delta to Jim's attempt via `update_usage`.
   - **Do not** update `session.cumulative_*` here — Scott's outer `submit` will roll up the same tokens via `result.usage()` of the outer run; updating cumulative twice would double-count. Cumulative is updated **only at root attempts**.
   - Emit `LlmCallCompleted` with delta values (already happens; just point it at the delta instead of `result.usage()`).

4. **`agents/spawn.py make_spawn_minion_tool`** (lines ~101–148): same as Jim, child semantics.

**Important invariant:** `SessionUsageUpdated` is emitted **only at root-attempt completion** (i.e. inside `coordinator.submit` and `coordinator.submit_slash_run`). Sub-agent sites emit `LlmCallCompleted` for trace visibility but do not touch session cumulative state.

### Step 5 — Track direct-LLM summariser calls as attempts

`src/jac/tools/summarize.py` currently calls `model_request(...)` and returns the summary. Wrap it:

```python
# Before the model_request call:
from jac.state.attempts import AttemptRow  # or via repo
attempt_id = await attempts_repo.create(
    run_id=run_id,
    role="summariser",          # convention; document in STATE_SCHEMA
    parent_attempt_id=session.active_attempt_id,
    call_type="direct_llm",
    model=selection.model_ref,
    tier="scout",
)
started_at = time.perf_counter()
response = await model_request(...)
elapsed_ms = int((time.perf_counter() - started_at) * 1000)

usage = response.usage  # property, not method, on direct API
await attempts_repo.update_usage(
    attempt_id,
    tokens_in=usage.input_tokens,
    tokens_out=usage.output_tokens,
    requests=1,
    tool_calls=0,
    duration_ms=elapsed_ms,
)
await attempts_repo.update_status(attempt_id, "passed")
```

The summariser is invoked from inside the `result_filter` wrapper. Thread `attempts_repo`, `run_id`, and `session.active_attempt_id` through the existing `Summariser` callable signature (built in `coordinator.__init__`).

Direct LLM calls **do not** drive `SessionUsageUpdated` on their own — they're internal. The next root completion will pick up cumulative naturally because `result.usage()` of the outer agent run already includes the summariser's tokens (it ran during a tool call inside the outer run). Wait — does it? **Verify during implementation:** if `model_request(...)` is a fully separate call outside the agent's run scope, its tokens are NOT in the outer run's `result.usage()`. In that case, also bump `session.cumulative_*` directly when persisting the direct_llm attempt. Add a brief test that proves this either way.

### Step 6 — Toolbar shows session usage

In `src/jac/cli/input.py`, the `_toolbar()` closure currently reads from `session_config_source()`. Extend the source contract so the toolbar can also read usage state. Two ways:

- (a) Pass `session_state_source: Callable[[], SessionState]` separately.
- (b) Have `session_config_source` return a richer wrapper (config + usage).

Pick **(a)** — separation of concerns; `SessionConfig` stays config-only.

`InputSession` accepts a new optional `session_usage_source: Callable[[], dict] | None = None`. The closure renders:

```
 model: opus-4-7 · tier: worker · mode: hitl · approval: interactive
 tokens: 12.4k in / 3.1k out · 4 reqs · ctx: 45k/200k (22%)
```

Two-line toolbar via `\n` (prompt_toolkit `bottom_toolbar` supports formatted text with newlines via `FormattedText` pieces). Format helper for `12.4k` style numbers — keep it as a small private function in `input.py`.

In `cli/app.py`, wire `session_usage_source=lambda: {...}` reading from `self.session` (the `SessionState`).

### Step 7 — `/usage` slash command (alias `/cost`)

Replace the existing `/cost` stub in `cli/app.py` with a real handler:

```python
async def usage_command(_args: str) -> None:
    if self.state is None:
        self.renderer.print_error("State store not configured.")
        return
    tree = await self.state.attempts.tree_for_run(self.session.run_id)
    totals = await self.state.attempts.totals_for_run(self.session.run_id)
    spec = spec_for(self.session.config.model or DEFAULT_MODEL)
    self.renderer.render_usage_breakdown(tree, totals, spec.max_context)

self.commands.register("usage", usage_command, "Show token usage tree for the current run")
self.commands.alias("cost", "usage")
```

`renderer.render_usage_breakdown(tree, totals, model_max)` builds a `rich.tree.Tree` rooted on the run, with one branch per root attempt, nested children. Each node label includes `role · model · tokens_in/out · requests`. After the tree, render a `Table` with by-role totals and a small Panel with the run grand total.

Visual sketch:

```
Run abc123 — 8 attempts · 14.2k in / 4.1k out · 7 reqs · 3 tool calls
├── manager (Scott · sonnet-4-6) — 1.1k in / 320 out
│   ├── builder (Jim · sonnet-4-6) — 4.5k in / 1.1k out
│   │   └── direct_llm (summariser · haiku-4-5) — 800 in / 120 out
│   └── planner (Pam · opus-4-7) — 6.8k in / 2.5k out
│       └── minion (research · haiku-4-5) — 1.0k in / 60 out

By role:
  manager   1.1k in / 320 out
  builder   4.5k in / 1.1k out
  planner   6.8k in / 2.5k out
  minion    1.0k in / 60 out
  direct    800 in / 120 out
```

### Step 8 — Extend `/context`

Current `/context` shows attached files and message count. Add:

- Last context size (`session.last_context_tokens`) and `% of model_max`.
- Headroom: `model_max − last_context_tokens`.
- Per-turn growth table: query `attempts` for the run ordered by `created_at`, render as a small `rich.Table`:

```
Run abc123 · 8 turns
Context: 45,231 / 200,000 tokens (22%) · headroom 154,769

Recent growth:
  #  attempt           in        Δ in    notes
  1  manager        1.2k          —      user prompt
  2  manager        3.4k       +2.2k     Scott reply + summon_jim
  3  builder       12.0k       +8.6k     Jim built file (large tool result)
  4  builder       44.1k      +32.1k     grep on 50-file repo (auto-summarised at 4k cap)
  5  manager       45.2k       +1.1k     Scott summary
```

The `notes` column is best-effort — pull from a heuristic (e.g. recent `ToolCallCompleted` event the attempt was associated with). If too noisy in practice, drop the column and ship just the growth numbers; the value is in `Δ in`.

### Step 9 — `/clear` resets cumulative

Locate the existing `/clear` handler in `cli/app.py`. After whatever it does today (delete messages, reset `session.run_id` if applicable), add:

```python
self.session.reset_usage_counters()
await self.events.emit(SessionUsageUpdated(  # zero-emission so toolbar redraws
    tokens_in=0, tokens_out=0, requests=0, tool_calls=0,
    last_context_tokens=0,
    context_max=spec_for(self.session.config.model or DEFAULT_MODEL).max_context,
    context_pct=0.0,
    model=self.session.config.model or "",
))
```

Add a code comment near the helper noting `/compact` (when it lands at C12) **must not** call `reset_usage_counters`.

### Step 10 — Tests

New test files:

- `tests/test_usage_tracking.py`:
  - `update_usage` writes to all four columns + duration_ms.
  - `tree_for_run` builds correct parent→child structure.
  - `totals_for_run` sums correctly and groups by role/tier.
  - Sub-agent delta computation (mock `ctx.usage` snapshots) yields right values.
  - Root attempt's `tokens_in` = total − Σ children.
  - Direct LLM call creates a row with `call_type='direct_llm'`.
  - `SessionUsageUpdated` is emitted from root sites only (not sub-agent sites).
  - `/clear` zeroes cumulative and emits a SessionUsageUpdated with zeros.

- `tests/test_context_growth.py`:
  - `/context` renders with no attempts (empty growth).
  - Growth table shows correct deltas across 3 attempts.
  - Headroom math correct against TOML-loaded `max_context`.
  - Unknown model falls back to `defaults.max_context`.

Migrate existing tests:
- Anywhere `CostUpdated` is referenced → `SessionUsageUpdated` with the right fields.
- Anywhere `latest_cost_summary` is referenced → cumulative counters.
- `tests/test_cli_commands.py` `/cost` dispatch test → `/usage` (and verify `/cost` alias still dispatches).

Run the smoke checks per CLAUDE.md:
```bash
just qa   # lint + typecheck + tests
uv run jac --help
uv run jac "say hello"
uv run jac chat
```

### Step 11 — Documentation updates

These are part of C7 ship, not a follow-up. Each doc must end up consistent with the implementation.

**Locked contracts (bump `Last revised` date on each):**

- **`docs/contracts/EVENT_CONTRACT.md`** — remove `CostUpdated`, add `SessionUsageUpdated` with the field list from D3. Add a one-line note on emission site (root attempts only).
- **`docs/contracts/STATE_SCHEMA.md`** — note that `attempts.tokens_in`, `tokens_out`, `duration_ms` are populated as of C7. Add the new `requests` and `tool_calls` columns (migration 003). Document the `'direct_llm'` value for `call_type`.
- **`docs/contracts/CLI_DESIGN.md`** — document `/usage` (alias `/cost`), `/clear`'s reset semantics, the new `/context` growth view, and the two-line toolbar shape. Note `/compact` is reserved for C12 and does NOT reset cumulative.

**Reference & dev docs:**

- **`docs/dev/runtime-layer.md`** — usage capture rules (D1), root vs. child semantics, `SessionUsageUpdated` flow, model_specs loader.
- **`docs/dev/state-layer.md`** — `attempts.update_usage`, `tree_for_run`, `totals_for_run`. Note migration 003.
- **`docs/dev/cli-layer.md`** — toolbar usage source, `/usage` rendering, `/context` growth.
- **`docs/guide/usage.md`** — show `/usage` and the new `/context` output. Walkthrough of the toolbar.
- **`docs/guide/configuration.md`** — note the location of `model_specs.toml` and the (future) `JAC_MODEL_SPECS_PATH` override knob.

**Living docs:**

- **`docs/ROADMAP.md`** — flip C7 to **Done** with a 2026-MM-DD ship line at the top of the file (matching the C6/C6b/C6c style). Rewrite the C7 entry body:
  - Title: "C7 — Usage Tracking & `/usage`" (rename).
  - Drop all `$`/cost language.
  - Update the **Ships** list to: per-attempt token persistence, `direct_llm` attempt rows, `SessionUsageUpdated` event, bottom-toolbar usage banner, `/usage` (alias `/cost`), extended `/context`, `/clear` reset.
  - Update **Evaluation** bullets to reflect token assertions (no $ assertions).
- **`docs/README.md`** — only update if a doc row needs adding; otherwise leave alone.
- **`README.md`** — quickstart/command snippets: replace any `/cost` reference with `/usage`. Mention the toolbar shows tokens + ctx headroom.

**Versioning:**

- **`pyproject.toml`** + **`src/jac/__init__.py`** — bump version (user-visible: new toolbar, new commands, contract change). Per CLAUDE.md alpha policy: `0.x.y` → `0.x+1.0` is appropriate since `CostUpdated` removal is a breaking event-contract change.
- Run `uv lock` if any dep moves.

---

## Acceptance checks

A reviewer should be able to verify each:

1. **Per-attempt usage persists.** Run `jac chat` → submit a build prompt that triggers Scott → Jim → minion. Inspect `state.db`:
   ```bash
   sqlite3 .agents/state.db 'SELECT attempt_id, role, call_type, parent_attempt_id, tokens_in, tokens_out, requests, tool_calls, duration_ms FROM attempts WHERE run_id = ?;'
   ```
   Every row has non-zero `tokens_in`, `tokens_out`, `duration_ms`. `requests >= 1` for every row that made a model call.

2. **Tree sums to total.** `SUM(tokens_in)` over all rows in the run equals `result.usage().input_tokens` of the root agent run. (Within ±0 since deltas are exact.)

3. **`/usage` renders the tree** with role labels, parent/child structure, by-role totals, run grand total. `/cost` is accepted and runs the same handler.

4. **Toolbar shows live usage.** Bottom toolbar updates after each prompt; `tokens: Xk in / Yk out · N reqs · ctx: Mk/200k (P%)` is present.

5. **`/context` growth table** prints deltas per turn; headroom math is correct against `model_specs.toml`.

6. **`/clear` resets** cumulative counters; toolbar redraws with zeros; subsequent calls accumulate from zero.

7. **Direct LLM call tracked.** Trigger a tool result over 4k tokens (e.g. read a large file). A `call_type='direct_llm'` row appears under the active attempt with `tokens_in`/`tokens_out` matching the summariser's `response.usage`.

8. **Unknown model falls back.** Set `JAC_MODEL=anthropic:claude-totally-fake` (if the model gateway lets the call through, even with an error) — `spec_for(...)` returns `defaults.max_context` and the toolbar still renders.

9. **No `CostUpdated` references** remain. `rg CostUpdated src tests` returns nothing.

10. **`just qa` green.** `just precommit-run` clean.

---

## Risks / open questions

- **Direct-LLM tokens double-counted?** When `model_request(...)` runs inside a tool call during an outer agent run, are its tokens already in the outer `result.usage()`? Pydantic AI's `direct` API is a separate code path; likely **not** rolled up. Verify with a small probe early in implementation. If not rolled up, the implementation in Step 5 must bump `session.cumulative_*` directly when persisting the direct_llm row, and root completions must subtract direct_llm children from the "self" computation alongside agent children. Document the answer in `docs/dev/runtime-layer.md`.

- **Multiple concurrent `spawn_minion` calls** (Pydantic AI auto-scheduling several in a turn): each one snapshots `ctx.usage` independently. Snapshots taken at the same instant return the same numbers; deltas computed at completion time will overlap if the runs interleave. **Mitigation:** wrap the snapshot+call+delta in an `asyncio.Lock` per parent attempt, or compute deltas via per-run usage objects (`Usage()` not shared) and only roll up at root. Recommend the latter — minions' attempts get their own `Usage()` instance and we re-add to parent at attempt close. This may require a small change to `spawn_minion`'s `usage=ctx.usage` plumbing; verify on the way in.

- **TOML packaging.** Confirm the build backend (`hatch` per `pyproject.toml`) actually includes `src/jac/data/*.toml` in the wheel. `importlib.resources.files("jac.data")` must work both in editable installs and in `uv tool install`.

- **`/context` "notes" column noise.** If the heuristic picks bad descriptions, ship without it and revisit.

---

## Out-of-scope reminders

- No $ cost.
- No tokenizer pass for per-message context split.
- No auto-`/compact` triggers.
- No cross-session usage history.
- No new providers in `model_specs.toml` beyond what we use today; appended later as needed.
