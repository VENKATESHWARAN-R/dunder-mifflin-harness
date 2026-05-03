# C6 — Scott (Manager) + Jim (Builder)

> **Status:** Ready for implementation  
> **Last revised:** 2026-05-04  
> **Depends on:** C5 (shipped)  
> **Roadmap entry:** [`docs/ROADMAP.md` §C6](../ROADMAP.md#c6--scott-manager--jim-builder)  
> **Design record:** [`lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md`](../../lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md)  
> **Schema:** [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md) v1.3

---

## Goals

1. Replace the current single generic `chat` agent with **Michael Scott** (`manager` role) as JAC's default and persistent agent.
2. Wire **Jim Halpert** (`builder` role) as Scott's first specialist, called via Pydantic AI agent-as-tool delegation.
3. Activate the `attempts` table with a parent→child call tree (`parent_attempt_id`).
4. Persist `persona` and `display_name` on `agent_configs` rows so events and the CLI can show character names.

## Non-goals (C6b and later)

- Pam (analyst), Date Mike (planner), Dwight (evaluator) — C6b / C9
- `context_store` write/read — C6b
- `tasks` table population — C6b
- Cost fields (`tokens_in/out`, `cost`, `CostUpdated` event) fully wired — C7
- Graph orchestration — C10
- Minion spawning / Holly — C14

---

## Key files and integration points

| File | What it is | C6 touches it? |
|---|---|---|
| `src/jac/agents/base.py` | Factory (`config_loader`). The only place `Agent(...)` is called. | Yes — extend `AgentConfig` dataclass; add `summon_jim` tool construction |
| `src/jac/agents/seeds.py` | Idempotent DB seeding for `agent_configs` rows | Yes — replace `chat` seed with `manager` (Scott) + `builder` (Jim) seeds |
| `src/jac/agents/personas.py` | **New file.** System prompts and persona metadata for each character | Yes — create |
| `src/jac/agents/tools.py` | **New file.** Scott's delegation tools (`summon_jim`) | Yes — create |
| `src/jac/runtime/coordinator.py` | Facade the CLI talks to; calls `config_loader` | Yes — swap role from `chat` to `manager` |
| `src/jac/runtime/events.py` | Typed events | Yes — add `AgentDelegated`, `AttemptRecorded` events |
| `src/jac/state/attempts.py` | **New file.** `AttemptsRepo` for the `attempts` table | Yes — create |
| `src/jac/state/db.py` | `StateStore`; wires repos | Yes — add `AttemptsRepo` |
| `src/jac/state/agent_configs.py` | `AgentConfigsRepo` | Yes — add `persona`/`display_name` to `AgentConfigRow`; update `create`/`get_by_run_and_role` |
| `src/jac/state/migrations/002_c6_scott_jim.sql` | **New migration.** Adds columns to `agent_configs`; adds `attempts` table | Yes — create |
| `tests/test_c6_scott_jim.py` | **New test file** | Yes — create |

---

## Step-by-step implementation

### Step 1 — DB migration (`002_c6_scott_jim.sql`)

New file: `src/jac/state/migrations/002_c6_scott_jim.sql`

This migration runs as schema v1.3 (migration ordinal 002 → `_ordinal_to_version_string(2)` = `"1.1"`).

> **Note on version mapping:** The existing `db.py` maps ordinal 001 → `"1.0"`, 002 → `"1.1"`, etc. Schema version 1.3 in the contract doc is the *semantic* version; the migration ordinal is independent. The `schema_meta` row after this migration will read `"1.1"`.

```sql
-- C6 migration. Mirrors docs/contracts/STATE_SCHEMA.md (schema v1.3).
-- Adds persona/minion fields to agent_configs; creates attempts table.

-- agent_configs: new columns from schema v1.3
ALTER TABLE agent_configs ADD COLUMN persona        TEXT;
ALTER TABLE agent_configs ADD COLUMN display_name   TEXT;
ALTER TABLE agent_configs ADD COLUMN is_minion      INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_configs ADD COLUMN parent_role    TEXT;
ALTER TABLE agent_configs ADD COLUMN depth          INTEGER NOT NULL DEFAULT 0;

-- attempts: full table (was referenced in schema but not yet created)
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id          TEXT PRIMARY KEY,
    task_id             TEXT,                                   -- nullable at C6; required once tasks table is live (C6b)
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    parent_attempt_id   TEXT REFERENCES attempts(attempt_id),  -- nullable; call tree
    call_type           TEXT NOT NULL DEFAULT 'agent',
    role                TEXT NOT NULL DEFAULT 'builder',
    model               TEXT NOT NULL,
    tier                TEXT NOT NULL,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    cost                REAL NOT NULL DEFAULT 0.0,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    eval_score          REAL,
    eval_passed         INTEGER,
    eval_feedback       TEXT,
    status              TEXT NOT NULL DEFAULT 'running',
    created_at          TEXT NOT NULL
);

INSERT OR REPLACE INTO schema_meta VALUES ('version', '1.1');
```

> `task_id` is nullable at C6 because the `tasks` table isn't populated until C6b. Once C6b ships, attempts written by Jim will reference a real `task_id`.

---

### Step 2 — Persona metadata (`src/jac/agents/personas.py`)

New file. Centralises all character names, display names, default tiers, and system prompts. No external dependencies — pure Python constants.

```python
"""Character personas for JAC's manager-specialist cast.

Each entry is the source of truth for the persona name, display name,
default model tier, and system prompt. Tiers and prompts can be overridden
at seed time via config or environment.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Persona:
    role: str           # semantic role stored in agent_configs.role
    persona: str        # full character name shown in events
    display_name: str   # short name shown in CLI output
    default_tier: str   # scout | worker | architect


PERSONAS: dict[str, Persona] = {
    "manager": Persona(
        role="manager",
        persona="Michael Scott",
        display_name="Scott",
        default_tier="worker",
    ),
    "builder": Persona(
        role="builder",
        persona="Jim Halpert",
        display_name="Jim",
        default_tier="worker",
    ),
}

# System prompts — kept here so seeds.py stays clean.

SCOTT_SYSTEM_PROMPT = """\
You are Michael Scott, the manager of JAC — an agentic coding harness.
Your job is to help the user and route work to the right specialist.

ROUTING RULES:
- For simple questions, greetings, explanations, or anything that doesn't
  require writing or running code: answer directly yourself.
- For tasks that require writing, editing, or executing code (new scripts,
  features, bug fixes, refactors): call summon_jim and hand the full task
  description to Jim.
- When in doubt, answer directly rather than delegating — Jim is for real
  coding work, not quick lookups.

RESPONSE STYLE:
- Friendly and direct. Skip unnecessary preamble.
- When delegating, tell the user briefly what you're handing to Jim.
- When Jim returns a result, summarise it clearly for the user.
"""

JIM_SYSTEM_PROMPT = """\
You are Jim Halpert, the builder in JAC's agentic harness.
You receive a specific coding task from Scott and execute it completely.

YOUR JOB:
- Read the task description carefully.
- Use your file and shell tools to implement the task.
- Run the code to verify it works.
- Return a concise summary of what you did and the outcome.

RULES:
- Work only within the current workspace directory.
- Do not ask clarifying questions — implement based on what you have.
- If you hit an ambiguity, make a reasonable choice and note it in your summary.
- Your output is read by Scott and shown to the user, so keep it clear.
"""
```

---

### Step 3 — Attempts repo (`src/jac/state/attempts.py`)

New file. Minimal CRUD for the `attempts` table — only what C6 needs (create + update status).

```python
"""Repository for the `attempts` table."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class AttemptRow:
    attempt_id: str
    task_id: str | None
    run_id: str
    parent_attempt_id: str | None
    call_type: str
    role: str
    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    cost: float
    duration_ms: int
    eval_score: float | None
    eval_passed: int | None
    eval_feedback: str | None
    status: str
    created_at: str


class AttemptsRepo:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        role: str,
        model: str,
        tier: str,
        task_id: str | None = None,
        parent_attempt_id: str | None = None,
        call_type: str = "agent",
    ) -> AttemptRow:
        now = _now()
        attempt_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO attempts
                (attempt_id, task_id, run_id, parent_attempt_id,
                 call_type, role, model, tier, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (attempt_id, task_id, run_id, parent_attempt_id,
             call_type, role, model, tier, now),
        )
        await self._connection.commit()
        return AttemptRow(
            attempt_id=attempt_id, task_id=task_id, run_id=run_id,
            parent_attempt_id=parent_attempt_id, call_type=call_type,
            role=role, model=model, tier=tier, tokens_in=0, tokens_out=0,
            cost=0.0, duration_ms=0, eval_score=None, eval_passed=None,
            eval_feedback=None, status="running", created_at=now,
        )

    async def update_status(self, attempt_id: str, status: str) -> None:
        await self._connection.execute(
            "UPDATE attempts SET status = ? WHERE attempt_id = ?",
            (status, attempt_id),
        )
        await self._connection.commit()
```

---

### Step 4 — Wire `AttemptsRepo` into `StateStore` (`src/jac/state/db.py`)

Add to imports and `StateStore.__init__`:

```python
from jac.state.attempts import AttemptsRepo

class StateStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        ...
        self.attempts = AttemptsRepo(connection)   # add this line
```

---

### Step 5 — Extend `AgentConfigRow` and repo (`src/jac/state/agent_configs.py`)

Add `persona`, `display_name`, `is_minion`, `parent_role`, `depth` to `AgentConfigRow`. Update `create`, `get_by_run_and_role`, and `_agent_config_from_row` to handle the new columns.

Key changes only (not a full rewrite):

**`AgentConfigRow`** — add five fields:
```python
persona: str | None
display_name: str | None
is_minion: int          # 0 or 1
parent_role: str | None
depth: int
```

**`create()`** — add parameters `persona`, `display_name`, `is_minion=0`, `parent_role=None`, `depth=0`. Extend the `INSERT` statement to include the five new columns.

**`_agent_config_from_row()`** — read the five new columns from the row dict using `.get()` with sensible defaults to stay compatible if the row predates this migration:
```python
persona=row["persona"],
display_name=row["display_name"],
is_minion=row["is_minion"] if "is_minion" in row.keys() else 0,
parent_role=row["parent_role"],
depth=row["depth"] if "depth" in row.keys() else 0,
```

**`AgentConfig`** dataclass in `base.py** — mirror the same five new fields so `config_loader` can access them.

---

### Step 6 — New events (`src/jac/runtime/events.py`)

Add two events. Append to the existing event definitions:

```python
@dataclass(frozen=True, slots=True)
class AgentDelegated(RuntimeEvent):
    """Emitted when Scott hands work to a specialist."""
    from_role: str      # 'manager'
    to_role: str        # 'builder'
    persona: str        # 'Jim Halpert'
    display_name: str   # 'Jim'
    task_summary: str   # first 120 chars of the task


@dataclass(frozen=True, slots=True)
class AttemptRecorded(RuntimeEvent):
    """Emitted when an attempt row is created in the DB."""
    attempt_id: str
    role: str
    call_type: str
    parent_attempt_id: str | None
```

---

### Step 7 — Scott's delegation tools (`src/jac/agents/tools.py`)

New file. Contains the `make_summon_jim` factory — a function that closes over the runtime context (state, session, events) and returns a Pydantic AI tool function that Scott can call.

```python
"""Delegation tools for the manager (Scott) agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import RunContext

if TYPE_CHECKING:
    from jac.agents.base import AgentDeps


def make_summon_jim_tool(state, settings, session, events):
    """Return a tool function that Scott can call to delegate to Jim."""

    async def summon_jim(ctx: RunContext["AgentDeps"], task: str) -> str:
        """Delegate a coding task to Jim Halpert (builder).

        Args:
            task: Full description of the coding task for Jim to execute.
        """
        from jac.agents.base import config_loader
        from jac.agents.personas import PERSONAS
        from jac.runtime.events import AgentDelegated, AttemptRecorded

        jim_persona = PERSONAS["builder"]

        # Emit delegation event so CLI can show "Handing to Jim..."
        await events.emit(AgentDelegated(
            from_role="manager",
            to_role="builder",
            persona=jim_persona.persona,
            display_name=jim_persona.display_name,
            task_summary=task[:120],
        ))

        # Record Jim's attempt row (task_id is None at C6; set at C6b)
        # Model/tier resolved after agent is built — store provisional values
        jim_attempt = await state.attempts.create(
            run_id=session.run_id,
            role="builder",
            model=settings.resolve_model_selection(
                model_override=None, tier=jim_persona.default_tier
            ).model_id,
            tier=jim_persona.default_tier,
            parent_attempt_id=ctx.run_context.run_id if hasattr(ctx, "run_context") else None,
            call_type="agent",
        )
        await events.emit(AttemptRecorded(
            attempt_id=jim_attempt.attempt_id,
            role="builder",
            call_type="agent",
            parent_attempt_id=jim_attempt.parent_attempt_id,
        ))

        # Build Jim and run
        jim_agent = await config_loader(
            state=state,
            settings=settings,
            run_id=session.run_id,
            role="builder",
            events=events,
        )

        async with jim_agent:
            result = await jim_agent.run(task)

        output = result.output
        await state.attempts.update_status(jim_attempt.attempt_id, "passed")
        return output

    return summon_jim
```

> **Note on `parent_attempt_id`:** At C6 we don't yet thread the parent Scott attempt ID through via `RunContext` cleanly — `ctx.run_context` is illustrative. The precise approach depends on how Scott's own attempt row is created (Step 8). Implementer: if threading is awkward, store Scott's `attempt_id` in `session` temporarily for Jim to reference.

---

### Step 8 — Update `seeds.py` — replace `chat` with Scott + Jim

Replace `ensure_default_run_config` with two seed functions: `ensure_manager_config` and `ensure_builder_config`. Also keep a thin `ensure_default_run_config` shim that calls `ensure_manager_config` so existing call sites don't break.

```python
from jac.agents.personas import PERSONAS, SCOTT_SYSTEM_PROMPT, JIM_SYSTEM_PROMPT

async def ensure_manager_config(state, run_id, *, model_override=None) -> AgentConfig:
    """Idempotently seed the manager (Scott) agent_configs row."""
    persona = PERSONAS["manager"]
    return await _ensure_role_config(
        state, run_id,
        role="manager",
        persona=persona.persona,
        display_name=persona.display_name,
        model_tier=persona.default_tier,
        model_override=model_override,
        system_prompt=SCOTT_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )

async def ensure_builder_config(state, run_id, *, model_override=None) -> AgentConfig:
    """Idempotently seed the builder (Jim) agent_configs row."""
    persona = PERSONAS["builder"]
    return await _ensure_role_config(
        state, run_id,
        role="builder",
        persona=persona.persona,
        display_name=persona.display_name,
        model_tier=persona.default_tier,
        model_override=model_override,
        system_prompt=JIM_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )

async def ensure_default_run_config(state, run_id, **kwargs) -> AgentConfig:
    """Shim — seeds manager config. Preserves old call sites."""
    return await ensure_manager_config(state, run_id)
```

---

### Step 9 — Update `coordinator.py`

Three changes:

1. **Seed both roles** at `_ensure_agent` time:
   ```python
   from jac.agents import ensure_manager_config, ensure_builder_config
   await ensure_manager_config(self.state, self.session.run_id)
   await ensure_builder_config(self.state, self.session.run_id)
   ```

2. **Build Scott's agent with Jim as a tool.** In `build_agent`, after calling `config_loader` for `manager`, attach the Jim tool:
   ```python
   from jac.agents.tools import make_summon_jim_tool
   scott = await config_loader(..., role="manager")
   jim_tool = make_summon_jim_tool(self.state, self.settings, self.session, self.events)
   scott = scott.clone(tools=[*scott._function_tools.values(), jim_tool])
   ```
   > If Pydantic AI doesn't expose `clone`, register the tool before `Agent(...)` is constructed — rework in the factory instead. See note below on factory approach.

3. **Record Scott's own attempt row** before `agent.run(...)`:
   ```python
   if self.state is not None:
       scott_attempt = await self.state.attempts.create(
           run_id=run_id,
           role="manager",
           model=<resolved model id>,
           tier=<resolved tier>,
           call_type="agent",
       )
   ```

> **Factory approach for Jim tool:** Pydantic AI's `Agent` takes `tools=` at construction time. The cleanest path is to pass a `delegation_tools` parameter to `config_loader` so the factory can include them when building Scott. Alternatively, build Jim's tool function first and pass it through `config_loader`'s tool resolution. The implementer should pick whichever approach keeps `Agent(...)` exclusively inside `base.py`. Do **not** call `Agent(...)` outside the factory.

---

### Step 10 — Update `__init__.py` exports

`src/jac/agents/__init__.py` — export the new seed functions:
```python
from jac.agents.seeds import (
    ensure_default_run_config,
    ensure_manager_config,
    ensure_builder_config,
)
```

`src/jac/state/__init__.py` — export `AttemptsRepo` if other modules need it directly.

---

### Step 11 — CLI renderer: show delegation

`src/jac/cli/renderer.py` — add a handler for `AgentDelegated`:
```python
case AgentDelegated():
    console.print(f"[dim]→ Handing to {event.display_name}…[/dim]")
```

---

## Acceptance checks

Run after implementation:

```bash
just test                              # full test suite
uv run jac "what is 2 plus 2"         # Scott answers directly; no Jim call
uv run jac "write a python script that prints fibonacci(10)"  # Scott delegates to Jim
```

For the delegation case, verify:
- [ ] CLI shows "→ Handing to Jim…" before Jim's output
- [ ] Two rows in `attempts`: one for Scott (`role='manager'`, `parent_attempt_id=NULL`), one for Jim (`role='builder'`, `parent_attempt_id=<scott's id>`)
- [ ] Jim's file (`fibonacci.py` or similar) written to disk and run output visible
- [ ] `agent_configs` has two rows with correct `persona` / `display_name` values

---

## Tests (`tests/test_c6_scott_jim.py`)

Test scope — keep to what can run without a live LLM (mock the agent output):

1. `test_manager_seed_idempotent` — call `ensure_manager_config` twice; assert one DB row
2. `test_builder_seed_idempotent` — same for builder
3. `test_agent_config_row_has_persona` — seed manager; assert `persona == "Michael Scott"`, `display_name == "Scott"`
4. `test_attempt_create_and_update` — create an attempt row, call `update_status`, assert DB reflects it
5. `test_attempt_parent_child` — create Scott attempt, create Jim attempt with `parent_attempt_id=scott.attempt_id`, assert FK relation holds
6. `test_migration_002_applies` — open a fresh DB, confirm `attempts` table exists and `agent_configs` has `persona` column

---

## Version bump

Bump `pyproject.toml` version `0.x.y → 0.x.(y+1)` and update the fallback in `src/jac/__init__.py` to match. New manager-based routing is a user-visible behavior change.

---

## Open implementer notes

- **Jim tool registration:** Pydantic AI requires tools to be registered at `Agent(...)` construction. The cleanest approach is to add an optional `extra_tools: list[ToolFn] | None` parameter to `config_loader` that gets merged with local tools from `allowed_tools`. That keeps `Agent(...)` inside the factory and lets coordinator pass Jim's delegation tool for Scott's build.
- **`parent_attempt_id` threading:** Scott's attempt row needs to exist before Jim runs so Jim can reference it. Create Scott's row in coordinator before `agent.run()`, store the `attempt_id` in `session.active_attempt_id` (add this field to `SessionState`), and read it from `session` inside the Jim tool.
- **Scott's `task_id`:** At C6, `task_id=None` on Scott's attempt row is correct — tasks aren't tracked until C6b.
