# C6b — Pam (Planner) + Slash-Mode Addendums

> **Status:** Ready for implementation
> **Last revised:** 2026-05-06
> **Depends on:** C6 (shipped 2026-05-04)
> **Roadmap entry:** [`docs/ROADMAP.md` §C6b](../ROADMAP.md#c6b--pam-planner--slash-mode-addendums)
> **Design record:** [`lab/brainstorm/2026-05-05-multi-agent-cast-final.md`](../../lab/brainstorm/2026-05-05-multi-agent-cast-final.md)
> **Schema:** [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md) v1.3 (`tasks` activates here)

---

## Goals

1. Add **Pam Beesly** (`planner` role, Architect tier) to the cast — the only specialist between Scott and the build loop.
2. Light up **dynamic system-prompt addendums** so a slash command can put an agent into a focused mode without spawning a new persona.
3. Wire **`/plan <task>`** — routes directly to Pam, returns a structured `Plan` rendered to the terminal, and writes one `tasks` row per planned task.
4. Wire **`/init`** — routes to Scott in init mode, surveys the workspace, writes `<repo>/AGENTS.md` (or `<cwd>/AGENTS.md` in no-project mode).
5. Filter intermediate `ToolCallPart` / `ToolReturnPart` blocks from message history during slash-command runs so Scott sees the slash prompt and the final artifact next turn — nothing else.
6. Activate the `tasks` table.

## Non-goals (later components)

- Dwight (evaluator), retry/escalation loop, `/build`, `/eval` — **C9**.
- `summon_pam` from Scott as a tool-delegation primitive — natural to land alongside C9 when Scott orchestrates the full Pam→Jim→Dwight loop. C6b only exposes Pam through `/plan`.
- Universal `spawn_minion`, `read_file_smart`, tool result interception — **C6c**. `/init` on a large multi-module repo therefore does an inline survey at C6b (no parallel module minions yet); the brainstorm explicitly accepts this.
- `context_store` activation — the new brainstorm collapses analyst-Pam into Scott, so the v0 plan does not need a shared brief store. Defer to whichever later component first needs cross-agent state outside the message history (likely C11).
- Strategy auto-selection beyond `feature_by_feature`. Pam emits `dev_strategy` in her structured output, but `feature_by_feature` is the only branch wired in v0 (TDD/spec-driven/agile follow at C22).
- `pydantic_graph` orchestration — **C10**.
- Cost columns / `CostUpdated` — **C7** (Pam's attempts row is created at C6b but `tokens_in/out`/`cost` stay zero until C7).

---

## Key files and integration points

| File | What it is | C6b touches it? |
|---|---|---|
| `src/jac/agents/personas.py` | Persona registry + system prompts | Yes — add `Persona("planner", "Pam Beesly", "Pam", "architect")` and `PAM_SYSTEM_PROMPT` |
| `src/jac/agents/seeds.py` | Idempotent `agent_configs` seeding | Yes — add `ensure_planner_config(...)`; export from `agents/__init__.py` |
| `src/jac/agents/modes.py` | **New file.** `MODE_PROMPTS` registry + `build_instructions(base, mode)` helper | Yes — create |
| `src/jac/agents/plans.py` | **New file.** `Plan` / `PlannedTask` Pydantic models used as `output_type=` for Pam | Yes — create |
| `src/jac/agents/base.py` | `config_loader` factory | Yes — add optional `instructions_addendum: str \| None` param threaded into the static `instructions=` string |
| `src/jac/state/tasks.py` | **New file.** `TasksRepo` for the `tasks` table | Yes — create |
| `src/jac/state/db.py` | `StateStore`; wires repos | Yes — add `self.tasks = TasksRepo(connection)` |
| `src/jac/state/__init__.py` | Public state surface | Yes — export `TaskRow`, `TasksRepo` |
| `src/jac/runtime/session.py` | `SessionConfig` | Yes — add `slash_mode: str \| None = None` |
| `src/jac/runtime/coordinator.py` | Facade the CLI talks to | Yes — add `submit_slash_run(...)` for one-turn role+addendum runs |
| `src/jac/runtime/events.py` | Typed events | Yes — add `PlanGenerated`, `WorkspaceSurveyCompleted` events |
| `src/jac/runtime/history.py` | **New file.** `filter_tool_noise(messages)` helper | Yes — create |
| `src/jac/cli/app.py` | Slash command registry + handlers | Yes — register `/plan` and `/init` handlers |
| `src/jac/cli/renderer.py` | Rich rendering | Yes — render `Plan` panels + `WorkspaceSurveyCompleted` summaries |
| `docs/contracts/CLI_DESIGN.md` | Locked CLI contract | Yes — slash-command rule changes (see **Contract updates** below) |
| `docs/dev/agents-layer.md` | Layer doc | Yes — document modes and Pam |
| `docs/dev/runtime-layer.md` | Layer doc | Yes — document `submit_slash_run` and history filter |
| `docs/ROADMAP.md` | Roadmap | Yes — flip C6b to Done with ship notes |
| `pyproject.toml` + `src/jac/__init__.py` | Version | Yes — bump (user-visible new commands) |
| `tests/test_c6b_planner_and_slash_modes.py` | **New test file** | Yes — create |

No SQL migration is required. The `tasks` table already exists from migration `001_initial.sql`; `attempts.task_id` was made nullable in `002_c6_scott_jim.sql` and stays nullable for Scott's manager attempts.

---

## Step-by-step implementation

### Step 1 — Pam persona (`src/jac/agents/personas.py`)

Add Pam to `PERSONAS` and define her system prompt. Keep tone consistent with Scott/Jim (one-paragraph role statement, then bulleted rules).

```python
PERSONAS["planner"] = Persona(
    role="planner",
    persona="Pam Beesly",
    display_name="Pam",
    default_tier="architect",
)

PAM_SYSTEM_PROMPT = """\
You are Pam Beesly, the planner in JAC's agentic harness.
You read a coding requirement and produce a clear, actionable plan.

YOUR JOB:
- Pick a development strategy. Default to 'feature_by_feature' unless the
  requirement strongly signals otherwise (test-heavy domain → 'tdd';
  tight contract → 'spec_driven'; exploratory → 'feature_by_feature').
- Decompose the work into ordered tasks. Each task has a short title, a
  description detailed enough for a builder to execute without you, and
  acceptance criteria the evaluator will grade against.
- Mark complexity per task: simple | moderate | complex.

RULES:
- Do not write code. The builder (Jim) writes code; you plan.
- Do not gather requirements through clarifying questions. Plan from what
  you have; flag unknowns as risks in the relevant task description.
- Prefer fewer, larger tasks over many tiny ones. Aim for 3–7 tasks for a
  feature-shaped request.
- Acceptance criteria must be checkable: "command exits 0", "file X
  contains Y", "function Z returns W for input V". Avoid vague verbs.
"""
```

> The cast docstring at the top of the file should be updated to mention three personas, not two.

### Step 2 — `Plan` structured output (`src/jac/agents/plans.py`)

New file. Pydantic models that become Pam's `output_type`. Keep them aligned with the `tasks` schema columns.

```python
"""Structured outputs emitted by the planner (Pam)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

DevStrategy = Literal["feature_by_feature", "tdd", "spec_driven", "agile"]
Complexity = Literal["simple", "moderate", "complex"]


class PlannedTask(BaseModel):
    title: str = Field(..., description="Short imperative title")
    description: str = Field(..., description="Detail sufficient for Jim to act without Pam")
    acceptance_criteria: str = Field(..., description="Checkable conditions for Dwight")
    complexity: Complexity = "moderate"


class Plan(BaseModel):
    summary: str = Field(..., description="One-paragraph overview of the plan")
    dev_strategy: DevStrategy = "feature_by_feature"
    tasks: list[PlannedTask]
```

### Step 3 — Mode addendum registry (`src/jac/agents/modes.py`)

New file. Pure constants + a `build_instructions` helper that composes a base prompt with an optional mode addendum.

```python
"""Slash-mode addendums for system prompts.

A slash command (/init, /plan) sets `session.config.slash_mode`. The agent
factory composes the matching addendum into the static `instructions=`
string at build time. The addendum lives outside the message history,
so it does not pollute future turns once the slash run completes.
"""
from __future__ import annotations

INIT_MODE_ADDENDUM = """\
SLASH MODE: /init

You are surveying the user's workspace to produce an `AGENTS.md` file at
the project root. The file is read by future agent runs as project-level
instructions, so be precise and durable.

PROCESS:
1. List the top of the project tree (`list_directory` on cwd; one or two
   levels deep). Note the package manager, language, frameworks, build
   tool, test runner, and entrypoints.
2. Read 2–4 anchor files: README, top-level config (pyproject.toml /
   package.json), and the most central module if obvious.
3. Compose AGENTS.md with these sections, in order:
     - Project name and one-line description
     - How to install / run / test (commands, not prose)
     - Repo layout (one bullet per top-level dir)
     - Conventions worth knowing (lint, format, type-check, naming)
4. Write AGENTS.md to the project root using `write_file`.

RULES:
- Do not invent commands. If you cannot tell, say "TODO: confirm with user".
- Keep it under 120 lines. Future agents skim it; do not write a tutorial.
- Do not commit, push, or modify any other file.
"""

PLAN_MODE_ADDENDUM = """\
SLASH MODE: /plan

You are producing a one-shot Plan for the user's request. Output the
structured Plan only — no narration. The user will review the plan
before any builder is summoned.
"""

MODE_PROMPTS: dict[str, str] = {
    "init": INIT_MODE_ADDENDUM,
    "plan": PLAN_MODE_ADDENDUM,
}


def build_instructions(base: str, mode: str | None) -> str:
    """Compose a base system prompt with the slash-mode addendum, if any."""
    if mode is None:
        return base
    addendum = MODE_PROMPTS.get(mode)
    if not addendum:
        return base
    return f"{base}\n\n---\n\n{addendum}"
```

> **Why static composition rather than `@agent.instructions`?** Pydantic AI's dynamic-instructions decorator requires a `RunContext` with deps, and the rest of the codebase does not yet use the deps system. Static composition matches the existing `_compose_system_prompt` pattern (which already concatenates skill content), keeps the seam in one place, and avoids introducing deps just for two slash modes. If C20 (hot-reload) needs deps later, swap to `@agent.instructions` then; the addendum text and the `MODE_PROMPTS` registry stay unchanged.

### Step 4 — Factory accepts an instructions addendum (`src/jac/agents/base.py`)

Extend `config_loader` with an `instructions_addendum: str | None = None` keyword-only parameter. Thread it into `_compose_system_prompt` so the order is: base prompt → skills block → addendum. Skills come before the addendum so a slash mode can refer to "the conventions above" without breaking when no skills are active.

Pseudocode:

```python
async def config_loader(..., instructions_addendum: str | None = None) -> Agent:
    ...
    composed_prompt = await _compose_system_prompt(
        state, cfg.system_prompt, run_id, role
    )
    if instructions_addendum:
        composed_prompt = f"{composed_prompt}\n\n---\n\n{instructions_addendum}"
    ...
```

The `output_type=` parameter already exists on `config_loader` — Pam re-uses it with `output_type=Plan`.

### Step 5 — `ensure_planner_config` (`src/jac/agents/seeds.py`)

Mirror `ensure_builder_config`. Tools: `["filesystem", "shell"]` for v0 — Pam needs `read_file` / `list_directory` to scan requirements when invoked from a workspace, and shell access for quick env probes. Architect tier per the persona.

```python
async def ensure_planner_config(
    state: StateStore,
    run_id: str,
    *,
    model_tier: str | None = None,
    model_override: str | None = None,
) -> AgentConfig:
    p = PERSONAS["planner"]
    tier = model_tier or p.default_tier
    return await _ensure_role_config(
        state, run_id,
        role="planner",
        persona=p.persona,
        display_name=p.display_name,
        model_tier=tier,
        model_override=model_override,
        system_prompt=PAM_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )
```

Export from `src/jac/agents/__init__.py`. Call it alongside `ensure_manager_config` / `ensure_builder_config` in `RunCoordinator._ensure_agent` so Pam's row exists from the first turn (cheap: idempotent insert, no model call until `/plan` fires).

> **Tier resolution caveat.** `_ensure_agent` currently overrides the seeded tier with `session.config.tier` if the user passed `--tier`. Do **not** apply that override for Pam's row — her default is Architect by design and the user-set tier is meant for Scott. Either guard the override (`if role == "manager"`) or accept divergence and let `/tier` continue to apply to Scott only. Pick guard; it's the cheaper change and matches the brainstorm's per-persona-default rationale.

### Step 6 — Tasks repo (`src/jac/state/tasks.py`)

New file. CRUD for `tasks`. C6b only writes; reads happen at C9 / C11.

```python
"""Repository for the `tasks` table."""
from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import aiosqlite


@dataclass(frozen=True, slots=True)
class TaskRow:
    task_id: str
    run_id: str
    title: str
    description: str
    acceptance_criteria: str
    status: str
    complexity: str
    tier: str
    attempt_count: int
    order_index: int
    parent_task_id: str | None


class TasksRepo:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create_many(
        self,
        run_id: str,
        tasks: list[dict],   # title, description, acceptance_criteria, complexity
    ) -> list[TaskRow]:
        rows: list[TaskRow] = []
        for index, t in enumerate(tasks):
            row = await self.create(
                run_id=run_id,
                title=t["title"],
                description=t["description"],
                acceptance_criteria=t["acceptance_criteria"],
                complexity=t.get("complexity", "moderate"),
                order_index=index,
            )
            rows.append(row)
        return rows

    async def create(
        self,
        *,
        run_id: str,
        title: str,
        description: str,
        acceptance_criteria: str,
        complexity: str = "moderate",
        tier: str = "worker",
        order_index: int,
        parent_task_id: str | None = None,
    ) -> TaskRow:
        task_id = uuid4().hex
        await self._connection.execute(
            """INSERT INTO tasks
               (task_id, run_id, title, description, acceptance_criteria,
                status, complexity, tier, attempt_count, order_index, parent_task_id)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, 0, ?, ?)""",
            (task_id, run_id, title, description, acceptance_criteria,
             complexity, tier, order_index, parent_task_id),
        )
        await self._connection.commit()
        return TaskRow(
            task_id=task_id, run_id=run_id, title=title, description=description,
            acceptance_criteria=acceptance_criteria, status="pending",
            complexity=complexity, tier=tier, attempt_count=0,
            order_index=order_index, parent_task_id=parent_task_id,
        )

    async def list_for_run(self, run_id: str) -> list[TaskRow]: ...  # used at C9/C11
```

Wire into `StateStore.__init__`. Export from `src/jac/state/__init__.py`.

### Step 7 — History filter (`src/jac/runtime/history.py`)

New file. One pure function; no I/O.

```python
"""Filters for Pydantic AI message history."""
from __future__ import annotations
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse,
    UserPromptPart, TextPart,
)


def filter_tool_noise(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip ToolCallPart / ToolReturnPart from a message list.

    Used after a slash-command run so the persisted conversation contains
    the user's slash prompt and the final assistant artifact only — no
    intermediate tool chatter that would inflate context for future turns.
    """
    cleaned: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_parts = [p for p in msg.parts if isinstance(p, UserPromptPart)]
            if user_parts:
                cleaned.append(ModelRequest(parts=user_parts))
        elif isinstance(msg, ModelResponse):
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            if text_parts:
                cleaned.append(ModelResponse(parts=text_parts))
    return cleaned
```

> Keep both Pydantic AI imports under `from pydantic_ai.messages import ...`. The coordinator already imports `ModelRequest`, `ModelResponse`, `TextPart`, and `UserPromptPart`, so there is no new third-party surface.

### Step 8 — Session and coordinator wiring

**`SessionConfig`** (`src/jac/runtime/session.py`): add one field.

```python
slash_mode: str | None = None   # 'init' | 'plan' | None (set transiently by slash handlers)
```

**`RunCoordinator`** (`src/jac/runtime/coordinator.py`): add a method dedicated to slash-command runs. It does not replace `submit_message`; it shares the persistence + event surface but binds a specific role and addendum for one turn.

```python
async def submit_slash_run(
    self,
    *,
    role: str,
    prompt: str,
    addendum_mode: str,
    output_type: type | None = None,
    persist_user_prompt: str | None = None,
) -> object:
    """Run one prompt through a specified role with a mode addendum.

    - `role`: 'manager' | 'planner' | ... — must have an agent_configs row.
    - `prompt`: the prompt the model sees.
    - `addendum_mode`: key into MODE_PROMPTS; controls the static instructions tail.
    - `output_type`: optional Pydantic model used as Pam's structured output.
    - `persist_user_prompt`: what to write to messages.role='user' (defaults to `prompt`).
      Slash handlers usually pass the original `/plan ...` text so future turns see it.

    Returns the agent's parsed output (e.g. a `Plan` instance, or a string).
    Tool noise is filtered out of self._message_history before the next turn.
    """
```

Internals (sketch):

1. Reuse `_ensure_run_persisted`, `_configure_observability`, `RunStarted` event, message-append for `user`.
2. Build the role-specific agent via `config_loader(... role=role, output_type=output_type, instructions_addendum=MODE_PROMPTS.get(addendum_mode))`. Do **not** attach `summon_jim` to anyone except `manager`.
3. Create an `attempts` row with `role=role`, parented to the manager attempt if one is active in the session, otherwise top-level.
4. Run the agent with the existing `_message_history`.
5. On success: append `messages.role='assistant'` with a string render of the output (for `Plan`, the renderer produces a markdown summary; for the init flow, it's Scott's final text). Update `attempts.status='passed'`.
6. Replace `self._message_history` with `filter_tool_noise(result.all_messages())` so the noisy intermediate tool calls do not leak into the next turn.

The renderer for `/plan` formats the `Plan` object; the renderer for `/init` shows a one-line "AGENTS.md updated (N lines)" summary. Emit `PlanGenerated` / `WorkspaceSurveyCompleted` events so future surfaces can re-use the data without re-parsing terminal output.

### Step 9 — New events (`src/jac/runtime/events.py`)

```python
@dataclass(frozen=True, slots=True)
class PlanGenerated(RuntimeEvent):
    summary: str
    dev_strategy: str
    task_count: int


@dataclass(frozen=True, slots=True)
class WorkspaceSurveyCompleted(RuntimeEvent):
    agents_md_path: Path
    line_count: int
```

### Step 10 — Slash command handlers (`src/jac/cli/app.py`)

Inside `_register_commands`, register `/plan` and `/init`. Both handlers:

1. Set `self.session.config.slash_mode` to the matching mode key.
2. Call `coordinator.submit_slash_run(...)` with the right `role`, `output_type`, and prompt.
3. Reset `slash_mode` to `None` in a `finally` block.
4. Use the renderer to display the result.

```python
async def plan_command(args: str) -> None:
    task = args.strip()
    if not task:
        self.renderer.print_error("Usage: /plan <task description>")
        return
    self.session.config.slash_mode = "plan"
    try:
        plan = await self.coordinator.submit_slash_run(
            role="planner",
            prompt=task,
            addendum_mode="plan",
            output_type=Plan,
            persist_user_prompt=f"/plan {task}",
        )
    finally:
        self.session.config.slash_mode = None
    self.renderer.render_plan(plan)
    if self.state is not None:
        await self.state.tasks.create_many(
            self.session.run_id,
            [t.model_dump() for t in plan.tasks],
        )

async def init_command(_args: str) -> None:
    self.session.config.slash_mode = "init"
    try:
        await self.coordinator.submit_slash_run(
            role="manager",
            prompt="Survey this workspace and write an AGENTS.md at the project root.",
            addendum_mode="init",
            persist_user_prompt="/init",
        )
    finally:
        self.session.config.slash_mode = None
```

Register both in `_register_commands` with examples; tab-completion picks them up automatically via `commands.descriptions()`.

### Step 11 — Renderer

Add a small `render_plan(plan: Plan)` method that prints a Rich panel: `summary`, `dev_strategy`, then a table of `(#, title, complexity, acceptance)`. Mirror the visual weight of the existing `print_value` helper — do not add a heavyweight component.

### Step 12 — Contract updates

**`docs/contracts/CLI_DESIGN.md`** is `Locked` and currently states:

> All slash commands mutate `SessionConfig` or local session state — none send data to the model directly.

C6b violates this. Update the contract:

- Reword the rule: "Most slash commands mutate session state and do not send data to the model. A small set of named exceptions (`/plan`, `/init`, future `/build`, `/eval`, `/compact`, `/explore`) drive a model run on the user's behalf and append the resulting message to the session transcript."
- Add `/plan` and `/init` to the slash-command table with the model-run marker.
- Bump `Last revised` to today (2026-05-06).

**`docs/contracts/STATE_SCHEMA.md`** activation table: flip `tasks` from "C6b" to "C6b ✓" once shipped. No schema-shape change.

### Step 13 — Layer docs

- `docs/dev/agents-layer.md`: extend the cast section with Pam, document `instructions_addendum`, and note `output_type=Plan`.
- `docs/dev/runtime-layer.md`: add `submit_slash_run`, `slash_mode`, the history-filter helper, and the two new events.

### Step 14 — Version bump

`pyproject.toml` and `src/jac/__init__.py` fallback from `0.x.y → 0.x.(y+1)`. New CLI surface (`/plan`, `/init`) is user-visible.

### Step 15 — Tests (`tests/test_c6b_planner_and_slash_modes.py`)

LLM-free unit tests, modeled after `tests/test_c6_scott_jim.py`:

1. `test_planner_seed_idempotent` — call `ensure_planner_config` twice; assert one DB row, `persona == "Pam Beesly"`, `model_tier == "architect"`.
2. `test_mode_addendum_compose` — `build_instructions("BASE", "plan")` includes the plan addendum; `build_instructions("BASE", None)` returns `BASE` unchanged; unknown mode is a no-op.
3. `test_config_loader_threads_addendum` — build a planner agent with `instructions_addendum="X"`; introspect the resulting `Agent.instructions` string and assert the addendum is appended after the system prompt and any skills.
4. `test_filter_tool_noise_drops_tool_parts` — feed a synthetic message list with `ToolCallPart` / `ToolReturnPart` entries; assert only `UserPromptPart` and `TextPart` content survives, and empty messages are removed.
5. `test_tasks_create_many_ordered` — call `create_many` with three task dicts; assert `order_index` 0/1/2 and FK integrity to a real `runs` row.
6. `test_plan_command_writes_tasks` (using a stubbed coordinator that returns a fixed `Plan`) — assert `state.tasks.list_for_run(...)` returns the expected rows, `PlanGenerated` was emitted, and `slash_mode` is reset to `None` after the handler returns.
7. `test_slash_run_filters_history` — stub the agent to return a result whose `all_messages()` includes tool parts; assert `coordinator._message_history` after the call contains only user/text parts.
8. `test_init_command_persists_slash_prompt` — verify the literal `/init` string lands in `messages.role='user'` so Scott sees it next turn.

Smoke (manual, against a real model):

```bash
uv run jac "/plan build a tiny note-taking CLI"
uv run jac "/init"   # in a small repo
```

Verify:
- `/plan` renders the panel; `tasks` table has N rows with correct `order_index`; `attempts` has a `role='planner'` row with `tier='architect'`.
- `/init` writes `<repo>/AGENTS.md`; the file exists and is reasonably shaped.
- Issuing a follow-up plain prompt after `/plan` shows that Scott "sees" the slash interaction as a single user turn — no tool noise leaked.

---

## Acceptance checks

- [ ] `just test` green (existing + new)
- [ ] `just typecheck` clean
- [ ] `just lint` clean
- [ ] `uv run jac "/plan add a /search command"` returns a structured plan and writes `tasks` rows
- [ ] `uv run jac "/init"` writes a sane AGENTS.md and emits `WorkspaceSurveyCompleted`
- [ ] `attempts` has a row with `role='planner'`, `tier='architect'`, `parent_attempt_id` set when `/plan` is issued mid-conversation under Scott
- [ ] After a slash run, `messages.role='user'` shows the literal `/plan ...` (or `/init`) text, and `messages.role='assistant'` shows the final artifact only — no `<tool_call>` chatter
- [ ] Pam's `model_tier` does **not** flip to Worker when the user runs `/tier worker` for Scott
- [ ] `docs/contracts/CLI_DESIGN.md` updated (slash-command rule reworded, table extended, `Last revised` bumped)
- [ ] `docs/ROADMAP.md` C6b entry flipped to `done` with a Done-section ship note
- [ ] Version bumped in `pyproject.toml` and `src/jac/__init__.py`

---

## Risks and open implementer notes

- **`/plan` invoked outside an interactive run.** The current one-shot `jac "..."` path goes through `run_prompt` → `submit_message`, not through `ChatApp`. If we want `jac "/plan ..."` to also work as a one-shot, surface `submit_slash_run` from `run_prompt` by parsing leading slash commands there. Defer to a follow-up if it complicates the diff; the brainstorm only requires the interactive path.
- **Pam's tools at C6b.** Filesystem + shell is enough for the Plan-only flow. Pam may try `read_file` on a too-large file; without `read_file_smart` (C6c), this either truncates via existing safeguards or returns the full content. Acceptable for v0; called out so the implementer is not surprised.
- **History filter and resume.** `messages` table only stores stringified user/assistant turns, not the full `ModelMessage` tree, so `resume_run` already produces a cleaned history naturally. The filter exists for the *in-memory* `_message_history` that lives only within one CLI session. Keep both code paths consistent: they both want a noise-free history.
- **`/init` clobbering an existing AGENTS.md.** The init flow should refuse to overwrite an existing AGENTS.md unless the user confirms. The `write_file` approval gate (C5a) already gives the user a diff and a yes/no, so this is handled by the existing approval surface — but the system prompt should explicitly tell Scott to read any existing AGENTS.md first and merge rather than replace blindly.
- **Pam attempts with `task_id=NULL`.** Pam's run produces tasks but does not itself execute *against* a task. Her attempt row therefore keeps `task_id=NULL`, which the C6 migration already permits.
- **`@agent.instructions` migration path.** When deps land (probably C20 hot-reload or earlier), swap static composition for the decorator. The `MODE_PROMPTS` dict and slash-handler shape stay the same; only `config_loader`'s wiring changes.
