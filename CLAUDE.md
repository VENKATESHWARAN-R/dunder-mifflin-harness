# CLAUDE.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

JAC ("Just Another CLI") is a multi-agent CLI coding harness with two coupled goals: (1) measure whether four roles with tiered model routing can build small applications autonomously at materially lower cost than a single-Opus baseline, on the **Notes CLI** benchmark + a **single-function bug-fix** secondary; and (2) be a usable terminal coding assistant — chat, plan, build, evaluate, resume — backed by SQLite persistence and an event/runtime contract that's surface-independent. Repo directory is `dunder-mifflin-harness` (predecessor project nod); the product is **JAC**.

> **2026-05-08 reset.** The original 31-component plan is replaced by 5 milestones (M1–M5) for Phase 1 + an evidence-gated Phase-2 catalog. M1 ships Scott alone; later personas land driven by M1+ run-log evidence. Source-of-truth brainstorm: [`lab/brainstorm/2026-05-08-jac-reset-from-scratch.md`](lab/brainstorm/2026-05-08-jac-reset-from-scratch.md). C0–C8 historical and being cut/rebuilt in M1.

This file is stable guidance, not status truth. Use:

- [`docs/reference/PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md) — the five principles
- [`docs/contracts/SUBSTRATE.md`](docs/contracts/SUBSTRATE.md) — YAML/TOML/JSON/Markdown/SQLite boundary
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Phase-1 milestones (M1–M5) + Phase-2 catalog
- [`docs/contracts/STATE_SCHEMA.md`](docs/contracts/STATE_SCHEMA.md) — current schema (1.5)
- `docs/dev/*.md` — what is currently built per layer (these go stale during M1 rebuild; refreshed after M1 lands)

## Core Invariants

Five principles + operational rules. Each rules something *out*. Full text + teeth in [`docs/reference/PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md).

**Principles:**

1. **The runtime is the contract; the UI is one surface.** UI may import runtime; runtime never imports UI. Anything that should outlive the terminal (browser, A2A, headless) lives in `runtime/`.
2. **Keep the spine thin.** Spine: prompt → plan → route → build → evaluate → ledger → report. New structure justified only by a logged run that breaks the spine.
3. **Measure before you abstract.** No new persona, tool, contract, or dependency without a `attempts` row or run trace that motivates it.
4. **State is durable; agents are disposable.** SQLite holds runs/messages/attempts/tasks/configs. Agent objects, sessions, in-memory caches must be reconstructible from state.
5. **Config-as-data, not config-as-code.** YAML / TOML / JSON / Markdown / SQLite per [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md). Python files contain code, never tunable values. No `if role == "manager":` branches.

**Operational invariants (don't violate these even if a principle seems to permit it):**

- **Dependency direction is inward.** `cli` or `surfaces/*` → `runtime` → `agents` → `tools / state / config`. The full layer matrix is in [`PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md#dependency-direction).
- **`src/jac/agents/base.py` is the only site that calls `pydantic_ai.Agent.from_file()` / `Agent.from_spec()`.** Tier resolution, MCP toolset wiring, approval middleware, and persona-YAML loading happen there. If you find yourself constructing an Agent elsewhere, route through the factory.
- **Two communication primitives, not one.** `EventBus` emits typed dataclass events (one-way, fire-and-forget). Approvals and questions are request/response with `asyncio.Future` waiters — they block the coordinator until the UI answers. Don't conflate them.
- **Slash commands are local control, not model input.** They mutate session/runtime config and never get sent to the LLM.
- **Three-tier model resolution.** For every agent build, model is resolved by walking: (1) per-run `agent_configs` override in SQLite → (2) user `settings.json` → (3) shipped `data/model_specs.toml`. First non-null wins.

## Phase-1 reminder

M1 ships **Scott alone**. There are no `nodes/` or `workflows/` packages yet — they enter when M2/M3 evidence requires them. Phase-2 components (compaction, hooks, skills, HITL, alt strategies, sandboxing, browser UI, multi-repo A2A, agent teams) are catalogued in [`docs/ROADMAP.md`](docs/ROADMAP.md) with explicit data triggers; none ship until M5's data motivates them. If you're adding code that looks like Phase-2 capability, stop and re-read principle B3 (measure before you abstract).

## Commands

`uv` for package management, `just` for tasks.

```bash
just sync         # install core + dev + lab groups
just run jac ...  # run the CLI (e.g. `just run jac "say hello"`, `just run jac chat`)
just test         # pytest
just lint         # ruff check
just format       # ruff format
just fix          # ruff check --fix && ruff format
just typecheck    # ty
just qa           # lint + typecheck + full test suite
just precommit-install  # install git pre-commit + pre-push hooks
just precommit-run      # run hooks across all files
just clean        # remove build artifacts and caches
```

Run `just qa` after any significant `src/jac/` change before handing off. For changes that alter user-visible behavior or architecture boundaries, also update the relevant docs as described in this file.

Run a single test:
```bash
uv run pytest tests/test_cli.py::test_name
```

CLI smoke checks:
```bash
uv run jac --help
uv run jac "say hello"
uv run jac chat
```

## Repo Layout

```
src/jac/        # the product — terminal adapter + runtime + tools + state
src/jac/data/   # shipped declarative data: personas/*.yaml, model_specs.toml
tests/          # pytest suite mirroring src/jac/
docs/           # contracts/, reference/, ROADMAP.md, dev/, guide/, implementation_docs/ (historical)
lab/            # experiments (brainstorm/, scripts/, notebooks/, specimens/)
```

**M1 layout (target):**

- `src/jac/cli/` — terminal adapter only: Click commands, prompt_toolkit input, slash commands, Rich rendering, prompt views.
- `src/jac/surfaces/a2a/` — A2A peer adapter (Pydantic AI's `agent.to_a2a()`).
- `src/jac/runtime/` — UI-agnostic runtime: events, sessions, approvals, questions, `RunCoordinator`, model factory.
- `src/jac/agents/` — agent factory at `agents/base.py`, the sole site of `Agent.from_file()` / `Agent.from_spec()` calls. Persona YAML loader + tool/MCP resolution + approval middleware composition.
- `src/jac/tools/` — local tool helpers (filesystem, shell, task-list CRUD). Each tool carries `ToolApprovalMeta`.
- `src/jac/state/` — SQLite repos and migrations (M1 starting at `001_m1_initial.sql`).
- `src/jac/data/personas/` — shipped persona YAMLs (M1: `scott.yaml`).
- `src/jac/data/model_specs.toml` — shipped vendor data (model context windows, tier defaults).
- `src/jac/config.py` — env-backed settings (`pydantic-settings`).
- `src/jac/workspace.py` — workspace discovery (`~/.jac/`, `<repo>/.agents/`).

**M2/M3 additions (not yet present):** `src/jac/nodes/` (BaseNode subclasses), `src/jac/workflows/` (small Python files wiring nodes via `pydantic_graph`).

- `lab/` — research workspace; the `lab` dependency group covers extras only used here.
- `docs/dev/` — layer deep-dives. **Stale during M1 rebuild — they describe pre-reset code.** Refreshed after M1 lands.

## Doc System

Every `docs/*` file has a status header:

```markdown
> **Status:** Locked · **Last revised:** YYYY-MM-DD · **Type:** ...
```

| Status | Meaning |
|---|---|
| `Locked` | Contract — code must follow it; change requires intent. Lives in `docs/contracts/`. |
| `Reference` | Stable narrative — read for context, don't treat as binding. Lives in `docs/reference/`. |
| `Living` | Actively updated (e.g. `ROADMAP.md`). |
| `Draft` | In flight; do not rely on it yet. |

**Always check the status before treating a doc as authoritative.** If a `Draft` and a `Locked` doc disagree, the `Locked` one wins.

Full doc index: [`docs/README.md`](docs/README.md). Locked contracts in `docs/contracts/`, stable narratives in `docs/reference/`, layer deep-dives in `docs/dev/`.

**Product changes → docs:** When you add, update, or remove meaningful behavior or surface area (CLI, runtime, agents, tools, config, schema), keep documentation in step: the relevant **contracts**, **reference** pages, **`docs/dev/<layer>.md`** when a layer changes, **`docs/README.md`** when the index needs a new row, and **`docs/ROADMAP.md`** when shipped status, scope, or the living component plan changes. See **Developer Docs**, **Where to Put a New X**, and **Working Rules** below. Trivial internal-only fixes with no user-visible or architectural impact can skip doc churn.

## Where to Put a New X

| New thing | Path | Required reading |
|---|---|---|
| Runtime or CLI code | `src/jac/runtime/` or `src/jac/cli/` | [`PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md) |
| A2A or future browser surface adapter | `src/jac/surfaces/<name>/` | [`PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md) — must consume runtime events, never duplicate logic |
| New persona | `src/jac/data/personas/<role>.yaml` (shipped) or `~/.jac/personas/` (user) | [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md) — never as Python literal |
| Local tool helper | `src/jac/tools/` | [`TOOLS_CONTRACT.md`](docs/contracts/TOOLS_CONTRACT.md) |
| Workflow node (M2+) | `src/jac/nodes/` (package doesn't exist yet) | Wait until M2 evidence requires it |
| Workflow wiring (M3+) | `src/jac/workflows/<name>.py` (package doesn't exist yet) | Wait until M3 evidence requires it |
| Vendor-data file | `src/jac/data/<name>.toml` | [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md) |
| Skill / MCP server stub | Markdown / JSON in `~/.jac/` or `<repo>/.agents/` | [`WORKSPACE.md`](docs/contracts/WORKSPACE.md), [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md) |
| Test | `tests/` (mirror package layout) | — |
| Locked design doc | `docs/contracts/` + add to `docs/README.md` | Add status header dated today |
| Stable narrative doc | `docs/reference/` + add to `docs/README.md` | Add status header dated today |
| Implementation plan (handoff for coding) | `docs/implementation_docs/<slug>.md` | After analysis + human agreement; see **Planning vs implementation** below |
| Developer doc update (layer changed) | `docs/dev/<layer>.md` | After M1 lands or a layer changes meaningfully post-M1 |
| Rough idea, half-formed | `lab/brainstorm/YYYY-MM-DD-<slug>.md` | Promote later if it matures |
| Runnable experiment | `lab/scripts/` + add row to `lab/README.md` | — |
| Notebook | `lab/notebooks/` + add row to `lab/README.md` | — |
| Reference project | `lab/specimens/<name>/` (workspace member) | Don't import from `src/jac/` |
| Research-only dep | `[dependency-groups].lab` in `pyproject.toml` | Keeps core install lean |

When promoting a brainstorm to a contract, add a row to `docs/README.md` and either delete the brainstorm or leave a one-line pointer.

## Brainstorm Sessions

Use the `brainstrom` skill for free-form design sessions. When a discussion finalises a decision that changes a contract, schema, event surface, or scope — propagate it to the relevant `Locked` doc and `docs/ROADMAP.md`. Don't silently accept an idea that contradicts a `Locked` contract; either revise the contract or push back.

## Database Scope

Always check [`STATE_SCHEMA.md`](docs/contracts/STATE_SCHEMA.md) (Activation Sequence) before writing to a new table or changing persistence behavior. Schema is at v1.5 after the 2026-05-08 reset.

**M1 active tables:** `runs`, `messages`, `attempts`, `tasks`, `agent_configs` (single-row), `mcp_servers`, `skills`, `run_mcp_servers`, `run_skills`. Reserved-but-unused tables (`agent_instances`, `agent_teams`, `agent_messages`, `context_store`) were **dropped** in the reset and only return via Phase-2 evidence triggers.

When proposing state changes:
- Schema changes (fields, relationships) → update `STATE_SCHEMA.md` first, then write a new numbered migration.
- Re-introducing a removed table → confirm the Phase-2 trigger fired (record in the relevant brainstorm note); add the table back to `STATE_SCHEMA.md`; write a migration.
- Migrations are append-only and ordered by filename; never edit a shipped migration.
- Per [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md): SQLite holds per-run state. Persona prompts → YAML; user settings → JSON; vendor specs → TOML; skills/instructions → Markdown. The DB is not the canonical source for any value with a config-file home.

## Architecture Direction

The target architecture is four-role multi-agent (manager + planner + builder + evaluator) with tiered model routing. Phase 1 builds the cast incrementally: M1 ships Scott (manager) alone with full tool surface, A2A peer surface, and long-running task orchestration; M2 onward adds personas based on M1 logs. See [`docs/reference/IDEA.md`](docs/reference/IDEA.md) "Locked product definition (2026-05-08)" for the full framing and [`docs/ROADMAP.md`](docs/ROADMAP.md) for milestone-level scope.

When `nodes/` and `workflows/` packages enter (M2/M3): nodes are `BaseNode` subclasses with state-in/state-out `run()` methods; workflows are small Python files (~50–80 LOC) that wire nodes via `pydantic_graph`. **Workflows are not YAML** — `pydantic_graph` has no YAML loader; conditional edges live in code.

## Current Design Decisions

- **Dual product goal.** JAC is a research harness (cost-quality measurement on Notes CLI + bug-fix benchmarks) AND a usable terminal coding assistant.
- **The CLI is one surface, not the orchestrator.** Anything that should outlive the terminal lives in `runtime/`. A2A peer surface lands in M1 alongside the CLI.
- **Runtime communication uses typed events and explicit request/response handshakes.** Approvals and questions are separate primitives — never use approval prompts for clarification.
- **Slash commands mutate local session/runtime config; never sent to the model.**
- **Orchestration library: Pydantic AI throughout.** `Agent.from_file()` / `Agent.from_spec()` for persona loading; `pydantic_graph` for workflow graphs (M3+); `pydantic_evals` for evaluation when it matters. No LangGraph, no ADK in core.
- **State store: SQLite** at `<workspace>/state.db`. Schema in [`STATE_SCHEMA.md`](docs/contracts/STATE_SCHEMA.md).
- **Personas live as YAML** under `src/jac/data/personas/<role>.yaml` (shipped) ← `~/.jac/personas/` (user) ← `<repo>/.agents/personas/` (project). Code never holds persona prompts as Python strings (substrate rule, principle 5).
- **Three-tier model resolution.** Per-run `agent_configs` override → user `settings.json` → shipped `data/model_specs.toml`. First non-null wins. In-flight model swap is built into the factory from M1 day one.
- **Context reads are scoped per agent role** (when multiple roles exist post-M1).
- **The `tasks` table is JAC's context-resilient memory** — Scott maintains it via task-CRUD tools; the list is injected as a system reminder every turn so it survives compaction and resume. This is what makes JAC a long-running harness.
- **Not everything is an agent.** Simple one-off calls use `pydantic_ai.direct` and are recorded in `attempts` with `call_type='direct_llm'`.
- **Tools are MCP-aware.** `allowed_tools` in `agent_configs` is a JSON array of tool names and MCP server IDs (`mcp:` prefix). Local tools only in M1; remote MCP transports are a Phase-2 evidence-trigger item.
- **Multi-agent infra is post-evidence.** `spawn_minion`, agent teams, `agent_messages`, `context_store` only return via Phase-2 evidence triggers documented in [`ROADMAP.md`](docs/ROADMAP.md).

## Change-impact doc checklist

When code changes, update docs by impact area (not just by file touched):

- **CLI/user-visible behavior changed** (flags, commands, output, defaults, safety prompts):
  - `docs/contracts/CLI_DESIGN.md` (Locked)
  - `docs/guide/usage.md`, `docs/guide/getting-started.md`, `docs/guide/configuration.md` (user guides)
  - `README.md` (quickstart/command snippets)
  - relevant `docs/dev/cli-layer.md` / `docs/dev/runtime-layer.md`
- **Runtime events/request-response semantics changed**:
  - `docs/contracts/EVENT_CONTRACT.md` (Locked)
  - relevant `docs/dev/runtime-layer.md`, `docs/dev/cli-layer.md`
- **Agent/tool wiring changed** (factory at `agents/base.py`, approval wrapper, delegation, MCP/skills wiring):
  - `docs/contracts/TOOLS_CONTRACT.md`, `docs/contracts/MCP_INTEGRATION.md` (Locked)
  - `docs/dev/agents-layer.md`, `docs/dev/runtime-layer.md`
- **Persona shape, prompt, or new persona added** (YAML files):
  - `src/jac/data/personas/<role>.yaml` (shipped) or `~/.jac/personas/` (user override)
  - `docs/contracts/SUBSTRATE.md` (if a substrate boundary moves)
  - `docs/contracts/STATE_SCHEMA.md` (if `agent_configs.role` adds a new value worth documenting)
- **Substrate boundary moved** (e.g., a value migrated Python → YAML/TOML/JSON/Markdown):
  - `docs/contracts/SUBSTRATE.md` "Current violations" section gets a record
- **State/schema/persistence changed** (tables, columns, activation, status semantics):
  - `docs/contracts/STATE_SCHEMA.md` (Locked, first)
  - new migration in `src/jac/state/migrations/`
  - `docs/dev/state-layer.md`
- **Workspace/init/doctor/seeding changed**:
  - `docs/contracts/WORKSPACE.md` (Locked)
  - `docs/guide/getting-started.md`, `docs/guide/configuration.md`
  - `docs/dev/cli-layer.md`, `docs/dev/state-layer.md`
- **Roadmap scope/status changed**:
  - `docs/ROADMAP.md` (Living)
  - if docs inventory changed, update `docs/README.md` and `docs/dev/README.md`

## Where to verify before/after changes

- **Current shipped scope & ordering:** `docs/ROADMAP.md`
- **Binding behavior contracts:** `docs/contracts/*.md` (check status + last revised)
- **What is actually built:** `docs/dev/*.md` (then verify against `src/jac/**`)
- **User-facing truth:** `README.md` + `docs/guide/*.md`
- **Tests as executable spec:** `tests/` (especially CLI/runtime/agents/state files for touched area)

## Developer Docs

`docs/dev/` is the source of truth for what's built, how the pieces connect, and where to look when something breaks. Update the relevant `docs/dev/<layer>.md` after any component ships or a layer's structure changes meaningfully. Index at `docs/dev/README.md`; delegate large update passes to a sub-agent (the source code is the ground truth).

## Versioning (alpha)

JAC is still in alpha; breaking changes are expected. Even so, **bump the package version whenever you ship a meaningful change in `src/jac/`** so installed CLIs (`uv tool install`, `jac --version`) and support/debugging stay honest.

- **Canonical version:** `[project].version` in `pyproject.toml`. That is what `jac --version` and `importlib.metadata.version("jac")` report after install.
- **In-tree fallback:** `src/jac/__init__.py` defines `__version__` from installed metadata, with a fallback string only for editable/uninstalled edge cases. If you bump `pyproject.toml`, update that fallback to the same value so dev checkouts without a proper install stay consistent.
- **When to bump:** user-visible behavior (new CLI commands, config/env semantics, default model or deps), fixes that matter for released installs, or **breaking** changes (removed flags, changed defaults, contract-level behavior). For tiny internal-only refactors with no observable effect, a bump is optional.
- **How to bump:** use [SemVer](https://semver.org/) shape while on `0.x.y` — e.g. `0.1.0` → `0.1.1` for fixes/small additions, `0.2.0` when you intentionally break compatibility or ship a larger surface change. Document breaking changes in the commit/PR body either way.
- **Dependencies:** if `pyproject.toml` dependencies change, run `uv lock` and commit `uv.lock` so global installs resolve the same graph.

## Planning vs implementation (roadmap items, refactors, new work)

Non-trivial work — roadmap components, refactors, new subsystems, or anything that changes boundaries — should follow a **two-phase** pattern so analysis can run on a capable model and implementation can run in a separate session on a lighter model using a written handoff.

1. **Ground analysis first.** Before writing production code, establish what must change, how it fits `docs/reference/PHILOSOPHY.md` and existing contracts, which modules and events are touched, migration or CLI implications, and risks or open questions. Read the relevant **Locked** docs; do not contradict them without revising them.

2. **Align with the human.** When scope, tradeoffs, or contract updates are unclear, discuss and **finalize** decisions with the user before coding. Skip lengthy debate for obviously small, localized fixes.

3. **Write an implementation doc.** Produce a single markdown file under **`docs/implementation_docs/`** (e.g. `docs/implementation_docs/C7-foo.md` or a descriptive slug). It should be **detailed enough** that another session can implement without re-doing architecture discovery: goals, non-goals, key files and integration points, ordered steps, acceptance checks, and pointers to contracts. It should **not** be overwhelming — no essay-length background; link to existing docs instead of duplicating them.

4. **Implement from the doc.** A follow-up session (or a sub agent with smaller/faster model such as sonnet) should treat that file as the source of truth, implement against it, and update or remove the doc when the work lands if that keeps the tree honest.

Trivial one-line fixes and test-only tweaks do not need this ceremony.

## Working Rules

**Hygiene (every change):**
- **Testable.** Every new behaviour gets a test. Bug fixes get a test that fails before the fix. Use `TestModel` / `FunctionModel` from Pydantic AI for deterministic agent tests; never call live providers in unit tests.
- **Documented.** When a layer's structure changes meaningfully, update the relevant `docs/dev/<layer>.md`. When a contract changes, bump `Last revised`. When a top-level concept enters or leaves, update this file.

**Code placement:**
- Dependencies flow inward (see **Core Invariants** above and [`PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md#dependency-direction) for the layer matrix).
- Add backend behavior behind runtime events or future workflow nodes before exposing it in the CLI.
- Prefer small, testable modules over large app objects.
- Reuse existing settings, event, approval, question, and tool helper patterns before creating new abstractions.
- Avoid hardcoding values that belong in settings, the state DB, or a shipped data file (see [`SUBSTRATE.md`](docs/contracts/SUBSTRATE.md) and **Core Invariants**).

**Doc discipline:**
- Keep docs aligned when changing boundaries or adding a new top-level subsystem — update the relevant contract and `docs/README.md` index.
- Respect the doc status field. Don't change a `Locked` contract casually; if you do, bump `Last revised`.
- Don't import from `lab/specimens/`. Specimens are inspiration only.

**Progress tracker (every implementation slice):**
After any non-trivial implementation slice within a milestone, update the **Progress tracker** subsection of the relevant milestone in [`docs/ROADMAP.md`](docs/ROADMAP.md). The tracker is the durable handoff between sessions — without it, a fresh session has to re-derive state from git log and code reading. Each entry records:

- **Shipped slice** — date (absolute, `YYYY-MM-DD`), one-line summary of what landed, key files added/changed.
- **Stubs / deviations** — anything implemented differently than the milestone spec implies: returning empty values, hardcoded scope, a deferred branch, etc. Always say *why* (usually "downstream slice will replace this") so a future session knows when the stub becomes load-bearing vs. when it's still fine.
- **Remaining slices** — what's left in this milestone, in order.
- **Open questions still deferred** — items the brainstorm flagged that this slice didn't resolve.

Keep tracker entries terse — link to commits, implementation docs, or brainstorm notes for the long form. A tracker is a status snapshot, not a history book.

Trivial one-line fixes and test-only tweaks don't earn a tracker entry. A new file under `src/jac/`, a new schema row, a new contract revision, or anything that closes/opens a roadmap line item does.
