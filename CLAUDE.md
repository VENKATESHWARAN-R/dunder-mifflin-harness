# CLAUDE.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

JAC ("Just Another CLI") is an R&D harness exploring whether a multi-agent system with tiered model routing can match Anthropic's long-running coding harness at 3–5× lower cost. The repo directory is `dunder-mifflin-harness` (a nod to the predecessor project), but the product is **JAC**.

The project is in **early implementation**. C0 shipped a Click/prompt_toolkit/Rich CLI over a UI-agnostic runtime boundary with typed events. C1 added SQLite persistence (`runs` + `messages`) and `jac resume`. C2 added workspace file seeding (`skills`, `mcp_servers`) and `jac doctor` diagnostics. The current runtime wraps a simple Pydantic AI agent; later components replace coordinator internals with graph workflows without changing the CLI/event boundary. The roadmap is now component-wise (C0..Cn) — see `docs/ROADMAP.md`.

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
just clean        # remove build artifacts and caches
```

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
src/jac/        # the product — terminal adapter + runtime + tools
tests/          # pytest suite mirroring src/jac/
docs/           # authoritative docs (contracts/, reference/, implementation_docs/) + ROADMAP.md
lab/            # experiments (brainstorm/, scripts/, notebooks/, specimens/)
```

- `src/jac/cli/` — terminal adapter only: Click commands, prompt_toolkit input, slash commands, Rich rendering, prompt views.
- `src/jac/runtime/` — UI-agnostic runtime: events, sessions, approvals, questions, `RunCoordinator`.
- `src/jac/tools/` — shared local tool helpers (filesystem attachments, shell execution).
- `src/jac/config.py` — env-backed settings.
- `lab/` — research workspace; the `lab` dependency group covers extras only used here.

## Doc Status Convention

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

## Core Documents

Index: [`docs/README.md`](docs/README.md). Every entry lists status.

**Locked contracts** (binding):
- [`docs/contracts/STATE_SCHEMA.md`](docs/contracts/STATE_SCHEMA.md) — SQLite schema. Update before touching the database layer.
- [`docs/contracts/EVENT_CONTRACT.md`](docs/contracts/EVENT_CONTRACT.md) — typed events, requests, commands at the runtime↔UI boundary.
- [`docs/contracts/CLI_DESIGN.md`](docs/contracts/CLI_DESIGN.md) — terminal adapter design, input grammar, slash commands.
- [`docs/contracts/MCP_INTEGRATION.md`](docs/contracts/MCP_INTEGRATION.md) — MCP servers/skills loaded from DB into Pydantic AI agents (`config_loader` contract).
- [`docs/contracts/TOOLS_CONTRACT.md`](docs/contracts/TOOLS_CONTRACT.md) — agent tool interface. Required reading before adding a tool.

**Reference** (stable narrative):
- [`docs/reference/IDEA.md`](docs/reference/IDEA.md) — project genesis, hypothesis, V0 scope.
- [`docs/reference/PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md) — where code belongs, dependency boundaries, extension rules. **Read before adding new subsystems.**
- [`docs/reference/V0_BENCHMARK.md`](docs/reference/V0_BENCHMARK.md) — Notes CLI test case with acceptance criteria.

**Living**:
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — component plan in dependency order (C0..Cn).

**Lab**:
- [`lab/README.md`](lab/README.md) — index of all experiments, scripts, notebooks, brainstorms, and specimens.

## Where to Put a New X

| New thing | Path | Required reading |
|---|---|---|
| Runtime or CLI code | `src/jac/runtime/` or `src/jac/cli/` | `docs/reference/PHILOSOPHY.md` |
| Local tool helper | `src/jac/tools/` | `docs/contracts/TOOLS_CONTRACT.md` |
| Test | `tests/` (mirror package layout) | — |
| Locked design doc | `docs/contracts/` + add to `docs/README.md` | Add status header dated today |
| Stable narrative doc | `docs/reference/` + add to `docs/README.md` | Add status header dated today |
| Implementation plan (handoff for coding) | `docs/implementation_docs/<slug>.md` | After analysis + human agreement; see **Planning vs implementation** below |
| Rough idea, half-formed | `lab/brainstorm/YYYY-MM-DD-<slug>.md` | Promote later if it matures |
| Runnable experiment | `lab/scripts/` + add row to `lab/README.md` | — |
| Notebook | `lab/notebooks/` + add row to `lab/README.md` | — |
| Reference project | `lab/specimens/<name>/` (workspace member) | Don't import from `src/jac/` |
| Research-only dep | `[dependency-groups].lab` in `pyproject.toml` | Keeps core install lean |

When promoting a brainstorm to a contract, add a row to `docs/README.md` and either delete the brainstorm or leave a one-line pointer.

## Brainstorm Sessions

A brainstorm (the `brainstrom` skill, or any free-form discussion) can legitimately end in any of: a spoken decision and nothing written, a note in `lab/brainstorm/`, a runnable spike in `lab/scripts/` or `lab/notebooks/`, an edit to an existing `Locked` contract, a new `Draft` doc, a `docs/ROADMAP.md` update, or a real implementation task. Most sessions need none of these — that's fine.

When a discussion *does* finalize a decision, propagate it. The common cascade:

- Decision changes how state is represented → revise `docs/contracts/STATE_SCHEMA.md` and bump `Last revised`.
- Decision changes the runtime↔UI surface → revise `docs/contracts/EVENT_CONTRACT.md`.
- Decision changes how tools are written → revise `docs/contracts/TOOLS_CONTRACT.md`.
- Decision changes scope, order, or adds work → update `docs/ROADMAP.md`.
- New component or surface → new `Draft` doc under `docs/contracts/` or `docs/reference/`, plus a row in `docs/README.md`.

Don't silently accept an idea that contradicts a `Locked` contract. Either revise the contract or push back.

## Database Scope (post-C2)

C1 + C2 have shipped. SQLite is wired through `src/jac/state/` and `runs` + `messages` are populated by `RunCoordinator` on every invocation. On boot, `src/jac/state/seeder.py` walks `~/.jac/skills/`, `~/.jac/mcp/`, `.agents/skills/`, and `.agents/mcp/` and upserts into `skills` and `mcp_servers`. State lives at `Workspace.state_db_path` (`<repo>/.agents/state.db` for projects, `~/.jac/runs/<cwd-hash>/state.db` otherwise). Schema version `1.0` is recorded in `schema_meta`; future migrations land as numbered files in `src/jac/state/migrations/`.

The other tables in `docs/contracts/STATE_SCHEMA.md` are **created** by the v1.0 migration but **not yet populated** — they activate as their owning components arrive (see the schema's "Activation Sequence" table). Until then, code that needs `tasks`, `attempts`, `agent_configs`, etc. can prepare against the locked schema, but should not start writing to those tables outside the component that owns them.

When proposing state-related changes:
- If the change is about how state is *represented* (fields, relationships, lifecycle), update `STATE_SCHEMA.md` first and write a new numbered migration alongside the code change.
- If the change is about *new persistence* (a previously-empty table goes live), make sure it lines up with the activation sequence — or revise the activation table with reasoning.
- Migrations are append-only and ordered by filename; never edit a shipped migration.

## Architecture Direction

The planned harness has three compositional layers:

1. **Nodes** — atomic units: LLM calls (plan, build, evaluate) or deterministic logic (routing, state reads/writes, cost tracking).
2. **Workflows** — directed graphs wiring nodes for a specific dev strategy (feature-by-feature, TDD, POC-swarm, etc.).
3. **Modes** — top-level configs selecting which workflow to run (Autopilot vs HITL).

Every node follows a uniform interface: receive state `(run_id, task_id, context, config)`, return updated state plus status. Workflows compose nodes. Modes choose workflow compositions.

**First workflow** (feature-by-feature, ships through C11; alternatives at C22):
```
plan → task_router → context_loader → config_loader → build → evaluate → pass_check
                                                                              ├── Pass → state_writer → task_router (next task or DONE)
                                                                              └── Fail → hr_escalation → config_loader → build (retry)
```

**Model tiers** (provider-agnostic):
- Tier 1 (Scout): cheap/fast — file reading, boilerplate, formatting.
- Tier 2 (Worker): balanced — feature impl, testing, evaluation.
- Tier 3 (Architect): most capable — planning, architecture, complex debugging.

**Persistent state store** tracks: run state, tasks (with status/complexity/tier), attempt records (model, tokens, cost, eval scores), agent configs, and scoped context per agent role.

**Tool abstraction layer**: agents call tools through a standardized interface so the underlying execution environment (local → container → cloud) can change without touching agent code.

## Current Design Decisions

- The CLI is a presentation adapter, not the agent orchestrator.
- Runtime communication uses typed events and explicit request/response handshakes.
- Approvals and user questions are separate primitives.
- Slash commands mutate local session/runtime config and are not sent to the model.
- Orchestration is graph-based with conditional branching (not a linear pipeline).
- **Orchestration library: Pydantic AI throughout.** Agents for roles, `pydantic_graph` for the workflow graph (introduced at C10), `pydantic_evals` for evaluation. No LangGraph, no ADK in core (LangGraph lives only in `lab/` for research).
- **State store: SQLite.** Single file, local-first, crash-safe. Schema in `docs/contracts/STATE_SCHEMA.md`.
- Agent configs (model tier, prompts, tools) are stored in the state store and loaded at instantiation time — enabling mid-run updates (hot-reload at C20).
- Context reads are scoped per agent role: agents see only what's relevant to their task.
- Only one workflow ships before C22 (feature-by-feature) but node interfaces are designed for reuse from day one.
- HITL and Autopilot are **separate workflow compositions** sharing the same node library.
- The CLI/runtime event contract is the stable integration point for terminal UI now and browser/A2A surfaces later.
- **Not everything is an agent.** Simple one-off tasks use direct LLM calls (`pydantic_ai.direct`). These are still recorded in the `attempts` table with `call_type = 'direct_llm'` for cost tracking.
- **Tools are MCP-first.** Agent tool access is configured via `allowed_tools` in `agent_configs` as a JSON array of tool names and MCP server IDs. Local tools only until remote MCP transports come online at C17.
- **Agent teams (C15).** Multiple agent instances can run in parallel within a run and coordinate via the `agent_messages` queue. Schema is defined now; wiring lands at C15.

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

- Keep dependencies pointing inward: UI/server adapters depend on runtime; runtime depends on domain/tool abstractions; domain logic must not import adapters.
- Add backend behavior behind runtime events or workflow nodes before exposing it in the CLI.
- Prefer small, testable modules over large app objects.
- Reuse existing settings, event, approval, question, and tool helper patterns before creating new abstractions.
- Keep docs aligned when changing boundaries or adding a new top-level subsystem — update the relevant contract and `docs/README.md` index.
- Respect the doc status field. Don't change a `Locked` contract casually; if you do, bump `Last revised`.
- Don't import from `lab/specimens/`. Specimens are inspiration only.
