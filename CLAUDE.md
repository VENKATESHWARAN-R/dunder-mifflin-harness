# CLAUDE.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

JAC ("Just Another CLI") is an R&D harness exploring whether a multi-agent system with tiered model routing can match Anthropic's long-running coding harness at 3–5× lower cost. The repo directory is `dunder-mifflin-harness` (a nod to the predecessor project), but the product is **JAC**.

This file is intentionally stable guidance. Do not treat it as the source of
truth for shipped component status, active tables, or milestone progress.
Use:

- `docs/ROADMAP.md` for shipped/planned component status
- `docs/contracts/STATE_SCHEMA.md` for table definitions + activation sequence
- `docs/dev/*.md` for "what is currently built" in each layer

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
- `src/jac/agents/` — agent factory: primary managed site for agent construction. `config_loader` reads DB config and builds agents.
- `src/jac/tools/` — shared local tool helpers (filesystem attachments, shell execution).
- `src/jac/config.py` — env-backed settings.
- `lab/` — research workspace; the `lab` dependency group covers extras only used here.
- `docs/dev/` — layer deep-dives and component history. **Read the relevant `docs/dev/<layer>.md` before touching an existing layer.** Index at `docs/dev/README.md`.

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
| Runtime or CLI code | `src/jac/runtime/` or `src/jac/cli/` | `docs/reference/PHILOSOPHY.md` |
| Local tool helper | `src/jac/tools/` | `docs/contracts/TOOLS_CONTRACT.md` |
| Test | `tests/` (mirror package layout) | — |
| Locked design doc | `docs/contracts/` + add to `docs/README.md` | Add status header dated today |
| Stable narrative doc | `docs/reference/` + add to `docs/README.md` | Add status header dated today |
| Implementation plan (handoff for coding) | `docs/implementation_docs/<slug>.md` | After analysis + human agreement; see **Planning vs implementation** below |
| Developer doc update (layer changed) | `docs/dev/<layer>.md` | After the component ships; see **Developer Docs** below |
| Rough idea, half-formed | `lab/brainstorm/YYYY-MM-DD-<slug>.md` | Promote later if it matures |
| Runnable experiment | `lab/scripts/` + add row to `lab/README.md` | — |
| Notebook | `lab/notebooks/` + add row to `lab/README.md` | — |
| Reference project | `lab/specimens/<name>/` (workspace member) | Don't import from `src/jac/` |
| Research-only dep | `[dependency-groups].lab` in `pyproject.toml` | Keeps core install lean |

When promoting a brainstorm to a contract, add a row to `docs/README.md` and either delete the brainstorm or leave a one-line pointer.

## Brainstorm Sessions

Use the `brainstrom` skill for free-form design sessions. When a discussion finalises a decision that changes a contract, schema, event surface, or scope — propagate it to the relevant `Locked` doc and `docs/ROADMAP.md`. Don't silently accept an idea that contradicts a `Locked` contract; either revise the contract or push back.

## Database Scope

Do not encode "currently active tables" in this file. Always check
`docs/contracts/STATE_SCHEMA.md` (Activation Sequence) before writing to a new
table or changing persistence behavior.

When proposing state changes:
- Schema changes (fields, relationships) → update `STATE_SCHEMA.md` first, write a new numbered migration alongside the code.
- New table going live → confirm it lines up with the activation sequence, or revise it with reasoning.
- Migrations are append-only and ordered by filename; never edit a shipped migration.

## Architecture Direction

The planned harness is three layers: **Nodes** (atomic LLM/deterministic units) → **Workflows** (directed graphs, `pydantic_graph`) → **Modes** (Autopilot vs HITL). Model tiers are Scout (cheap/fast), Worker (balanced), Architect (most capable). See [`docs/reference/IDEA.md`](docs/reference/IDEA.md) §5 for the full design and [`docs/dev/architecture.md`](docs/dev/architecture.md) for what's built. Read both before touching the orchestration layer.

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
- **Agent/tool wiring changed** (`config_loader`, approval wrapper, delegation, MCP/skills wiring):
  - `docs/contracts/TOOLS_CONTRACT.md`, `docs/contracts/MCP_INTEGRATION.md` (Locked)
  - `docs/dev/agents-layer.md`, `docs/dev/runtime-layer.md`
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

- Keep dependencies pointing inward: UI/server adapters depend on runtime; runtime depends on domain/tool abstractions; domain logic must not import adapters.
- Add backend behavior behind runtime events or workflow nodes before exposing it in the CLI.
- Prefer small, testable modules over large app objects.
- Reuse existing settings, event, approval, question, and tool helper patterns before creating new abstractions.
- Keep docs aligned when changing boundaries or adding a new top-level subsystem — update the relevant contract and `docs/README.md` index.
- Respect the doc status field. Don't change a `Locked` contract casually; if you do, bump `Last revised`.
- Don't import from `lab/specimens/`. Specimens are inspiration only.
