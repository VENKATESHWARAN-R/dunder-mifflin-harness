# CLAUDE.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

An R&D project building a multi-agent agentic harness capable of autonomously developing software from a single high-level prompt. The core hypothesis: tiered model routing (cheap models for simple tasks, expensive models for hard ones) can match Anthropic's harness quality at 3–5x lower cost.

This project is in the **early implementation phase**. The foundation is a Click/prompt_toolkit/Rich CLI over a CLI-agnostic runtime boundary with typed events. The current runtime still wraps a simple Pydantic AI agent; future work should replace that coordinator internals with graph workflows without changing the CLI/event boundary.

## Commands

This project uses `uv` for package management and `just` as the task runner.

```bash
just sync        # Install all dependencies (--all-groups --all-extras)
just run [args]  # Run any command via `uv run --group harness --env-file .env`
                 # Example: just run harness "say hello"
                 # Args with spaces may need nested quotes depending on shell usage
just test        # Run pytest
just lint        # Lint with ruff
just format      # Format with ruff
just fix         # Lint + format with auto-fix
just typecheck   # Type check with ty
just clean       # Remove build artifacts and __pycache__
```

Run a single test:
```bash
uv run pytest path/to/test_file.py::test_name
```

Useful CLI smoke checks:
```bash
uv run harness --help
uv run harness "say hello"
uv run harness chat
```

## Core Documents

- `brainstrom/00-IDEA.md` - original architecture and research direction.
- `docs/PROJECT_PHILOSOPHY.md` - where code belongs, dependency boundaries, and extension rules.
- `docs/CLI_DESIGN.md` - CLI/event/input/rendering design.
- `ROADMAP.md` - current implementation slices.

Read `docs/PROJECT_PHILOSOPHY.md` before adding new subsystems such as custom agents, MCP servers, A2A servers, sandboxing, workflow runners, or new UI surfaces.

## Source Layout

- `src/dunder_mifflin_harness/cli/` - terminal adapter only: Click commands, prompt_toolkit input, slash commands, Rich rendering, and human prompt views.
- `src/dunder_mifflin_harness/runtime/` - UI-agnostic runtime boundary: events, sessions, approvals, questions, and `RunCoordinator`.
- `src/dunder_mifflin_harness/tools/` - shared local tool helpers such as filesystem attachments and shell execution.
- `src/dunder_mifflin_harness/config.py` - environment-backed settings.
- `tests/` - focused coverage for entrypoints, parsers, renderer behavior, runtime contracts, config, and local helpers.
- `specimens/` - standalone reference projects. Treat these as inspiration only; do not import from them or conflate them with main code.
- `lab/` - experiments and notebooks.

## Architecture Direction

The planned harness has three compositional layers:

1. **Nodes** — atomic units: LLM calls (plan, build, evaluate) or deterministic logic (routing, state reads/writes, cost tracking)
2. **Workflows** — directed graphs wiring nodes for a specific dev strategy (feature-by-feature, TDD, POC-swarm, etc.)
3. **Modes** — top-level configs selecting which workflow to run (Autopilot vs HITL)

Every node should follow a uniform interface: receive state `(run_id, task_id, context, config)`, return updated state plus status. Workflows compose nodes. Modes choose workflow compositions.

**V0 workflow** (feature-by-feature, the only one shipping initially):
```
plan → task_router → context_loader → config_loader → build → evaluate → pass_check
                                                                              ├── Pass → state_writer → task_router (next task or DONE)
                                                                              └── Fail → hr_escalation → config_loader → build (retry)
```

**Model tiers** (provider-agnostic):
- Tier 1 (Scout): cheap/fast — file reading, boilerplate, formatting
- Tier 2 (Worker): balanced — feature impl, testing, evaluation
- Tier 3 (Architect): most capable — planning, architecture, complex debugging

**Persistent state store** tracks: run state, tasks (with status/complexity/tier), attempt records (model, tokens, cost, eval scores), agent configs, and scoped context per agent role.

**Tool abstraction layer**: agents call tools through a standardized interface so the underlying execution environment (local → container → cloud) can change without touching agent code.

## Current Design Decisions

- The CLI is a presentation adapter, not the agent orchestrator.
- Runtime communication uses typed events and explicit request/response handshakes.
- Approvals and user questions are separate primitives.
- Slash commands mutate local session/runtime config and are not sent to the model.
- Orchestration is graph-based with conditional branching (not a linear pipeline)
- Agent configs (model tier, prompts, tools) are stored in the persistent state store and loaded at instantiation time — enabling mid-run updates (v1+)
- Context reads are scoped per agent role: agents see only what's relevant to their task
- V0 ships one workflow only (feature-by-feature) but node interfaces are designed for reuse from day one
- HITL and Autopilot are **separate workflow compositions** sharing the same node library, not a single workflow with conditional checkpoints
- The CLI/runtime event contract is the stable integration point for terminal UI now and browser/A2A surfaces later

## Working Rules

- Keep dependencies pointing inward: UI/server adapters depend on runtime; runtime depends on domain/tool abstractions; domain logic must not import adapters.
- Add backend behavior behind runtime events or workflow nodes before exposing it in the CLI.
- Prefer small, testable modules over large app objects.
- Use existing settings, event, approval, question, and tool helper patterns before creating new abstractions.
- Keep docs aligned when changing boundaries or adding a new top-level subsystem.
