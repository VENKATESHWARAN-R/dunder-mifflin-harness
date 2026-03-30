# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An R&D project building a multi-agent agentic harness capable of autonomously developing software from a single high-level prompt. The core hypothesis: tiered model routing (cheap models for simple tasks, expensive models for hard ones) can match Anthropic's harness quality at 3–5x lower cost.

This project is in the **design/ideation phase**. No production code exists yet — `main.py` is a stub. The primary design document is `brainstrom/00-IDEA.md`.

## Commands

This project uses `uv` for package management and `just` as the task runner.

```bash
just sync        # Install all dependencies (--all-groups --all-extras)
just run         # Run main.py
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

## Architecture (Planned)

The harness is built on three compositional layers:

1. **Nodes** — atomic units: LLM calls (plan, build, evaluate) or deterministic logic (routing, state reads/writes, cost tracking)
2. **Workflows** — directed graphs wiring nodes for a specific dev strategy (feature-by-feature, TDD, POC-swarm, etc.)
3. **Modes** — top-level configs selecting which workflow to run (Autopilot vs HITL)

Every node follows a uniform interface: receives a state object `(run_id, task_id, context, config)`, returns updated state + status. This makes nodes composable across workflows.

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

## Workspace Structure

- `brainstrom/00-IDEA.md` — full design document; primary reference for architecture decisions
- `specimens/` — standalone reference projects; treat as separate workspaces. Currently contains `pygemini` (a Gemini-based CLI agent built with Google GenAI SDK). These are inspiration/technical reference only — do not conflate with the main project.
- `lab/notebooks/` — experimental notebooks
- `pyproject.toml` — workspace root; `specimens/pygemini` is a uv workspace member with its own dependency group

## Key Design Decisions

- Orchestration is graph-based with conditional branching (not a linear pipeline)
- Agent configs (model tier, prompts, tools) are stored in the persistent state store and loaded at instantiation time — enabling mid-run updates (v1+)
- Context reads are scoped per agent role: agents see only what's relevant to their task
- V0 ships one workflow only (feature-by-feature) but node interfaces are designed for reuse from day one
- HITL and Autopilot are **separate workflow compositions** sharing the same node library, not a single workflow with conditional checkpoints
