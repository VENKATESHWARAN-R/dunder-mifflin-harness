# PyGeminiCLI

A Python clone of Google's Gemini CLI, built as a learning exercise to deeply understand agentic loop mechanics, tool systems, and harness design.

## Quick Start

```bash
uv sync                    # Install dependencies
uv run pygemini            # Run the CLI
uv run pytest              # Run tests
uvx ruff check src/        # Lint
uvx ty check src/          # Type check
```

## Architecture

```
src/pygemini/
├── cli/        # Frontend — terminal UI, input, rendering. NEVER calls Gemini API directly.
├── core/       # Backend — agent loop, Gemini API, history, config. Talks to CLI via events.
├── tools/      # Tool system — BaseTool ABC, ToolRegistry, individual tool implementations.
│   └── filesystem/  # File tools: read, write, edit, list
├── context/    # GEMINI.md discovery, memory store, file discovery.
├── session/    # Session save/load, checkpoints, conversation compression.
├── safety/     # Approval modes, policy engine, sandboxing.
├── hooks/      # Lifecycle hook system (shell commands with JSON protocol).
└── mcp/        # MCP (Model Context Protocol) client integration.
```

## Key Design Patterns

- **Core never imports from CLI**. Communication is via `EventEmitter` (`core/events.py`).
- **All tools extend `BaseTool`** (`tools/base.py`) and register with `ToolRegistry` (`tools/registry.py`).
- **`AgentLoop`** (`core/agent_loop.py`) is the heart: ReAct cycle of prompt → stream → tool dispatch → loop.
- **Config** uses Pydantic with layered loading: defaults → `~/.pygemini/settings.toml` → project settings → env vars → CLI flags.
- **Async throughout**: `asyncio` native, streaming responses, async tool execution.

## Critical Files (Read These First)

1. `src/pygemini/core/agent_loop.py` — ReAct loop (the heart of the system)
2. `src/pygemini/core/content_generator.py` — Gemini API integration
3. `src/pygemini/core/events.py` — Core↔CLI event bridge
4. `src/pygemini/tools/base.py` — Tool contract (BaseTool ABC, ToolResult)
5. `src/pygemini/tools/registry.py` — Tool dispatch center
6. `src/pygemini/cli/app.py` — REPL orchestrator wiring everything together
7. `src/pygemini/core/config.py` — Configuration foundation

## Conventions

- Python 3.12+, managed with `uv`
- `ruff` for formatting and linting
- Type hints on all public functions
- Pydantic models for all config/schema types
- Tests in `tests/` mirroring `src/` structure
- Docstrings on public classes and methods
- No agentic framework (direct google-genai SDK calls — the whole point is learning the loop)

## Reference

- [Gemini CLI repo](https://github.com/google-gemini/gemini-cli) — The TypeScript original we're cloning
- [google-genai Python SDK](https://pypi.org/project/google-genai/) — API client we use

## Sub-Agent Model Assignments
- **opus**: Complex architectural work (edit_file, shell, agent_loop mods, policy engine, MCP)
- **sonnet**: Standard implementations (tools, tests, wiring)
- **haiku**: Boilerplate (exports, `__init__.py`, simple registrations)

## Environment Variables

| Variable                 | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `GEMINI_API_KEY`         | Gemini API key (required)                     |
| `PYGEMINI_MODEL`         | Override default model                        |
| `PYGEMINI_HOME`          | Override config dir (default: `~/.pygemini/`) |
| `PYGEMINI_SANDBOX`       | Sandbox mode: `none`, `docker`                |
| `PYGEMINI_APPROVAL_MODE` | `interactive`, `auto_edit`, `yolo`            |
| `PYGEMINI_DEBUG`         | Enable debug logging                          |
