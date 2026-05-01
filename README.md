# JAC — Just Another CLI

A research harness exploring whether a multi-agent system with smart model routing can match Anthropic's long-running agentic coding harness at 3–5× lower cost.

> Repo name (`dunder-mifflin-harness`) is a nod to the predecessor project [dunder-mifflin-play](https://github.com/VENKATESHWARAN-R/dunder-mifflin-play). The product itself is **JAC**.

## Status

Early implementation. W0 (CLI + runtime foundation) shipped. W1+ slices are in flight — see [`docs/ROADMAP.md`](docs/ROADMAP.md). Not a product. A deliberate experiment.

## Quickstart

```bash
just sync                       # install core + dev + lab groups
just run jac --help             # CLI help
just run jac "say hello"        # one-shot prompt
just run jac chat               # interactive REPL
just test                       # run pytest
```

Set up `.env` from `.env.template` first — `JAC_MODEL` and one of `PYDANTIC_AI_GATEWAY_API_KEY` / `GEMINI_API_KEY` are the minimum needed for live calls.

## Where things live

```
.
├── src/jac/                # the product
│   ├── cli/                # terminal adapter (Click, prompt_toolkit, Rich)
│   ├── runtime/            # UI-agnostic runtime: events, sessions, coordinator
│   ├── tools/              # local tool helpers (filesystem, shell)
│   └── config.py           # env-backed settings
│
├── tests/                  # pytest suite
│
├── docs/                   # authoritative docs — see docs/README.md
│   ├── contracts/          # Locked specs (state schema, events, tools, MCP, CLI design)
│   ├── reference/          # Stable narrative (idea, philosophy, V0 benchmark)
│   └── ROADMAP.md          # Living weekly slice plan
│
└── lab/                    # experiments — see lab/README.md
    ├── brainstorm/         # rough ideas, free-form notes
    ├── scripts/            # runnable Python spikes
    ├── notebooks/          # Jupyter explorations
    └── specimens/          # reference projects (pygemini) — inspiration only
```

## Reading order for new contributors

1. This README.
2. [`docs/README.md`](docs/README.md) — what's locked, what's reference, status legend.
3. [`docs/reference/IDEA.md`](docs/reference/IDEA.md) — why this project exists.
4. [`docs/reference/PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md) — where code belongs.
5. [`docs/contracts/`](docs/contracts/) — the actual specs.
6. [`docs/ROADMAP.md`](docs/ROADMAP.md) — what we're working on this week.

## Where to put a new X

| New thing | Goes in | Notes |
|---|---|---|
| Runtime or CLI code | `src/jac/runtime/` or `src/jac/cli/` | Follow [`docs/reference/PHILOSOPHY.md`](docs/reference/PHILOSOPHY.md). |
| Local tool helper | `src/jac/tools/` | Read [`docs/contracts/TOOLS_CONTRACT.md`](docs/contracts/TOOLS_CONTRACT.md). |
| New test | `tests/` | Mirror the package structure. |
| Locked design doc | `docs/contracts/` + index entry in `docs/README.md` | Add status header. |
| Stable narrative doc | `docs/reference/` + index entry | Add status header. |
| Rough idea / design spike note | `lab/brainstorm/<date>-<slug>.md` | Promote later if it matures. |
| Runnable experiment | `lab/scripts/` + index entry in `lab/README.md` | |
| Notebook exploration | `lab/notebooks/` + index entry | |
| Reference project | `lab/specimens/<project>/` | Workspace member; don't import from main code. |
| Research-only dep | `[dependency-groups].lab` in `pyproject.toml` | Keeps the core install lean. |

## Toolchain

`uv` (package manager) · `just` (task runner) · `ruff` (lint/format) · `ty` (type check) · `pytest`. Python ≥ 3.13.

```bash
just lint
just format
just typecheck
just fix         # lint + format with auto-fix
just clean       # nuke caches and build artifacts
```
