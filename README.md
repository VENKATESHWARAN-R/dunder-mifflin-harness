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

For an installed CLI, run:

```bash
uv tool install .
jac --version
```

## CLI Setup

Create your user-global JAC workspace:

```bash
jac init --global
```

This creates `~/.jac/settings.json` for non-secret settings and `~/.jac/.env`
for provider credentials. The default provider is Pydantic AI Gateway, but the
interactive setup can configure Gateway, Anthropic, OpenAI, Google AI Studio,
Ollama, OpenRouter, or LiteLLM.

For a project-local workspace, run from the repo root:

```bash
jac init
jac init --env-local   # also create .agents/.env.local placeholders
```

Project files live under `.agents/`. Secrets stay in `.agents/.env.local`,
which JAC adds to `.gitignore`.

## Model Configuration

JAC uses three model tiers: `scout`, `worker`, and `architect`. Each tier can
hold one or more model ids; today JAC uses the first model in the selected tier.

Example `~/.jac/settings.json`:

```json
{
  "active_profile": "default",
  "default_provider": "gateway",
  "default_tier": "worker",
  "model_tiers": {
    "scout": ["gateway/google-vertex:gemini-3.1-flash-lite-preview"],
    "worker": ["gateway/anthropic:claude-sonnet-4-6"],
    "architect": ["gateway/anthropic:claude-opus-4-6"]
  }
}
```

Credentials are stored in dotenv files, not JSON:

```dotenv
PYDANTIC_AI_GATEWAY_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
LITELLM_API_BASE=https://litellm.example/v1
LITELLM_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Profiles

Profiles let you keep multiple provider setups and switch between them. For
example, use LiteLLM at work and Ollama at home:

```bash
jac profile add office --provider litellm
jac profile add home --provider ollama
jac profile list
jac profile use office
jac profile use home
```

Profile secrets use profile-scoped env names and are checked before the global
provider env names:

```dotenv
JAC_PROFILE_OFFICE_LITELLM_API_BASE=https://company-litellm.example/v1
JAC_PROFILE_OFFICE_LITELLM_API_KEY=...
JAC_PROFILE_HOME_OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Common Commands

```bash
jac "say hello"                  # one-shot prompt
jac run "summarize @README.md"   # explicit one-shot form
jac chat                         # interactive REPL
jac --model openai:gpt-5.4 "hi"  # one-call model override
jac --debug chat                 # verbose developer trace output
jac doctor                       # non-secret setup diagnostics
jac config                       # alias for current diagnostics
jac profile current              # show active profile
jac profile list                 # list configured profiles
jac profile use <name>           # switch active profile
jac profile add <name>           # configure a new profile
```

Interactive slash commands:

```text
/help
/model [model-id]
/tier [scout|worker|architect]
/mode [autopilot|hitl]
/approval [interactive|auto-edit|yolo]
/debug [on|off]
/params [temperature|max_tokens] <value>
/context
/cost
/history [n]
/save [file]
/undo
/clear
/capabilities
/quit
```

Use `@path` to attach files and `!command` to run local shell commands from
chat. User-typed shell commands run directly (with destructive-command
confirmation); approval policy applies to agent-requested tool calls.

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
just qa          # lint + typecheck + full test suite
just precommit-install
just precommit-run
just fix         # lint + format with auto-fix
just clean       # nuke caches and build artifacts
```
