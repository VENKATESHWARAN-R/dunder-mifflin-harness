# Substrate Boundary Contract

> **Status:** Locked · **Last revised:** 2026-05-08 · **Type:** contract
>
> Source-of-truth brainstorm: [`lab/brainstorm/2026-05-08-jac-reset-from-scratch.md`](../../lab/brainstorm/2026-05-08-jac-reset-from-scratch.md). Locks the boundary between code and the substrates that hold tunable values, prose, and state.

## The rule

> If it's **declarative and stable**, it's **TOML**.
> If it's an **agent-shape spec consumable by Pydantic AI**, it's **YAML**.
> If it's **runtime-context settings** (provider/profile/env-overlaid), it's **JSON**.
> If it's **prose meant to be read** by humans or agents, it's **Markdown**.
> If it's **structured state with relationships**, it's **SQLite**.
> Code never holds config; config never holds code.

This rule is load-bearing for [`PHILOSOPHY.md` principle B5](../reference/PHILOSOPHY.md#5-config-as-data-not-config-as-code) (config-as-data, not config-as-code). It exists so AI agents working on JAC don't accidentally reintroduce Python literals as the canonical source of values.

## The five substrates

| Substrate | Holds | Where on disk | Why |
|---|---|---|---|
| **YAML** | Persona blueprints. Agent specs consumable by `pydantic_ai.Agent.from_file()` / `Agent.from_spec()`. | `src/jac/data/personas/<role>.yaml` (shipped) ← `~/.jac/personas/<role>.yaml` (user) ← `<repo>/.agents/personas/<role>.yaml` (project) | Pydantic AI native; supports `model`, `instructions` (with `{{deps}}` templating), `capabilities`, `model_settings`. The factory layer fills in what YAML can't carry (function tools, output_type, deps_type, MCP server transports). |
| **TOML** | Vendor-fact data JAC ships. Model context windows, tier defaults per provider, package metadata. | `src/jac/data/model_specs.toml`, `pyproject.toml` | Vendor knowledge; rarely changes; Python-native (PEP 518). Comments + nesting + typed values without YAML's footguns. |
| **JSON** | User settings, profiles, MCP server stubs. | `~/.jac/settings.json`, `<repo>/.agents/settings.json`, `~/.jac/mcp/<name>.json`, `<repo>/.agents/mcp/<name>.json` | `pydantic-settings` already uses JSON + env overlay (`JAC_*` and profile-scoped `JAC_PROFILE_<SLUG>_*`). MCP ecosystem standard is JSON (mirrors Claude Desktop's `mcp.json`). |
| **Markdown** | Skills (with frontmatter for `name` / `domain` / `version`). Project instructions (`<repo>/AGENTS.md`). User instructions (`~/.jac/JAC.md`). Archived plans, brainstorms. | `src/jac/data/skills/<name>.md` (shipped, optional) ← `~/.jac/skills/<name>.md` (user) ← `<repo>/.agents/skills/<name>.md` (project); `<repo>/AGENTS.md`; `~/.jac/JAC.md` | Free-form prose for human + agent consumption. Frontmatter gives structured fields without losing readability. The seeder mirrors disk → SQLite for query, but the **canonical form is the file on disk**. |
| **SQLite** | Per-run state — `runs`, `messages`, `attempts`, `tasks`, `agent_configs` (per-run overrides), `mcp_servers` / `skills` registries, `run_mcp_servers` / `run_skills` bindings. | `<workspace>/state.db` | Queryable, FK-enforced, transactional. The ledger that proves or disproves the hypothesis must be queryable; markdown can't do that without re-parsing. |
| **Python** | Code only. Runtime, tools, factory, nodes (M2+), workflows (M3+), schema migrations. | `src/jac/**/*.py`, `src/jac/state/migrations/*.sql` (migration content is SQL, ordering is filename) | If a value changes without code review, it's config; move it out. |

## Source precedence

For any layered value (persona, skill, MCP server, settings):

```
project   <repo>/.agents/<thing>    (highest precedence — wins)
user      ~/.jac/<thing>            (user-global override)
shipped   src/jac/data/<thing>      (lowest precedence — JAC default)
```

A higher-precedence file with the same `name` overrides the lower one. Duplicate names within one scope are **configuration errors**, not silent merges.

## Decision tree — where does this value go?

```
Is it a Python expression / control flow / runtime behaviour?
  └─ Yes → Python (code).

Does an LLM need to read it as prose?
  └─ Yes → Markdown (skills, AGENTS.md, JAC.md).

Does Pydantic AI's Agent.from_file/from_spec accept it directly
(model, instructions, capabilities, model_settings)?
  └─ Yes → YAML (persona file).

Is it user-runtime-context (env-overlaid, profile-scoped, secrets)?
  └─ Yes → JSON (settings.json + .env).

Is it shipped vendor data (model windows, tier maps, package metadata)?
  └─ Yes → TOML.

Does it have relationships / need queries / mutate per run?
  └─ Yes → SQLite (with a migration).

None of the above?
  └─ You probably don't need to add it. Re-read PHILOSOPHY principle B2 (thin spine).
```

## What this rules out

- **Python files holding system prompts.** No `personas.py` strings. Persona prompts are in YAML.
- **Python files holding mode addendums.** No `modes.py` Python literals for `/init` / `/plan` modes. If a mode is needed, it's part of a YAML persona or a workflow file.
- **Python files holding model lists.** Tier defaults live in `data/model_specs.toml`, user choices in `settings.json`, per-run overrides in SQLite.
- **`if role == "manager":` branches in code.** Role is config; dispatch goes through data lookups.
- **Tunable thresholds buried as integer literals** without a config-file analogue (e.g. `MAX_CONTEXT = 175_000` in Python). If the threshold is genuinely tunable, it lives in config.
- **Agent personas as Python classes.** No `class Manager(Persona): ...`. Personas are YAML files.
- **Skill content in TOML.** Skills are prose → Markdown. Frontmatter for metadata only.
- **Run state in Markdown.** Append-only event logs need indexes and joins — SQLite.

## Current violations being fixed in M1

When the M1 cut lands, these existing violations of the rule disappear:

| Today | Violation | M1 fix |
|---|---|---|
| `src/jac/agents/personas.py` (~112 LOC) | Persona system prompts as Python multi-line strings | Replaced by `src/jac/data/personas/scott.yaml`. Other personas YAML files arrive when their personas do (M2+). |
| `src/jac/agents/modes.py` (~53 LOC) | `MODE_PROMPTS` registry of slash-mode prompt addendums in Python | Removed. M1 has no `/plan` / `/init` modes; when planner arrives in M2+, modes are part of workflow selection or persona YAML, not inline prompt injection. |
| `src/jac/agents/seeds.py` role pattern matching | `ensure_manager_config` / `ensure_builder_config` / `ensure_planner_config` baking role-specific defaults into Python | Replaced by single `ensure_scott_config` that reads `data/personas/scott.yaml`. |
| `src/jac/runtime/coordinator.py` `if role == "manager":` branches | Coordinator special-casing on role names | Replaced by config-driven dispatch reading `agent_configs.role` from DB. |
| `src/jac/agents/plans.py` `Plan` Pydantic model + structured-output literals | Output-type schema lives in code (necessary), but planner-specific literals in `agents/` violate B1 too | When planner arrives, output type stays Python (it's a schema), but persona-specific text/tier/tools come from YAML. |

## Notes on YAML personas

Pydantic AI's `Agent.from_file('agent.yaml')` and `Agent.from_spec(dict, deps_type=...)` accept what YAML can carry. The factory at `agents/base.py` fills in the rest **at instantiation time, in code**:

| Carried by YAML | Carried by Python factory at instantiation |
|---|---|
| `model` (or `model_tier` resolved Python-side via `tier_defaults_for(provider)` → settings → SQLite override) | `tools=` resolved from `TOOL_REGISTRY` by name list |
| `instructions:` with `{{deps_field}}` templating | `output_type=` resolved from import path string in YAML |
| `capabilities:` (Thinking, WebSearch, MCP-as-capability with nested config) | `deps_type=` passed as kwarg |
| `model_settings:` | Approval middleware wraps tool functions |
| | MCP toolsets built from `mcp_servers` registry (matched by name) |
| | Hooks registered for telemetry / audit |

This split is not a workaround — it's the honest division of "what YAML can express" and "what is genuinely Python." Code stays Python; config stays config.

## Migration policy for substrate changes

- Adding a value: pick the substrate via the decision tree, place it, document where in the relevant contract.
- Moving a value between substrates: if a value graduates from Python literal → YAML (or TOML / JSON), update this contract's "Current violations" section as a record.
- Splitting a substrate: don't. The five substrates are the answer; further subdivision is over-spec.
