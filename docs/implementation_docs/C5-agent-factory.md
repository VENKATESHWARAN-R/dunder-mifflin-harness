> **Status:** Draft · **Last revised:** 2026-05-03 · **Type:** implementation plan (C5)

# C5 — Agent Factory & Config Loader

## Goals

- Introduce `src/jac/agents/` as the **only** site that calls `pydantic_ai.Agent(...)`.
- Define `agents/base.py::config_loader(...)` per the [`MCP_INTEGRATION.md`](../contracts/MCP_INTEGRATION.md) contract: read `agent_configs` + `run_mcp_servers`/`run_skills`, resolve `allowed_tools`, compose the system prompt, build the agent.
- Activate the `agent_configs`, `run_mcp_servers`, `run_skills` tables (per [`STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md) Activation Sequence).
- Reuse the existing **model factory** (`runtime/models.py::build_pydantic_model`) and tier resolution (`Settings.resolve_model_selection`) — agents must not re-implement provider/credential logic.
- Replace the inline `Agent(...)` call inside `RunCoordinator.build_agent` with a call into the new factory.
- Keep the runtime usable as a library/SDK (no CLI assumptions), preserving the existing `EventBus` boundary.

## Non-goals

- No graph orchestration, planner/builder roles, evaluation, retries, cost tracking, hooks, or skills classification — those are C6+.
- No live MCP transport (C17); MCP wiring exists but only `local` tools are resolved end-to-end now. We still build `MCPServerStdio`/`MCPServerStreamableHTTP` toolsets when rows exist, so C17 only has to add transport tests.
- No `/agents`, `/skills`, `/mcp` slash commands (defer to C16/C18).
- No hot-reload semantics (C20). The factory rebuilds the agent each time it is called; coordinator caching stays as today (single `_agent`).
- No agent_configs *seeding* from disk. C2 seeds skills/MCP from files; agent role definitions are seeded by the factory itself when a run starts (a single default `chat` role for now), to keep the existing single-agent C0 behaviour green.

## Anchors (read-only context, do not duplicate)

- [`docs/contracts/MCP_INTEGRATION.md`](../contracts/MCP_INTEGRATION.md) — instantiation flow, `allowed_tools` resolution, skills injection, tier resolution.
- [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md) — `agent_configs`, `run_mcp_servers`, `run_skills` columns and FK shape.
- [`docs/contracts/TOOLS_CONTRACT.md`](../contracts/TOOLS_CONTRACT.md) and `src/jac/tools/__init__.py::TOOL_REGISTRY` — local tool entries (`filesystem`, `filesystem:read`, `shell`, `shell:read`).
- [`docs/reference/PHILOSOPHY.md`](../reference/PHILOSOPHY.md) §"agents/base.py is the only place Agent() is called".
- `src/jac/runtime/models.py` — `build_pydantic_model(selection, settings)` does provider construction.
- `src/jac/config.py` — `Settings.resolve_model_selection`, `default_model_tiers`, `provider_definition`, `require_model_credentials`.
- `src/jac/onboarder.py` — emits `settings.json::model_tiers` and `.env`. The factory consumes that *through* `Settings`; no direct file reads here.

## Architectural alignment

### "Agents are independent" / SDK posture

`src/jac/agents/` must be importable without a CLI, a CLI session, or a process tty. Concretely:

- It depends only on `jac.config`, `jac.state`, `jac.tools`, `jac.runtime.models` (and `jac.runtime.events` for event emission).
- It must **not** import anything from `jac.cli.*`, `jac.runtime.session.SessionState`, or `jac.runtime.coordinator`. The session is a runtime detail; the factory takes plain values (`settings`, `state`, `run_id`, `role`).
- The factory returns a Pydantic AI `Agent` object the caller can use directly — `async with agent: result = await agent.run(...)` — without going through the coordinator. That is the SDK seam.

This means a downstream library user can do, after `pip install jac`:

```python
from jac.config import Settings
from jac.state import open_state_store
from jac.agents import config_loader, ensure_default_run_config

settings = Settings()
state = await open_state_store(settings.state_db_path)
run_id = "..."
await ensure_default_run_config(state, run_id, role="chat")
agent = await config_loader(state=state, settings=settings, run_id=run_id, role="chat")
async with agent:
    result = await agent.run("hi")
```

The coordinator becomes one consumer of this seam; CLI / browser / A2A become other consumers.

### Model tier + factory alignment

`agent_configs.model_tier` and `model_override` must round-trip through the same path the CLI uses today, so onboarding's `settings.json::model_tiers` and `JAC_MODEL` continue to drive tier resolution:

1. The factory builds a `ModelSelection` by calling `settings.resolve_model_selection(model_override=cfg.model_override, tier=cfg.model_tier)`.
2. It calls `build_pydantic_model(selection, settings)` (already provider-aware, already runs `require_model_credentials`).
3. It passes the result as `Agent(model=...)`.

This deliberately reuses **the same** code path that `RunCoordinator.build_agent` uses today, so:

- Onboarding profiles (`profile_env_name`, `JAC_PROFILE_*` env vars) still work — `Settings` already resolves them.
- `settings.json::model_tiers` from `init_global_workspace` is still authoritative when no `model_override` is set.
- Adding a new provider only requires editing `runtime/models.py` once.

The `agent_configs.model_tier` column stores the **logical tier** (`scout`/`worker`/`architect`); concrete model strings stay in workspace settings. This matches the schema: a CLI `/tier` command (C8) only writes the tier label, and the next config_loader call materialises whatever `model_tiers[tier][0]` resolves to today.

## Files

### New

- `src/jac/agents/__init__.py` — public surface: `config_loader`, `AgentConfig`, `ensure_default_run_config`, errors.
- `src/jac/__init__.py` — add `build_agent = config_loader` re-export so `from jac import build_agent` works for SDK consumers (notebooks, peer harnesses).
- `src/jac/agents/base.py` — the `config_loader` function and helpers (`_build_mcp_toolsets`, `_resolve_local_tools`, `_compose_system_prompt`).
- `src/jac/agents/seeds.py` — small helper that creates a default `chat` `agent_configs` row + empty `run_skills` / `run_mcp_servers` rows when a run starts. Lets C0 behaviour keep working without forcing every caller to pre-populate the DB.
- `src/jac/state/agent_configs.py` — `AgentConfigsRepo` (CRUD on `agent_configs`).
- `src/jac/state/run_mcp_servers.py` — `RunMcpServersRepo` (insert / list-active / toggle).
- `src/jac/state/run_skills.py` — `RunSkillsRepo` (insert / list-active / toggle).
- `tests/agents/test_config_loader.py` — see Acceptance.
- `tests/state/test_agent_configs.py`, `tests/state/test_run_mcp_servers.py`, `tests/state/test_run_skills.py`.

### Modified

- `src/jac/state/__init__.py` — export the three new repos.
- `src/jac/state/db.py` — wire the three new repos onto `StateStore`.
- `src/jac/runtime/coordinator.py` — `build_agent()` becomes a thin wrapper that calls `config_loader(...)`. The `async with agent:` lifecycle moves into `submit_message`. `_agent` cache stays for now; resetting it is unchanged.
- `src/jac/agents/base.py` adds nothing CLI-shaped, but the coordinator now needs to call `ensure_default_run_config` once per `run_id` before the first turn, alongside `_ensure_run_persisted`.

### Untouched

- No new migration (all three tables already exist from `001_initial.sql`).
- No changes to `tools/`, `cli/`, `onboarder.py`, `workspace.py`, or `config.py`.

## `AgentConfig` shape

```python
@dataclass(frozen=True, slots=True)
class AgentConfig:
    config_id: str
    run_id: str
    role: str
    model_tier: str           # scout | worker | architect
    model_override: str | None
    system_prompt: str
    allowed_tools: list[str]  # already JSON-decoded
    max_context_tokens: int
    created_at: str
    updated_at: str
```

## `config_loader` signature

```python
async def config_loader(
    *,
    state: StateStore,
    settings: Settings,
    run_id: str,
    role: str = "chat",
    output_type: type | None = None,   # default: str (chat-shaped); roles that need structured output pass a Pydantic model here
    events: EventBus | None = None,    # optional; emits NodeStarted/NodeCompleted around build
) -> Agent:
    ...
```

It performs the five-step flow from `MCP_INTEGRATION.md`:

1. Load `agent_configs WHERE run_id = ? AND role = ?`. Raise `AgentConfigNotFound` if missing.
2. Load `run_mcp_servers JOIN mcp_servers ON ... WHERE run_id = ? AND (agent_role = ? OR agent_role IS NULL) AND enabled = 1 AND mcp_servers.is_enabled = 1`. Build toolsets via `_build_mcp_toolsets` (transport mapping in MCP contract).
3. Resolve `allowed_tools` plain entries against `TOOL_REGISTRY`. Strip `mcp:` entries (already handled in step 2). Unknown name → `UnknownToolError` (with the offending name and the allowed registry keys).
4. Load `run_skills JOIN skills ON ... WHERE run_id = ? AND (agent_role = ? OR agent_role IS NULL) AND enabled = 1 AND skills.is_enabled = 1 ORDER BY skills.domain ASC, skills.name ASC` ("general first" — `domain = 'general'` sorts first because it precedes domain-specific labels alphabetically; if that proves wrong we add an explicit `priority` column later, not now).
5. Build the model via `settings.resolve_model_selection(model_override=cfg.model_override, tier=cfg.model_tier)` then `build_pydantic_model(selection, settings)`.
6. Return `Agent(model=..., instructions=composed_prompt, tools=local_tools, toolsets=mcp_toolsets, output_type=str, model_settings={"temperature": ...})`. Temperature stays in session config / env for now (see Open question 3).

The function is async because every state read is async; no networking happens here.

## `ensure_default_run_config`

```python
async def ensure_default_run_config(
    state: StateStore,
    run_id: str,
    *,
    role: str = "chat",
    system_prompt: str = DEFAULT_CHAT_PROMPT,
    allowed_tools: list[str] | None = None,
    model_tier: str = "worker",
    model_override: str | None = None,
) -> AgentConfig:
    ...
```

Idempotent: returns the existing row if one is present for `(run_id, role)`. Defaults pick the existing C0 prompt verbatim and `allowed_tools=[]` so behaviour does not change. Coordinator calls this once before the first `submit_message`.

It does **not** auto-populate `run_mcp_servers` or `run_skills`. Those stay empty until C16/C18.

## Coordinator integration

```python
# runtime/coordinator.py (sketch)
async def _ensure_agent(self) -> Agent:
    if self._agent is None:
        await ensure_default_run_config(
            self.state, self.session.run_id,
            role=self.session.config.role,
            model_tier=str(self.session.config.tier or self.settings.default_tier),
            model_override=self.session.config.model,
        )
        self._agent = await config_loader(
            state=self.state, settings=self.settings,
            run_id=self.session.run_id, role=self.session.config.role,
            events=self.events,
        )
    return self._agent
```

- `SessionState` gains `config.role: str = "chat"` (default unchanged for callers).
- The CLI continues to drive `tier` / `model_override` through `SessionConfig`; the coordinator passes those into `ensure_default_run_config` so the DB row matches what the user picked.
- `submit_message` wraps the run in `async with agent:` (required by the MCP contract; harmless when there are no toolsets — Pydantic AI tolerates an empty `toolsets` list).
- If `state` is `None` (the rare "no project / no DB" path), the coordinator falls back to today's inline build path. Document this as the only supported escape hatch; everywhere else, a `StateStore` is required.

## Acceptance checks

Tests live under `tests/agents/` and `tests/state/`. Hit each via `uv run pytest <path>::<test>`.

- `test_config_loader_builds_agent_from_default_config` — call `ensure_default_run_config` then `config_loader`; assert returned object is a `pydantic_ai.Agent`, model selection matches `settings.resolve_model_selection(...)`.
- `test_config_loader_resolves_local_tools` — insert a row with `allowed_tools=["filesystem:read"]`, build agent, inspect `agent.toolset` (or `_function_toolset`) to confirm the four read-only filesystem tools registered.
- `test_config_loader_strips_mcp_prefix_and_skips_when_no_row` — `allowed_tools=["mcp:nonexistent", "filesystem:read"]` with no `run_mcp_servers` row → builds successfully with only local tools; `allowed_tools=["mcp:foo"]` *with* a `run_mcp_servers` row pointing at a stdio entry → returns an agent with one MCP toolset.
- `test_config_loader_composes_skills` — seed two skills (`general`, `python`), wire via `run_skills`, assert composed prompt ends with the skills block in domain order.
- `test_config_loader_uses_model_override_over_tier` — config row with `model_override="anthropic:claude-sonnet-4-6"`; assert built model is the Anthropic one regardless of `tier`.
- `test_config_loader_raises_on_unknown_tool` — `allowed_tools=["bogus"]` → `UnknownToolError`.
- `test_config_loader_raises_on_missing_config` — no `agent_configs` row → `AgentConfigNotFound`.
- `test_ensure_default_run_config_is_idempotent` — call twice → single row, second call returns identical `config_id`.
- `test_coordinator_uses_factory` — `RunCoordinator.submit_message` round-trips through `config_loader` (mock the model so no network call); behaviour identical to pre-change tests.
- `tests/state/*` — straightforward repo CRUD per the existing `RunsRepo`/`MessagesRepo` patterns.

Smoke checks before merging:

```bash
just test
just lint
just typecheck
uv run jac "say hello"   # behaves identically to today
uv run jac chat          # ditto
uv run jac resume <id>   # still works
```

## Migration & risk

- No DB migration — three tables already exist from `001_initial.sql`.
- `agent_configs.system_prompt` is `NOT NULL`; the default seeded prompt covers it.
- Backwards compatibility: `RunCoordinator.build_agent` keeps the same name and returns the same Pydantic AI `Agent` type. Callers (none today outside the coordinator) see no behavioural change for the default `chat` role.
- Cache miss footgun: every coordinator turn currently reuses `self._agent`. We keep that behaviour; C20 will introduce per-turn rebuilds. Documented in the coordinator docstring so future maintainers don't think it is a bug.

## Application documentation (ships alongside C5)

Now that we are entering implementation territory, JAC needs human-facing docs that
describe the product as it exists, not just the contracts that govern its internals.
These are **separate from** the contracts in `docs/contracts/` and the implementation
plans in `docs/implementation_docs/`. They live under a new `docs/guide/` folder and
are written for a reader who wants to *use* JAC, not read its specs.

### New folder & files

- `docs/guide/README.md` — index of the guide; status `Reference`.
- `docs/guide/getting-started.md` — install (`uv tool install jac` or editable from this checkout), `jac init --global`, first run, `jac doctor`. Status `Reference`.
- `docs/guide/architecture.md` — narrative tour of `cli/ → runtime/ → agents/ (new in C5) → tools/ → state/`, with the current Mermaid diagram derived from the C0..C5 shipped surface. Cross-links to `PHILOSOPHY.md`, `EVENT_CONTRACT.md`, `MCP_INTEGRATION.md`, `STATE_SCHEMA.md`. Status `Reference`.
- `docs/guide/usage.md` — running `jac "..."`, `jac chat`, `jac resume`, `jac init`, `jac doctor`; `@`-file attachments; `!`-shell shortcuts; slash commands shipped today; environment + profile selection. Status `Reference`.
- `docs/guide/sdk.md` — using `jac` as a library: `from jac import build_agent`, lifecycle (`async with agent:`), wiring your own `StateStore`, embedding `RunCoordinator` in another process, no-CLI examples. Status `Reference`. Created in C5 because it documents the new SDK seam.
- `docs/guide/configuration.md` — `~/.jac/` vs `<repo>/.agents/`, `settings.json`, profiles, dotenv layering, model tiers, provider env vars. Cross-links to `WORKSPACE.md`. Status `Reference`.

### Index updates

- Add a new "Guide" section to `docs/README.md` listing the six files above.
- `CLAUDE.md` "Core Documents" section gains a "Guide" subsection pointer so future agents know it exists. (Single-line entries; do not duplicate content.)

### Scope guardrail

Guide pages describe **shipped behaviour only** (C0..C5). They must not promise C6+
features. When a later component ships, the relevant guide page is updated as part
of that component's implementation doc. This keeps the guide honest and prevents it
from drifting into roadmap territory.

### Acceptance for the guide work

- A reader who has only seen `docs/guide/` can install JAC, run `jac init --global`, run a chat session, resume it, and import `build_agent` from a Python script.
- Every guide page has the standard status header, dated 2026-05-03 on first commit.
- `docs/README.md` lists all six guide pages.
- No guide page contradicts a `Locked` contract.

## Decisions (confirmed 2026-05-03)

1. **Default role name.** `"chat"` — locked. `planner`/`builder` arrive in C6.
2. **Skills ordering.** Defer the explicit `priority` column. Sort by `domain ASC, name ASC` for now; revisit in C16 if ordering matters.
3. **Temperature / model settings.** Option (a): keep `temperature` on `SessionConfig.model_params`, pass via `Agent(model_settings=...)`. No column added to `agent_configs` in C5. Revisit in C8 when tier routing needs richer per-role settings.
4. **`output_type`.** `str` for C5. Structured output (e.g. `pydantic.BaseModel` outputs) lands per-role in C6 (planner) and any later role that needs it. The factory should accept an optional `output_type` argument so callers can opt in without changing the row schema; default stays `str`.
5. **SDK seam.** Expose both:
   - `from jac.agents import config_loader, ensure_default_run_config, AgentConfig` (primary).
   - `from jac import build_agent` as a package-root re-export of `config_loader` for convenience in scripts/notebooks.
