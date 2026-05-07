# C8 — Model Tier Routing

> **Status:** Ready for implementation
> **Last revised:** 2026-05-07
> **Depends on:** C5 (shipped), C7 (shipped 2026-05-07)
> **Roadmap entry:** [`docs/ROADMAP.md` §C8](../ROADMAP.md#c8--model-tier-routing)
> **Contract touchpoints:** [`docs/contracts/EVENT_CONTRACT.md`](../contracts/EVENT_CONTRACT.md), [`docs/contracts/CLI_DESIGN.md`](../contracts/CLI_DESIGN.md)
> **Reference:** [`docs/reference/IDEA.md` §2 Model Buckets](../reference/IDEA.md)

---

## What's already shipped (most of C8)

A lot of the C8 surface is already in the tree from earlier components. C8's job is the last 20% — wire the missing event, fix one persona-tier semantic bug, polish the slash UX, and lock the contracts.

| Capability | Status | Location |
|---|---|---|
| Provider-aware tier defaults (7 providers) | Done (C5) | `src/jac/config.py` `PROVIDER_DEFINITIONS[*].tier_defaults`, `default_model_tiers(provider)` |
| `Settings.resolve_model_selection(model_override, tier)` | Done (C5) | `src/jac/config.py:450` |
| `/model <id>` and `/tier <scout\|worker\|architect>` (aliases `/m`, `/t`) | Done (C6) | `src/jac/cli/app.py:200,209` |
| Tier completion in `JacCompleter` | Done (C6) | `src/jac/cli/input.py:45` |
| `agent_configs.model_tier` / `model_override` columns and seeders | Done (C5/C6) | `src/jac/agents/seeds.py`, migrations |
| Persona default tiers (manager=worker, builder=worker, planner=architect) | Done (C6/C6b) | `src/jac/agents/personas.py:21` |
| Mid-run propagation to `agent_configs` rows | Done (C6) | `coordinator._ensure_agent` |
| Per-attempt `tier` recorded in `attempts` rows | Done (C6) | `state/attempts.py:93` |
| `/usage` aggregation by tier | Done (C7) | `state/attempts.py:252` (`totals.by_tier`) |
| `spawn_minion(tier=Literal["scout","worker"])` | Done (C6c) | `agents/spawn.py:50` |

---

## Scope of C8

C8 adds:

1. **`SessionConfigChanged` event** — defined in `EVENT_CONTRACT.md` (Not started); add the dataclass and emit from slash handlers that mutate session config.
2. **Persona-default tier preservation under `/tier`** — fix the current behavior where `/tier scout` silently demotes Pam from architect to scout.
3. **Slash UX polish** — `/tier` and `/model` echo the resolved model id and warn on missing creds.
4. **Tier defaults helper for 2+ providers** — light public-API surface so docs/tests don't reach into `PROVIDER_DEFINITIONS` directly.
5. **Tests, doc updates, version bump.**

## Non-goals

- **Dwight (evaluator) persona seed.** C9 ships Dwight's persona row. C8 only guarantees the routing infra is ready for him; the C8 tier defaults table in the roadmap names "Dwight → Worker" prospectively.
- **Per-role tier slash command** (`/tier <role> <tier>`). Designed-compatible with C8's manager-scoped `/tier`, but not built in C8. See "Future extension" below.
- **HR-agent dynamic escalation** (IDEA §3.1). Separate component.
- **New providers in `PROVIDER_DEFINITIONS`.** Existing seven (gateway, anthropic, openai, google-gla, ollama, openrouter, litellm) cover the "≥2 providers" requirement many times over.
- **Pricing data.** Out of scope (same reasoning as C7).

---

## Design decisions (locked)

### D1. `/tier` scopes to the manager only; specialists keep persona defaults

The current `coordinator._ensure_agent` propagates `session.config.tier` to **every** seeded role (manager, builder, planner). That contradicts "Defaults per persona: Pam → Architect" in the roadmap — running `/tier scout` silently demotes Pam.

**Rule:** `session.config.tier` is treated as the **manager's tier**. Specialist roles (builder, planner, future evaluator/minions) use `Persona.default_tier` from `personas.py`. Only `model_override` (an explicit pin from `/model`) applies session-wide to all roles.

This matches the roadmap's persona-defaults table and avoids the silent-demotion footgun. It does mean `/tier scout` is no longer a session-wide cheap mode; that's intentional.

### D2. Future per-role tier override path is reserved, not built

The natural extension when this proves insufficient is `/tier <role> <tier>`, persisted on `SessionConfig` as a `role_tier_overrides: dict[str, ModelTier]`. **C8 does not ship this.** Per CLAUDE.md ("Don't design for hypothetical future requirements"), we add the field only when there's a concrete consumer. Document the extension point in `docs/dev/agents-layer.md` so a future component can plug in without re-deriving the design.

### D3. `SessionConfigChanged` shape

Per `EVENT_CONTRACT.md:334`:

```python
@dataclass(frozen=True, slots=True)
class SessionConfigChanged(RuntimeEvent):
    key: str        # 'model' | 'tier' | 'mode' | 'approval_mode' | 'params' | 'debug'
    old_value: Any
    new_value: Any
```

Emitted **after** the mutation lands (so subscribers see consistent state). One emission per command invocation that actually changes a value (no-op if user runs `/tier worker` while already on worker). Renderer's role: log a one-line confirmation; the toolbar already polls `session_config_source` so it picks up the new value naturally.

`StateUpdated` (the generic precursor) is **kept** for now — `EVENT_CONTRACT.md` says `SessionConfigChanged` "replaces" it, but no code currently emits `StateUpdated`. Removing it is a clean-up that can ride on this PR; do that as part of Step 5.

### D4. `/tier` UX — echo resolved model

When the user runs `/tier worker`, the renderer prints both the tier change and the resolved model id:

```
Preferred tier set to: worker
  → manager will use: gateway/anthropic:claude-sonnet-4-6
```

Computed via `settings.resolve_model_selection(tier="worker")`. If `model_override` is also set, note that it pins the model regardless of tier:

```
Preferred tier set to: worker
  (note: /model override gateway/anthropic:claude-haiku-4-5 still applies)
```

### D5. `/model` UX — provider-credential warning

When `/model <id>` is run, infer the provider via `infer_provider`, look up the credential requirements via `credential_requirements_for_model`, and if any required env var is missing, emit a `WarningRaised` event with a one-line hint. Do **not** block the command — the user might be about to set the env var or use a profile.

---

## Key files and integration points

| File | What it is | C8 touches it? |
|---|---|---|
| `src/jac/runtime/events.py` | Typed events | Yes — add `SessionConfigChanged`; remove unused `StateUpdated` (per D3) |
| `src/jac/cli/app.py` | `ChatApp` slash handlers | Yes — emit `SessionConfigChanged` from `/model`, `/tier`, `/mode`, `/approval`, `/params`, `/debug`; tier/model UX echoes |
| `src/jac/cli/renderer.py` | Rich renderer | Yes — subscribe `SessionConfigChanged` (one-line confirmation); migrate any `StateUpdated` handler |
| `src/jac/runtime/coordinator.py` | `RunCoordinator._ensure_agent` | Yes — stop propagating `session.config.tier` to builder/planner rows; keep `model_override` propagation |
| `src/jac/config.py` | `Settings`, `PROVIDER_DEFINITIONS` | Yes — add a thin `tier_defaults_for(provider)` public helper (delegates to `default_model_tiers`); no new provider data |
| `src/jac/agents/seeds.py` | Per-role seeders | No code change — already honors `Persona.default_tier`; verify behavior |
| `src/jac/agents/personas.py` | Persona table | No change in C8 (evaluator is C9) |
| `docs/contracts/EVENT_CONTRACT.md` | Locked | Yes — flip `SessionConfigChanged` from "Not started" to "Done"; bump revision date; remove `StateUpdated` row if removed in code |
| `docs/contracts/CLI_DESIGN.md` | Locked | Yes — clarify `/tier` is manager-scoped; specialists keep persona defaults; document `/tier` echo and `/model` cred warning; bump revision date |
| `docs/dev/agents-layer.md` | Layer doc | Yes — document the tier-resolution chain (model_override > session.tier (manager only) > persona.default_tier > settings.default_tier > JAC_MODEL); note the future per-role extension point |
| `docs/dev/runtime-layer.md` | Layer doc | Yes — `SessionConfigChanged` emission rules |
| `docs/dev/cli-layer.md` | Layer doc | Yes — `/tier` and `/model` UX |
| `docs/ROADMAP.md` | Living | Yes — flip C8 to **Done** with a 2026-MM-DD ship line; trim the body to reflect what shipped vs C9 carry-over (Dwight) |
| `pyproject.toml` + `src/jac/__init__.py` | Version | Yes — bump (new event, semantics change for `/tier`) |
| `tests/test_c8_tier_routing.py` | **New file** | Yes — create |
| `tests/test_runtime_contracts.py` | Existing | Yes — add `SessionConfigChanged` shape assertion |

**No SQL migration.** `agent_configs.model_tier` / `model_override` already populated.

---

## Step-by-step implementation

### Step 1 — Add `SessionConfigChanged` event

In `src/jac/runtime/events.py`:

```python
@dataclass(frozen=True, slots=True)
class SessionConfigChanged(RuntimeEvent):
    key: str
    old_value: Any
    new_value: Any
```

Re-export from `runtime/__init__.py` if other events are exported there. Remove `StateUpdated` (search confirms zero emission sites; only the dataclass and any test referencing it).

### Step 2 — Emit from slash handlers

In `src/jac/cli/app.py`, every slash command that mutates `session.config` (or `approvals.mode`) now emits a `SessionConfigChanged` after the mutation, only if the value actually changed. Apply to:

- `/model` — key=`"model"`
- `/tier` — key=`"tier"`
- `/mode` — key=`"mode"`
- `/approval` — key=`"approval_mode"`
- `/params <key> <value>` — key=`"params"`, value is the full updated dict
- `/debug` — key=`"debug"`

Pattern (model_command):

```python
async def model_command(args: str) -> None:
    model = args.strip()
    if not model:
        self.renderer.print_info(f"Current model: {self.session.config.model}")
        return
    old = self.session.config.model
    if old == model:
        self.renderer.print_info(f"Model already set to: {model}")
        return
    self.session.config.model = model
    self.coordinator.reset_agent()
    self.renderer.print_info(f"Model set to: {model}")
    self._maybe_warn_missing_creds(model)  # see Step 4
    await self.events.emit(
        SessionConfigChanged(key="model", old_value=old, new_value=model)
    )
```

Mirror for the others.

### Step 3 — Renderer subscription

In `src/jac/cli/renderer.py`, add a small handler (debug-mode-only, or always-on as a single-line trace — match the existing pattern for `WarningRaised`/`ApprovalResolved`). The toolbar refresh is automatic because it polls `session_config_source`.

```python
def _on_session_config_changed(event: SessionConfigChanged) -> None:
    if not self._debug:
        return
    self.print_dim(f"[config] {event.key}: {event.old_value!r} → {event.new_value!r}")
```

Wire in the renderer's setup. Keep it terse — it's a debug breadcrumb, not user-facing chatter.

### Step 4 — `/tier` echoes the resolved model; `/model` warns on missing creds

Helpers in `cli/app.py`:

```python
def _resolved_model_for_tier(self, tier: ModelTier) -> str:
    selection = self.settings.resolve_model_selection(
        model_override=self.session.config.model,
        tier=str(tier),
    )
    return selection.model_ref

def _maybe_warn_missing_creds(self, model: str) -> None:
    from jac.config import credential_requirements_for_model, infer_provider
    provider = infer_provider(model, self.settings.default_provider)
    groups = credential_requirements_for_model(model, provider=provider)
    missing = [
        g for g in groups
        if not any(self.settings.optional_env_var(name) for name in g)
    ]
    if not missing:
        return
    names = " or ".join(name for g in missing for name in g)
    asyncio.create_task(self.events.emit(
        WarningRaised(message=f"{model} needs {names}; set it before the next turn.")
    ))
```

Update the `/tier` handler so the echo is:

```
Preferred tier set to: worker
  → manager will use: gateway/anthropic:claude-sonnet-4-6
```

When `model_override` is set, append a parenthetical that the override still wins.

### Step 5 — Coordinator stops blanket-overriding specialist tiers

In `src/jac/runtime/coordinator.py:_ensure_agent` (around lines 169–229):

**Before** (current — overrides every role with session tier):
```python
expected_tier = str(self.session.config.tier or self.settings.default_tier)
await ensure_manager_config(..., model_tier=expected_tier, model_override=...)
await ensure_builder_config(..., model_tier=expected_tier, model_override=...)
await ensure_planner_config(..., model_override=...)
# Then for each role: if cfg.model_tier != expected_tier or model_override differs → update.
```

**After** (manager-scoped tier; specialists keep persona defaults; model_override still propagates):

```python
manager_tier = str(self.session.config.tier or self.settings.default_tier)
manager_cfg = await ensure_manager_config(
    self.state, self.session.run_id,
    model_tier=manager_tier,
    model_override=self.session.config.model,
)
# Specialists: do NOT pass model_tier — let the seeder fall back to persona.default_tier.
await ensure_builder_config(
    self.state, self.session.run_id,
    model_override=self.session.config.model,
)
await ensure_planner_config(
    self.state, self.session.run_id,
    model_override=self.session.config.model,
)

# Manager row update — only sync tier and override.
if manager_cfg.model_tier != manager_tier or manager_cfg.model_override != self.session.config.model:
    await self.state.agent_configs.update(
        manager_cfg.config_id,
        model_tier=manager_tier,
        model_override=self.session.config.model,
    )

# Specialists: only sync model_override; never overwrite their tier from session state.
for role in ("builder", "planner"):
    row = await self.state.agent_configs.get_by_run_and_role(self.session.run_id, role)
    if row is not None and row.model_override != self.session.config.model:
        await self.state.agent_configs.update(
            row.config_id,
            model_override=self.session.config.model,
        )
```

This is the load-bearing behavior change. After this:

- `/tier scout` → next manager attempt is on the scout-tier model; planner attempts (when Pam runs) stay on architect tier.
- `/model gateway/anthropic:claude-haiku-4-5` → all roles use that model (explicit pin, intentional).
- No `/tier` set → settings.default_tier (worker) used for manager; persona defaults govern specialists.

### Step 6 — `tier_defaults_for(provider)` helper

In `src/jac/config.py`:

```python
def tier_defaults_for(provider: str = DEFAULT_PROVIDER) -> dict[str, list[str]]:
    """Return the shipped tier→model defaults for a provider.

    Public API surface for docs/tests. Internally delegates to
    `default_model_tiers`, which already returns editable lists.
    """
    return default_model_tiers(provider)
```

Re-export at the package level if other config helpers are. Used by the test in Step 7 and referenced from `docs/dev/agents-layer.md`.

### Step 7 — Tests (`tests/test_c8_tier_routing.py`)

Cover six things:

1. **Mid-session tier swap is visible in the next attempt.** Build a coordinator with a fake state, run a turn, run `/tier scout`, run another turn, assert the second manager attempt's `tier == "scout"` and `model` resolves to the scout default.
2. **`/tier` does NOT change planner's tier.** Seed planner row with default architect; run `/tier scout`; trigger an `_ensure_agent` cycle; assert planner row's `model_tier` is still `"architect"`.
3. **`/model` propagates to all role rows.** After `/model gateway/anthropic:claude-haiku-4-5`, manager / builder / planner rows all carry that override.
4. **`SessionConfigChanged` is emitted exactly once per real change** with correct (key, old, new). No-op `/tier worker` while on worker emits nothing.
5. **`/tier` echo includes the resolved model id.** Capture renderer output; assert the second line.
6. **Tier-defaults helper covers ≥2 providers.** `tier_defaults_for("gateway")` and `tier_defaults_for("anthropic")` both return non-empty dicts with all three tiers.

Plus update `tests/test_runtime_contracts.py` to assert the `SessionConfigChanged` field shape and any test that touched `StateUpdated` (likely none) to remove it.

### Step 8 — Documentation updates

**Locked contracts (bump `Last revised`):**

- `docs/contracts/EVENT_CONTRACT.md` — flip `SessionConfigChanged` row to "Done"; remove `StateUpdated` row if the dataclass is gone.
- `docs/contracts/CLI_DESIGN.md` — under `/tier`, add: "Sets the manager's preferred tier. Specialists (builder, planner, evaluator) keep their persona defaults. To change a specialist's tier, see future per-role command (not yet implemented)." Document `/tier` echo and `/model` credential warning.

**Dev docs:**

- `docs/dev/agents-layer.md` — add a "Tier resolution chain" section: explicit `model_override` from `/model` > session `tier` (manager only) > `Persona.default_tier` (specialists) > `settings.default_tier` > `JAC_MODEL`. Note the future `role_tier_overrides` extension point.
- `docs/dev/runtime-layer.md` — `SessionConfigChanged` emission rules.
- `docs/dev/cli-layer.md` — `/tier` echo + `/model` warning UX.

**Living:**

- `docs/ROADMAP.md` — flip C8 to **Done** with a 2026-MM-DD ship line at the top of the file. Trim the body so the persona-defaults note matches what actually shipped (Dwight is C9).
- `README.md` — quickstart already shows `/tier`; check whether a one-line note about the manager-only scope is worth adding (probably not — this is internal nuance).

**Versioning:**

- `pyproject.toml` + `src/jac/__init__.py` — bump (new event + behavior change to `/tier`). `0.3.8` → `0.3.9` is appropriate (additive event, behavior fix, no breaking CLI surface).
- `uv lock` only if deps change (none expected).

### Step 9 — Smoke checks

Per CLAUDE.md:

```bash
just qa
uv run jac --help
uv run jac chat
# inside chat:
#   /tier worker        → echo includes resolved model
#   /tier scout         → echo
#   /model claude-haiku-4-5  → if creds missing, warning event surfaces
#   say hi              → manager attempt uses scout-tier model
#   /plan write a hello-world script  → planner attempt uses architect-tier model (NOT scout)
#   /usage              → at least two tiers present in by_tier
```

---

## Acceptance checks

A reviewer should be able to verify each:

1. **Mid-session tier swap works.** `/tier scout` mid-chat → next prompt's `attempts.tier == 'scout'` and `attempts.model` is the scout default for the configured provider.
2. **Persona default preserved.** After `/tier scout`, running `/plan ...` produces a planner attempt with `tier == 'architect'`. `agent_configs.model_tier` for the planner row is still `architect`.
3. **`/usage` shows ≥2 tiers in a single run.** A run that exercises Scott + Pam shows `worker` and `architect` (or `scout` + `architect` after `/tier scout`) in `totals.by_tier`.
4. **`/model` pins all roles.** After `/model <id>`, all three role rows have matching `model_override`.
5. **`SessionConfigChanged` event fires** with the right key/old/new on every real mutation; no-op commands emit nothing.
6. **`/tier` echo** shows the resolved model id; `/model` with missing creds surfaces a `WarningRaised` event.
7. **No `StateUpdated` references** if it was removed: `rg StateUpdated src tests` returns nothing.
8. **`just qa` green.**

---

## Risks / open questions

- **Subtle behavior change for users who relied on `/tier scout` to cheap the whole session.** D1 changes that. Mitigation: the change is documented in `CLI_DESIGN.md` and the `/tier` echo makes the new scope obvious. If feedback says users want the old blanket behavior, revisit by adding a `/tier-all` or by switching defaults — but design-wise, manager-only is correct.
- **Renderer subscription to `SessionConfigChanged` could become noisy** if every tiny config change prints a line. Keep it debug-mode-only (D3 and Step 3).
- **Future `/tier <role> <tier>`** will need `SessionConfig.role_tier_overrides`. Not built now (D2). When built, `_ensure_agent` should consult that map before falling back to persona defaults.
- **Evaluator persona** is named in the C8 roadmap text but its row is seeded by C9. C8 must not seed Dwight; just verify the tier infra accommodates him (it does — `_ensure_role_config` is generic).

---

## Out-of-scope reminders

- No new providers beyond the seven already in `PROVIDER_DEFINITIONS`.
- No per-role tier slash command.
- No HR-agent escalation logic.
- No Dwight persona seed (C9).
- No pricing data.
