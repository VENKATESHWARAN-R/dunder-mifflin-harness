# Manager–Specialist–Minion: JAC's Default Agentic Setup

> **Status:** Draft brainstorm · **Date:** 2026-05-04 · **Roadmap:** revises **C6**, adds **C6b**, refines **C9**, **C14**, **C15**, **C21**

## What this note locks in

JAC's default agentic shape is **manager → specialists → minions**, named after Dunder Mifflin's Scranton branch (per the `dunder-mifflin-play` predecessor). The user always interacts with **Michael Scott**; specialists are summoned by Scott as Pydantic AI tool-delegated sub-agents; one-off off-task work goes through **Holly Flax**, who hires a temp ("minion") with a custom prompt and toolset.

The previous C6 framing — "Planner + Builder, hardwired, every prompt goes through the loop" — is **wrong for our model**. Most prompts are chat-tier replies. Only build-shaped prompts should fan out into the planner→builder→evaluator pipeline. Scott decides; specialists execute.

## Cast (v0)

| Persona | Role | Tier | Purpose |
|---|---|---|---|
| **Michael Scott** | `manager` | Worker | Default agent. Persistent across turns. User-facing. Tool-routes everything. |
| **Pam Beesly** | `analyst` | Worker | Requirements gathering, env probing. Scott's "personal assistant"; only fires in autopilot mode. |
| **Date Mike** | `planner` | **Architect** | End-to-end implementation plans, task split, pseudo-code per task. Reasoning-heavy → highest tier. |
| **Jim Halpert** | `builder` | **Worker** | Follows the plan, writes the code, runs the shell. Plan does the thinking; builder executes. |
| **Dwight Schrute** | `evaluator` | Worker | By-the-book grading against acceptance criteria. The rivalry with Jim is thematically perfect for an evaluator. |
| **Holly Flax** | `recruiter` | Scout | Spawns minions on demand. Other agents call her, she creates the temp with custom prompt + tools. |

Deferred personas (slot reserved, not shipped in v0): **Pam Casso/Cake/Pamela** (UX/docs), **Creed Bratton** (security), **Erin Hannon** (additional QA), **Schrute-as-DBA** (DB-aware ops).

## Why this shape

1. **Stable user-facing identity.** The user always talks to Scott. Personality persists. UX consistency across modes and surfaces.
2. **Cost economics line up with tiered routing.** Scott on Worker is cheap chat-tier; Architect tokens only burn when Date Mike is actually planning. Most session traffic never reaches the upper tiers.
3. **Pydantic AI native.** Tool-based delegation is the cleanest of the multi-agent patterns Pydantic AI offers (agent-as-tool, `usage=ctx.usage` for cost rollup, `RunContext.deps` for shared state, message isolation by default).
4. **`agents/factory.py` rule preserved.** Every persona — including minions — is constructed there. Holly's tool calls into the factory; nothing else does.
5. **Holly = thematic minion factory.** Centralizes spawning at one named entity. Adds a natural budget/policy choke point. Maps minion lifecycle onto a workplace metaphor (temps from an agency).

## Flow

```
User → Scott (manager, persistent across turns)
         │
         ├─ trivial reply  (no delegation; one Worker call)
         │
         ├─ kick_off_build(brief)        ← Scott's tool; runs internally:
         │     │
         │     ├─ AUTOPILOT: Pam (analyst) probes env + synthesizes requirements
         │     │  HITL:      Scott asks the user directly (skip Pam)
         │     │
         │     ├─ Date Mike (planner)    → task list with pseudo-code
         │     ├─ Jim (builder)          → implements each task
         │     ├─ Dwight (evaluator)     → grades pass/fail
         │     │     └─ fail (code-fault) → retry Jim
         │     │     └─ fail (spec-fault) → bounce back to Pam
         │     └─ pass → done
         │
         └─ consult_holly(need, suggested_tools)    ← Scott's tool
                  └─ Holly spawns minion (single-call, depth ≤ 1)
                  └─ Minion runs once, returns result, gets discarded
```

**Key invariants:**
- Scott uses Pydantic AI **agent-as-tool delegation** to call the build pipeline. Pre-C10, the pipeline is sequential `Agent.run()` calls inside the tool. At C10 the tool's body becomes a `pydantic_graph` graph. The user-facing surface (Scott) doesn't change.
- Pam, Date Mike, Jim, Dwight are **also** independently summonable as Scott's tools (e.g., "ask Dwight to review this file" without running the full pipeline). Same agent config, two invocation contexts: graph node vs. direct tool call.
- Minions cannot spawn minions (depth ≤ 1). Their tool whitelist is bounded by the spawning agent's own tools (no privilege escalation). Their budget is inherited from the parent's remaining run budget.

## Decisions made (and why)

These were the open questions at the end of the brainstorm. All locked.

1. **Two-step planning (Pam → Date Mike), not one-step.** Pam keeps Scott's tool-routing simple: she handles all the "what does the user actually want, what's available in the environment" work, and hands a clean brief to Date Mike. Date Mike is pure structured task generation. Two LLM hops before any build, but the analyst stage is what makes autopilot from a sparse `jac "..."` prompt actually viable.

2. **HITL clarification is Scott's job, not Pam's.** In HITL chat mode, Scott himself asks the user clarifying questions (via runtime question events). Pam isn't summoned. Pam is autopilot-only. Simpler control flow, one less hop.

3. **Minions are single-call, not session-scoped.** Holly spawns, the temp runs once, returns a result, gets discarded. Matches the play repo's stateless temp-agent pattern. Session-scoped minions are a later optimization if usage shows we need them.

4. **Code uses semantic roles; persona is a flavor field.** `agent_configs.role = 'manager' | 'analyst' | 'planner' | 'builder' | 'evaluator' | 'recruiter'`. A new `persona` column carries `'Michael Scott' | 'Pam Beesly' | ...`. Code reads `role`; system prompts and event UI read `persona`. New contributors see roles immediately.

5. **`role` enum stays minimal in v0.** Only ship the six values above. Add `support | security | dba` etc. when those features actually wire.

6. **Tier choices.**
   - Manager: **Worker** (not Scout). Tool selection across many tools needs more reasoning than Scout reliably provides; bad routing is more expensive than the saved tokens.
   - Planner: **Architect**. Date Mike does the end-to-end reasoning — implementation plan, task decomposition, pseudo-code. This is where reasoning quality has the largest cost-of-error multiplier.
   - Builder: **Worker**. The plan does the thinking; Jim follows it.
   - Evaluator: **Worker**. Grading against acceptance criteria is well-scoped.
   - Analyst: **Worker**. Env probing is shell-tool-driven, not reasoning-heavy.
   - Recruiter: **Scout**. Holly's job is "match need → minion config"; cheap classification is enough.

## Schema implications (follow-up revision needed)

These are flagged here, not yet written into `STATE_SCHEMA.md`. Belongs in a separate revision pass once the brainstorm ships.

| Table | Change | Why |
|---|---|---|
| `agent_configs` | Add `persona TEXT` (nullable), `display_name TEXT` (nullable), `is_minion INTEGER NOT NULL DEFAULT 0`, `parent_role TEXT` (nullable, FK shape), `depth INTEGER NOT NULL DEFAULT 0` | Codify persona/minion taxonomy. |
| `agent_configs.role` | Enum gains: `manager | analyst | recruiter` (already has `planner | builder | evaluator | tester | reviewer`) | Six v0 roles supported. |
| `attempts` | Add `parent_attempt_id TEXT` (nullable, FK to `attempts.attempt_id`) | Reconstruct the call tree for cost rollups and audit. |
| Activation table | `context_store` activates at **C6b**, not C11 | Pam writes the requirements brief here; Date Mike reads. |

## Roadmap impact

The new shape:

- **C6 (rewritten):** Scott (manager) + Jim (builder), wired via Pydantic AI tool-delegation. Proves the manager-specialist primitive end-to-end. Single specialist for now — adding more is mechanical once the pattern is in.
- **C6b (new):** Pam (analyst) + Date Mike (planner). Autopilot mode wires Pam in front of Date Mike; HITL skips Pam (Scott asks directly). `context_store` activates here. Scott's `kick_off_build` tool now runs the full Pam → Date Mike → Jim sequence.
- **C9 (refined):** Dwight (evaluator) + retry loop + failure-type classification. Code-fault → retry Jim. Spec-fault → bounce back to Pam. Same single-task autonomy checkpoint.
- **C10:** unchanged in scope (graph orchestration), but the "graph" is now what lives inside Scott's `kick_off_build` tool. The CLI/runtime surface doesn't change.
- **C14 (refined naming):** "Holly Flax / Temp Agency" — clarifies depth ≤ 1, tool-whitelist, budget inheritance, factory-mediated.
- **C15:** unchanged. Holly's recruits can be grouped into teams.
- **C21 (HITL):** gains note — Scott handles clarification directly via question events; Pam is bypassed.

## Open follow-ups (not blockers for C6)

- **Minion budget mechanics.** Hard cap from parent's remaining run budget is the v0 plan. Need to decide where the budget tracker lives (runtime session? `attempts` aggregate?) — defer to C7 design.
- **Spec-fault vs code-fault classification.** Dwight emits one or the other. Precise rubric defers to C9.
- **Scott's tool surface.** Initial tools: `kick_off_build`, `consult_holly`, plus direct summons for each specialist (`ask_pam`, `ask_jim`, `ask_dwight`). Final list firms up during C6 implementation.
- **PHILOSOPHY.md addition.** A "Manager–Specialist–Minion" section codifying the pattern. Write after C6 ships, not before — pattern needs to survive contact with implementation first.
