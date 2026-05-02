# Hooks and Callbacks Module

> **Date:** 2026-05-02 · **Status:** open
> **Related:** `docs/contracts/TOOLS_CONTRACT.md` (approval), `lab/scripts/worker_observer.py`

## Question

We have an approval system that is effectively a *pre-tool* hook with a
fixed shape (`ToolApprovalMeta` → `ApprovalPolicy` → user prompt). We have
events emitted post-tool. What we don't have is a generic hook mechanism
that lets the harness, a workflow, or a parent agent inject deterministic
behaviour at well-known lifecycle points.

## Use Case Driving This

Master agent dispatches a Minion agent for "fix this bug". Today the
Master has to:

1. Tell the Minion to *also* run `uv run ty check` after fixing.
2. Trust the Minion to actually run it.
3. Wait for the Minion to report back.
4. Re-dispatch if verification failed.

Every step burns Master inference and assumes the Minion follows
instructions reliably. With hooks, the Master attaches a `post_run` hook
to the Minion at dispatch time:

```python
post_run_hooks = [run_shell("uv run ty check")]
```

The harness runs the hook deterministically after the Minion's loop ends.
If the hook fails, the harness re-dispatches the Minion with the failure
output as context — without involving the Master at all. Master is only
summoned when the loop converges or hits its retry limit. This collapses
N inference turns into deterministic post-processing.

## Pydantic AI Surface to Use

Pydantic AI exposes the agent execution as a graph (`agent.iter()` over
`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`). The lab
script `worker_observer.py` already uses this to inject mid-loop steering
and abort on hard limits — it's our existing prototype.

The taxonomy maps cleanly to Google ADK's callback model
(<https://adk.dev/callbacks/types-of-callbacks/>):

| ADK callback | JAC hook point | Pydantic AI seam |
|---|---|---|
| Before Agent | `pre_run` | Before `agent.iter()` enters the loop |
| After Agent | `post_run` | After `End` node, before context closes |
| Before Model | `pre_model` | At `ModelRequestNode`, before `agent_run.next()` |
| After Model | `post_model` | At `CallToolsNode`, before tools execute |
| Before Tool | `pre_tool` | Pydantic AI tool prep functions / approval layer |
| After Tool | `post_tool` | After tool function returns, before next request |

The approval system already occupies `pre_tool`. The other five points
are unfilled.

## Why a Separate Module

Hooks are a cross-cutting concern, not agent code. A `HookManager`
interface that:

- Lets the harness register built-in hooks (telemetry, cost tracking,
  audit).
- Lets users define hooks via `~/.jac/hooks/` or `<repo>/.agents/hooks/`
  (couples to the workspace contract).
- Lets a parent agent attach hooks to a spawned child via the
  `spawn_agent` toolgroup (C14) — the use case driving this idea.
- Composes naturally — multiple hooks at the same point fire in
  registration order; any can short-circuit by returning a "deny" or
  "retry" signal that the harness handles deterministically.

## Distinction From Approvals

Approvals exist to prompt a *human*. Hooks exist to run *deterministic
code* — type-checkers, linters, custom validators, telemetry pushers. The
two systems may share an event-emission layer but should remain
conceptually separate. The mistake to avoid: re-using `ApprovalPolicy` as
a generic hook system. They have different shapes and different audiences.

## Sequencing

Tracked as roadmap component **C13**. The driving use case
(`uv ty check` after Minion bug fix) requires `spawn_agent`, which
arrives at **C14** — so C13 sequences just ahead of C14, with the
graph orchestration of C10 as the prerequisite for the lifecycle
seams. Designing earlier would lock in a shape before real consumers
exist.

## Next Step

Promote the taxonomy above into a `HOOKS.md` contract under
`docs/contracts/` when C13 begins. The `worker_observer.py` script is
already a working prototype of the model-request and tool-call seams —
fold its lessons into the contract rather than re-deriving them.
