# Multi-Repo A2A Runs

> **Date:** 2026-05-02 · **Status:** open
> **Related:** `docs/reference/PHILOSOPHY.md`, `docs/contracts/WORKSPACE.md` (Draft)

## Idea

A real application often spans multiple repositories — application source,
docs, infrastructure. A single agent rooted in one repo can read across
boundaries via filesystem tools, but loses scope: it carries irrelevant
context, can't easily be tier-routed per repo, and its workspace settings
(skills, MCP servers, AGENTS.md) are anchored to one repo.

Future direction: spin up an agent per repo, each rooted in its own
`<repo>/.agents/` workspace. They communicate via an **A2A server** that
each agent optionally exposes on a local port. A coordinator (or a peer)
addresses them by name and dispatches subtasks across the boundary.

## Why This Fits JAC

- The runtime is already designed as an event-bus with adapters
  (CLI, future browser, future A2A server). `PHILOSOPHY.md` explicitly
  calls out `servers/a2a/` as a planned adapter.
- Each agent can have its own workspace (per-repo `.agents/`, per-repo
  `state.db`, per-repo skills/MCPs). The Workspace contract already
  scopes state to the repo, which is the right precondition for this.
- Tier routing per-repo becomes natural: the docs-repo agent uses Scout
  for most work; the infra-repo agent uses Architect more often.

## Open Questions

- **Discovery**: how does an agent in repo A find the A2A endpoint of an
  agent in repo B? Config file, registry service, or `~/.jac/peers.json`?
- **Trust**: A2A traffic is local but cross-process; do peers
  authenticate?
- **Run lifecycle**: is each agent its own `runs` row, or is there a
  coordinating "super-run" that links cross-repo runs together?
- **Failure semantics**: peer agent dies mid-call — does the caller
  retry, escalate, or fail the whole super-run?
- **Coordination transport**: the `agent_messages` table (currently
  scoped to one run) might extend to cross-run coordination, or A2A
  might use a different transport entirely.

## Sequencing

Tracked as roadmap component **C30** — the last entry in the dependency
chain:

```
spawn_agent (C14) → agent teams (C15) → A2A server (C29) → multi-repo (C30)
```

The idea already informs the workspace contract (project state is
repo-scoped, not user-global). Designing further now would jump
several prerequisites; capture the open questions here and revisit
when C29 lands.
