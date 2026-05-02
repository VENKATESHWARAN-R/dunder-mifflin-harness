# Docs Index

Authoritative documentation for JAC. Two folders, two intents:

- `contracts/` — **locked specs**. Treat these as binding. Update only via deliberate revision; bump the `Last revised` date in the file's status header when you do.
- `reference/` — **stable narrative**. Background, principles, project genesis. Less brittle than contracts, still trustworthy.

`ROADMAP.md` is the **living** document — expect it to change every weekend.

Every doc starts with a status header so you can tell at a glance where it sits:

```markdown
> **Status:** Locked · **Last revised:** YYYY-MM-DD · **Type:** ...
```

| Status | Meaning |
|---|---|
| `Locked` | Contract. Code must follow this; change requires intent. |
| `Reference` | Stable narrative. Read for context, not as a source of truth for code. |
| `Living` | Actively updated as work progresses. |
| `Draft` | In flight. Not safe to rely on yet. |

## Contracts

| Doc | Purpose |
|---|---|
| [contracts/STATE_SCHEMA.md](contracts/STATE_SCHEMA.md) | SQLite tables: runs, tasks, attempts, agent_configs, context, MCP registry. |
| [contracts/EVENT_CONTRACT.md](contracts/EVENT_CONTRACT.md) | Typed events, requests, commands between runtime and any UI surface. |
| [contracts/CLI_DESIGN.md](contracts/CLI_DESIGN.md) | Terminal adapter design — input grammar, rendering, slash commands. |
| [contracts/MCP_INTEGRATION.md](contracts/MCP_INTEGRATION.md) | How MCP servers and skills load from the DB into Pydantic AI agents. |
| [contracts/TOOLS_CONTRACT.md](contracts/TOOLS_CONTRACT.md) | Standardized agent tool interface. Required reading before adding a tool. |
| [contracts/WORKSPACE.md](contracts/WORKSPACE.md) | Workspace layout — `~/.jac/`, `<repo>/.agents/`, dotenv onboarding, AGENTS.md, file→DB seeding. |

## Reference

| Doc | Purpose |
|---|---|
| [reference/IDEA.md](reference/IDEA.md) | Project genesis, hypothesis, V0 scope. |
| [reference/PHILOSOPHY.md](reference/PHILOSOPHY.md) | Where code belongs, dependency boundaries, extension rules. |
| [reference/V0_BENCHMARK.md](reference/V0_BENCHMARK.md) | Locked V0 test case (Notes CLI). Acceptance criteria and success thresholds. |

## Living

| Doc | Purpose |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Component plan in dependency order (C0..Cn). C0 done; C1 next. |

## Promotion path

Idea originates in `lab/brainstorm/`. Once it's worth committing to:
1. Write a `Draft` doc under `contracts/` or `reference/`.
2. Iterate. When stable, flip status to `Locked` / `Reference`.
3. Add a row to this index.

Rule of thumb: if the codebase relies on a behavior described here, it belongs in `contracts/`. If it's narrative or motivation, `reference/`.
