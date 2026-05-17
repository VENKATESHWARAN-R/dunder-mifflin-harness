# 2026-05-08 — pydantic-ai-harness vs JAC: overlap analysis + strategic positioning

> **Status:** Locked brainstorm · **Date:** 2026-05-08 · **Role:** pre-M1 positioning analysis before fresh implementation begins.
>
> **Question answered:** Does `pydantic/pydantic-ai-harness` make JAC redundant? Should we contribute there instead of building JAC? Are there unique points that justify continuing?

---

## What is pydantic-ai-harness?

`pydantic-ai-harness` (https://github.com/pydantic/pydantic-ai-harness) is an **installable capability library** maintained by the Pydantic team. Their tagline: _"batteries for your Pydantic AI agent."_

You add it to your project:

```
uv add pydantic-ai-harness
```

And compose capabilities into an agent:

```python
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[MCP('...'), CodeMode()],
)
```

### What they've actually shipped (v0.2.0, April 2026)

One capability: **CodeMode** — wraps all tools into a single `run_code` call powered by their Monty sandbox, so the model can orchestrate multiple tool calls with Python code in a single round-trip instead of one model round-trip per tool call.

### What's in the pipeline (30+ active PRs)

| Category | Capability | PR |
|---|---|---|
| Tools | Filesystem (read/write/edit/search) | #177 |
| Tools | Shell (allowlists, denylists, timeouts) | #177 |
| Tools | Repo context injection (AGENTS.md/CLAUDE.md auto-load) | #175 |
| Tools | Verification loop (run tests, auto-fix) | #169 |
| Context | Sliding window, compaction, limit warnings | #191 |
| Context | Tool output management | #185 |
| Context | System reminders | #181 |
| Memory | Persistent key-value memory | #179 |
| Memory | Session persistence (save/restore state) | #176 |
| Orchestration | Sub-agents | #178 |
| Orchestration | Skills (progressive tool loading) | #183 |
| Orchestration | Planning | #180 |
| Orchestration | Task tracking | #65 (issue) |
| Orchestration | Teams + message bus | #195 (issue) |
| Safety | Input/output guardrails | #182 |
| Safety | Cost/token budgets | #182 |
| Safety | Tool access control + approval workflows | #173, #182 |
| Reliability | Stuck loop detection | #186 |
| Reliability | Tool error recovery | #171 |
| Reasoning | Adaptive reasoning (thinking effort) | #174 |

Their CLAUDE.md and AGENTS.md are currently empty — this is a very new project (31 commits as of today, v0.2.0 released 2026-04-24).

---

## The fundamental distinction

> **pydantic-ai-harness is a library. JAC is an application.**

This sentence resolves most of the comparison. They are not competing at the same level.

| Dimension | pydantic-ai-harness | JAC |
|---|---|---|
| **Type** | pip-installable library | CLI application you run |
| **How you use it** | `uv add pydantic-ai-harness` and compose capabilities | `jac chat`, `jac init`, `jac resume` |
| **Scope** | Standalone building blocks | Full system: CLI + runtime + agents + state + surfaces |
| **State** | Each capability is stateless (you compose them) | SQLite ledger: runs, messages, attempts, tasks, configs |
| **Research goal** | None — it's infrastructure | Measure cost-quality of tiered multi-agent routing |
| **Durability** | Provided by whichever capability you add | Core invariant: agents are disposable, state is durable |
| **Personas** | None — model-agnostic capability layer | 4 named YAML-loaded personas with 3-tier model resolution |
| **CLI** | None | Full CLI: chat REPL, slash commands, attachments, resume |
| **A2A surface** | None | M1: `agent.to_a2a()` adapter |
| **Deliverable** | Open-source library for others to build on | M5 comparison report proving/disproving the hypothesis |

The word "harness" means different things in each project. For them it's "capabilities/batteries you strap to an agent." For JAC it's "the rig that runs the agent under controlled, measurable conditions and records everything."

---

## Should we contribute to pydantic-ai-harness instead of building JAC?

**No.** The projects are at different abstraction levels. There is nothing JAC is building that directly substitutes for what they need in their repo.

More specifically:

- JAC's **unique value is application-level**: the hypothesis, the benchmark harness, the SQLite ledger, the 4-persona YAML system, the 3-tier model resolution, the A2A surface, the task-list-as-memory design, the CLI as a daily tool. None of this fits in a capability library.
- Contributing to pydantic-ai-harness would mean stripping out all context and delivering decontextualised building blocks. The research output (cost-quality data, M5 report) would be gone.
- Their design goal is "composable, stateless capabilities." JAC's design goal is "durable state + evidence-driven measurement." These are architecturally opposed.

**When contribution DOES make sense (post-M5):** if JAC develops a particularly clean implementation of context management, session persistence, or task tracking through repeated real-world iteration — and that implementation is genuinely better than their PR drafts — a stripped-down upstream contribution would be the right call. But that's a Phase-2 discussion, after the data exists.

---

## JAC's genuinely unique points

These are things JAC has or will have that pydantic-ai-harness is not building and has no intent to build:

1. **The cost-quality measurement hypothesis.** Does tiered model routing (architect/worker/micro tiers) achieve 3–5× cost reduction at comparable quality vs. a single-Opus baseline on the Notes CLI benchmark? Nobody else is running this experiment in this form. The M5 report is the deliverable.

2. **SQLite ledger as the ground truth.** `runs`, `messages`, `attempts` (with token counts, model used, tier, cost), `tasks` — all queryable. The M5 comparison report is generated directly from this data. pydantic-ai-harness has no persistence strategy; each capability is stateless.

3. **Task-list as context-resilient harness memory.** The `tasks` table + system reminder injection every turn is JAC's specific design decision: the task list survives compaction, process restart, and model context window overflows because it lives _outside_ the message history. This is what separates JAC from being a polished chatbot.

4. **Three-tier model resolution.** Per-run `agent_configs` override → user `settings.json` → shipped `data/model_specs.toml`. In-flight model swap via slash command. The measurement requires this; without it you can't prove tier routing changes anything.

5. **YAML persona system with layered precedence.** `data/personas/scott.yaml` (shipped) ← `~/.jac/personas/` (user) ← `<repo>/.agents/personas/` (project). Persona prompts never live as Python literals. Code never pattern-matches on role names. pydantic-ai-harness is model-agnostic; it has no persona concept.

6. **A2A peer surface.** An external peer can dispatch tasks to JAC via A2A and receive streamed events + a final result. This is how JAC-as-worker slots into multi-agent networks post-M1.

7. **The CLI as a usable tool.** The daily-driver goal (chat, plan, build, evaluate, resume) means JAC produces a real artifact regardless of the benchmark outcome. pydantic-ai-harness has no CLI.

---

## What JAC can take from pydantic-ai-harness

Three things worth watching:

### 1. CodeMode pattern (Round-trip reduction)

Their only shipped capability: wrapping all tools into a single `run_code` call where the model writes Python that orchestrates multiple tool calls. Their claim: one model round-trip replaces N. Their implementation uses the Monty sandbox.

JAC relevance: this pattern is highly relevant for the build phase of the harness (M2/M3). When Jim (builder) needs to read 5 files, write 3, and run tests, CodeMode-style batching would cut token cost significantly. This directly supports the cost-quality hypothesis.

**Action:** track their CodeMode stabilization. When M2/M3 shows round-trip overhead in the ledger, evaluate adopting it rather than rolling JAC's own equivalent.

### 2. Capabilities API as an alternative to custom middleware

JAC's current pattern: custom approval middleware composed in `agents/base.py`. Their pattern: pydantic-ai's `capabilities=[]` constructor argument.

In M2+, JAC's approval wrapper, context engineering hooks, and system reminder injection could potentially be restructured as pydantic-ai capabilities rather than custom middleware. This would make them composable and testable in isolation.

**Action:** when M1 context engineering lands and stabilizes, evaluate whether restructuring as capabilities reduces complexity or adds it. Restructure only if it's simpler.

### 3. Capability matrix as a checklist

Their 30-capability matrix is a useful inventory of what's being standardized in the pydantic-ai ecosystem. Before JAC builds any M2+ component (context management, skills, session persistence), check what their pipeline looks like. If they've shipped a clean implementation, adopting it is cheaper than building JAC's own.

Specifically:
- **Filesystem/shell tools (PR #177):** if this ships before JAC's M1 tool rebuild, consider adopting it instead of rolling JAC's own. Their tools are designed to be composable into any pydantic-ai agent — JAC qualifies.
- **Context sliding window + compaction (PR #191):** this is a Phase-2 evidence-gated item for JAC (C12 in the catalog). If they ship it before JAC's Phase-2 trigger fires, adopt rather than build.
- **Session persistence (PR #176):** JAC's SQLite-backed session persistence is more durable and queryable than whatever key-value approach they'll ship. Keep JAC's own.
- **Task tracking (#65):** JAC's task-list-as-harness-memory design is architecturally distinct (it's a system reminder, not just a tool). Don't conflate; keep JAC's own.

---

## What was decided

| Decision | Outcome |
|---|---|
| Is pydantic-ai-harness a substitute for JAC? | No. Different abstraction levels (library vs application). |
| Should JAC contribute to them instead? | Not now. Post-M5, individual capabilities (context management, session persistence) could be contributed if JAC's implementation is demonstrably better. |
| Does JAC have unique value? | Yes. The measurement hypothesis, SQLite ledger, task-list memory, tier routing, persona system, A2A surface are JAC-specific. |
| Should JAC adopt their filesystem/shell tools? | Watch PR #177. Adopt if it ships before M1 tools are rebuilt and the API fits JAC's approval middleware pattern. Otherwise build JAC's own and evaluate in M2. |
| Should JAC adopt CodeMode? | Not M1. Evaluate in M2/M3 when ledger data shows round-trip overhead. |
| `src` → `src-legacy` move? | Done. Fresh `src/jac/__init__.py` created. Version bumped to 0.5.0. |

---

## Open questions

- **PR #177 (filesystem/shell) timeline:** if it stabilizes before M1 code kicks off in earnest, JAC's tool layer could adopt it rather than re-implementing. Worth a quick check at M1 code start.
- **Approval middleware vs capabilities API:** pydantic-ai-harness approval workflow (PR #173) may shape how approval middleware should be structured in M1. Check it before writing `agents/base.py` middleware.
- **CodeMode + tier routing:** if CodeMode reduces round-trips significantly, it complicates the cost-quality measurement (fewer calls to cheaper models may not represent the same workload). The M5 methodology should account for this if CodeMode is adopted.
