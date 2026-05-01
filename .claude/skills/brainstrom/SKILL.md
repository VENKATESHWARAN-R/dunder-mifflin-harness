---
name: brainstrom
description: Thinks through speculative technical ideas for JAC, evaluates fit, and turns finalized decisions into the right artifact — a brainstorm note, a lab script, a doc revision, or an updated contract. Use when the user says brainstorm/brainstrom, wants to explore alternatives, assess overkill, review or revise contracts, redefine boundaries, spike an idea, or preserve thinking for later.
---

# Brainstrom

## Purpose
Help the user think through speculative ideas before they become implementation work. Most sessions are conversation, not code. The skill exists to:

1. Separate useful project insight from interesting-but-costly distractions.
2. Land each idea in the **right artifact**: a brainstorm note, a lab spike, a contract revision, a roadmap update, or just a discarded thought.
3. Keep the JAC docs honest — when a discussion changes a contract, update the contract; when it changes the plan, update the roadmap.

## Default Stance
Discussion first. Don't write code, scripts, or doc edits until the user confirms the landing spot.

A brainstorm session can end in several legitimate ways:
- A spoken decision and nothing written.
- A short note in `lab/brainstorm/`.
- A change to a `Locked` contract in `docs/contracts/`.
- A new `Draft` contract or reference doc.
- A line added to `docs/ROADMAP.md`.
- A runnable spike in `lab/scripts/` or `lab/notebooks/`.
- An "implementation candidate" handoff to a real task.

Many sessions need none of these. That's fine.

Prefer project fit over novelty:
- Favor simple experiments and small contract revisions over speculative new components.
- Challenge ideas that duplicate existing behavior, add avoidable dependencies, or move against the harness architecture.
- Treat `lab/` as a learning and evidence area, not a shortcut into production.

## Project Reality Check
Before recommending any structural change, remember where JAC actually is:

- **Pre-database.** SQLite is the planned state store, but it isn't wired in v0. If an idea touches "state", check whether a plain `dict`, `dataclass`, or in-memory module-level structure is enough for now. Premature persistence is overkill.
- **Pre-graph.** `pydantic_graph` arrives at W8 (see `docs/ROADMAP.md`). Workflow ideas before then should stay as design notes, not code.
- **One workflow only (V0).** Feature-by-feature. New workflow ideas (TDD, POC-swarm, etc.) belong in `lab/brainstorm/` until V0 lands.
- **MCP-first tools, but local-only in V0.** External MCP servers can be discussed but shouldn't be wired yet.

If the user proposes a "new component", first ask whether the existing component boundaries (CLI / runtime / tools / state) already cover it. Often the answer is yes.

## Required Context
Read only what the idea actually touches:

1. `CLAUDE.md` — repository direction, working rules, doc status convention.
2. `docs/reference/PHILOSOPHY.md` — before proposing new subsystems, runtime boundaries, agents, MCP integrations, workflow runners, or UI surfaces.
3. Relevant `docs/contracts/*.md` files when the idea touches them. Always check the status header (`Locked` vs `Draft`) before treating a doc as binding.
4. `docs/ROADMAP.md` — to know which slice we're in and what's already planned.
5. `lab/README.md` and existing `lab/brainstorm/`, `lab/scripts/`, `lab/notebooks/` entries — so new work doesn't duplicate old.

If the idea depends on Pydantic AI APIs or behavior, load the Pydantic AI agent-building skill before making implementation claims.

## Workflow

### 1. Capture the Idea
Restate the idea in concrete terms:
- What the user noticed or wants to try.
- Where it might fit in JAC (CLI, runtime, tools, state, docs, lab, or nowhere).
- What decision the discussion should help answer.
- What would change in the project if the idea was accepted.

Ask at most one or two clarifying questions if the idea is too vague to evaluate.

### 2. Evaluate Fit
Assess against this rubric:
- **Project alignment**: Does it support the tiered agentic harness direction?
- **Boundary fit**: Does it belong in runtime, CLI, tools, state, docs, or only `lab/`?
- **Existing-contract impact**: Does it require revising `docs/contracts/STATE_SCHEMA.md`, `EVENT_CONTRACT.md`, `TOOLS_CONTRACT.md`, `MCP_INTEGRATION.md`, or `CLI_DESIGN.md`? If yes, name the specific section.
- **Simpler alternative**: Can the same lesson be learned with a smaller change, an in-memory `dict`/`dataclass`, or an existing API?
- **Overkill risk**: Does it add abstraction, dependency, orchestration, or persistence before the project needs it?
- **Roadmap fit**: Which slice does this belong to? Is it ahead of where we are?
- **Evidence quality**: Does this need a runnable spike to prove, or is reasoning enough?

### 3. Recommend a Landing Spot
Pick **one** outcome and propose it before writing anything:

| Recommendation | Artifact | When |
|---|---|---|
| `Discard` | (nothing) | Interesting but not useful for JAC. |
| `Discuss more` | (nothing yet) | Promising but underspecified. Ask what's missing. |
| `Brainstorm note` | `lab/brainstorm/YYYY-MM-DD-<slug>.md` | Worth remembering, not yet actionable. |
| `Lab spike` | `lab/scripts/<name>.py` or `lab/notebooks/<name>.ipynb` | Need runnable evidence to decide. |
| `Contract revision` | edit existing `docs/contracts/*.md` | Idea changes a locked spec. Bump `Last revised`. |
| `New draft doc` | new `docs/contracts/*.md` (status `Draft`) or `docs/reference/*.md` | New surface area or new design narrative. Add row to `docs/README.md`. |
| `Roadmap update` | edit `docs/ROADMAP.md` | Idea changes scope or order of an existing slice, or adds one. |
| `Implementation candidate` | (handoff) | Strong enough to become a real task after evidence is in. |

A single session can produce more than one outcome (e.g. a spike **plus** a STATE_SCHEMA revision). Be explicit about each.

### 4. Get Confirmation
For any artifact, propose its exact shape before creating or editing:
- File path.
- One-line purpose.
- Which section of which doc gets touched (for revisions).
- Status header values (for new docs).

Wait for the user's confirmation.

### 5. Apply the Change
Once approved:

**For brainstorm notes** (`lab/brainstorm/`):
- Filename: `YYYY-MM-DD-<slug>.md`. Date prefix keeps chronology recoverable.
- Include: the question being explored, the current thinking, the open decision points, and any links to relevant docs.
- Add a row to `lab/README.md` under the "Brainstorm" section.

**For lab spikes** (`lab/scripts/` or `lab/notebooks/`):
- Top-level docstring: idea, why it matters, what it demonstrates, how it maps to JAC.
- Realistic but small. Deterministic over live services where possible.
- Print or assert a clear result.
- Don't import from `lab/specimens/`. Don't treat lab code as production.
- If the spike uses private or unstable APIs, say so in the docstring and the index row.
- Add a row to `lab/README.md` under "Scripts" or "Notebooks" with this format:
  ```markdown
  | [scripts/<name>.py](scripts/<name>.py) | Active · <caveat if any> | <one-line what it shows> |
  ```

**For contract revisions** (`docs/contracts/*.md`):
- Make the smallest change that captures the decision. Don't rewrite for style.
- Bump `Last revised:` in the status header to today.
- If the revision affects code that already exists, flag it so the user knows what needs to follow.
- If the contract is `STATE_SCHEMA.md` and the change is about how state is *represented* (not yet persisted), note explicitly that the v0 implementation can use in-memory `dict`/`dataclass` until the database is wired — the schema is the future shape, not a current requirement.

**For new draft docs**:
- Use status header: `> **Status:** Draft · **Last revised:** YYYY-MM-DD · **Type:** ...`.
- Add a row to `docs/README.md` in the appropriate section.
- A `Draft` doc is not binding. Code shouldn't depend on it until it flips to `Locked` or `Reference`.

**For roadmap updates** (`docs/ROADMAP.md`):
- Bump `Last revised:` in the status header.
- If the change reorders slices, say why in one line.
- If it adds work to a future slice, slot it under that slice — don't pull it forward into the current week unless the user agrees.

### 6. Close the Session
End with a short decision record:
- Recommendation(s) and where each landed (file path or "verbal decision only").
- What other docs got touched as a downstream effect (contract revision → roadmap line → new draft).
- Open questions: only the ones that matter for future work.
- Suggest the next concrete step if there is one (e.g. "draft a STATE_SCHEMA revision once the in-memory shape settles").

Do not claim an idea is production-ready unless it has tests, integration design, and alignment with the relevant `Locked` contracts.

## Output Style
Be direct and collaborative. Make tradeoffs visible. It is better to say "this is probably overkill for now" or "this needs a STATE_SCHEMA revision before any code" than to silently agree and create files.

When you suggest a contract revision, quote the exact section that needs to change and what it would say after — don't paraphrase.
