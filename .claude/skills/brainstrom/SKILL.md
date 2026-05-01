---
name: brainstrom
description: Evaluates rough technical ideas and turns promising docs, web, or Pydantic AI findings into indexed lab experiments for this harness. Use when the user says brainstorm/brainstrom, wants to explore alternatives, assess overkill, create a lab script, spike an idea, or preserve an experiment for future implementation.
---

# Brainstrom

## Purpose
Help the user think through speculative technical ideas before they become implementation work. The goal is to separate useful project insight from interesting-but-costly distractions, then preserve the useful parts as small runnable lab experiments that future implementation agents can find.

## Default Stance
Discussion first. Do not create scripts, indexes, dependencies, or production code until the idea has been evaluated and the user confirms that a lab experiment is worth creating.

Prefer project fit over novelty:
- Favor simple experiments that clarify one decision.
- Challenge ideas that duplicate existing behavior, add avoidable dependencies, or move against the harness architecture.
- Treat `lab/` as a learning and evidence area, not a shortcut into production.

## Required Context
When this skill is used in `dunder-mifflin-harness`, read only the context needed for the idea:

1. `CLAUDE.md` for repository direction and working rules.
2. `docs/PROJECT_PHILOSOPHY.md` before proposing new subsystems, runtime boundaries, agents, MCP integrations, workflow runners, or UI surfaces.
3. Relevant docs such as `docs/EVENT_CONTRACT.md`, `docs/STATE_SCHEMA.md`, or `docs/MCP_INTEGRATION.md` when the idea touches those contracts.
4. Existing files in `lab/scripts/` and `lab/scripts/README.md` if present, so new experiments do not duplicate old ones.

If the idea depends on Pydantic AI APIs or behavior, load the Pydantic AI agent-building skill before making implementation claims.

## Workflow

### 1. Capture the Idea
Restate the idea in concrete terms:
- What the user noticed or wants to try.
- Where it might fit in the harness.
- What decision the experiment should help answer.
- What evidence would make the idea worth keeping.

Ask at most one or two clarifying questions if the idea is too vague to evaluate.

### 2. Evaluate Fit Before Coding
Assess the idea against this rubric:
- Project alignment: Does it support the tiered agentic harness direction?
- Boundary fit: Does it belong in runtime, CLI, tools, state, docs, or only `lab/`?
- Simpler alternative: Can the same lesson be learned with a smaller change or existing API?
- Overkill risk: Does it add abstraction, dependency, orchestration, or persistence before the project needs it?
- Reuse value: Will future agents understand and reuse the result?
- Evidence quality: Can a small script prove or disprove the idea?

Give a recommendation before writing anything:
- `Discard`: interesting but not useful enough for this project.
- `Discuss more`: promising but underspecified.
- `Lab experiment`: worth capturing as a runnable script.
- `Implementation candidate`: strong enough to become a real task after a lab proof.

### 3. Get Confirmation
If the recommendation is `Lab experiment` or `Implementation candidate`, propose:
- Script filename under `lab/scripts/`.
- Short experiment goal.
- Expected run command.
- What will be added to `lab/scripts/README.md`.

Wait for the user's confirmation before creating or editing files.

### 4. Create the Lab Script
When approved, create one focused script in `lab/scripts/`.

Script requirements:
- Keep it runnable with the repo's normal Python environment unless the user approves a dependency.
- Include a top-level docstring explaining the idea, why it matters, what it demonstrates, and how it maps to the harness.
- Keep the example realistic but small. Prefer deterministic examples over live external services.
- Print or assert a clear result so the user can tell whether the experiment worked.
- Avoid importing from `specimens/` or treating lab code as production code.
- If the script uses private or unstable APIs, say so directly in the docstring and index entry.

### 5. Maintain the Lab Index
Create or update `lab/scripts/README.md` as the searchable index for experiments.

Use this entry format:

```markdown
## [script_name.py]

- Status: candidate | useful-reference | superseded | discarded
- Idea: One sentence explaining the explored idea.
- Demonstrates: The concrete behavior or API the script proves.
- Run: `uv run python lab/scripts/script_name.py`
- Project relevance: Where this could inform the harness later.
- Caveats: Dependency, private API, overkill risk, or reason not to promote.
```

Update an existing entry instead of adding duplicates when the new discussion extends an old experiment.

### 6. Close the Session
End with a short decision record:
- Recommendation: discard, keep as reference, expand the spike, or promote to implementation planning.
- Evidence produced: script path and index entry, if any.
- Open questions: only the ones that matter for future work.

Do not claim the experiment is production-ready unless it has tests, integration design, and alignment with the project docs.

## Output Style
Be direct and collaborative. Make tradeoffs visible. It is better to say "this is probably overkill for now" than to preserve every idea as code.
