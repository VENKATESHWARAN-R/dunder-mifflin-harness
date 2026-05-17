# Lab

Free-form workspace for experiments, brainstorms, and reference projects. Nothing here ships with the JAC core install — the `lab` dependency group covers extras (`langgraph`, `google-genai`, `langchain-openai`, etc.) that lab scripts need.

## Layout

```
lab/
├── brainstorm/     # rough notes, half-formed ideas, design spikes
├── scripts/        # runnable Python spikes
├── notebooks/      # Jupyter explorations
└── specimens/      # standalone reference projects, kept as inspiration
```

## How to use

- New idea you don't want to lose → drop a markdown file in `brainstorm/`.
- Spike that exercises a library → script in `scripts/` or notebook in `notebooks/`.
- Idea matures into a real design choice → promote to `docs/reference/` or `docs/contracts/` and update [`docs/README.md`](../docs/README.md). Then either delete the lab artifact or leave a one-line pointer to where it landed.

## Index

### Scripts

| File | Status | What it shows |
|---|---|---|
| [scripts/pydantic_ai_script.py](scripts/pydantic_ai_script.py) | Active | Minimal Pydantic AI tool-calling agent with Langfuse instrumentation. Reference for tool registration, dependency injection, structured output. Run: `uv run python lab/scripts/pydantic_ai_script.py` |
| [scripts/worker_observer.py](scripts/worker_observer.py) | Active · uses private API | `agent.iter()` over `CallToolsNode` / `ModelRequestNode` to inject mid-loop guidance and abort on hard limits. Informs future Worker supervision and Architect escalation. Caveat: imports from `pydantic_ai._agent_graph`. |

### Notebooks

| File | Status | What it shows |
|---|---|---|
| [notebooks/pydantic-ai.ipynb](notebooks/pydantic-ai.ipynb) | Active | Pydantic AI execution results, tool call logging, structured output exploration. |

### Brainstorm

Use date prefixes (`2026-05-02-routing-heuristics.md`) so chronology is recoverable.

| File | What it captures |
|---|---|
| [brainstorm/2026-05-02-context-management-module.md](brainstorm/2026-05-02-context-management-module.md) | Decision space for compaction / `history_processor` module. Roadmap **C12**. |
| [brainstorm/2026-05-02-hooks-and-callbacks-module.md](brainstorm/2026-05-02-hooks-and-callbacks-module.md) | Hooks taxonomy mapped from ADK callbacks to Pydantic AI graph seams. Roadmap **C13**. |
| [brainstorm/2026-05-02-multi-repo-a2a-runs.md](brainstorm/2026-05-02-multi-repo-a2a-runs.md) | Per-repo agents communicating via A2A. Roadmap **C30**. |
| [brainstorm/2026-05-04-manager-specialist-minion-pattern.md](brainstorm/2026-05-04-manager-specialist-minion-pattern.md) | _Superseded by 2026-05-05._ Original manager-specialist-minion sketch with analyst-Pam, Date Mike, and Holly. |
| [brainstorm/2026-05-05-multi-agent-cast-final.md](brainstorm/2026-05-05-multi-agent-cast-final.md) | Final v0 cast: Scott + Pam (planner) + Jim + Dwight + universal `spawn_minion`. Drops Holly and analyst-Pam. Rewrites **C6b**, adds **C6c**, refines **C9**/**C12**, supersedes **C14**. |
| [brainstorm/2026-05-08-pydantic-ai-harness-vs-jac.md](brainstorm/2026-05-08-pydantic-ai-harness-vs-jac.md) | `pydantic/pydantic-ai-harness` overlap analysis. Library vs application distinction, JAC's unique points, what to adopt vs build, CodeMode relevance. Pre-M1 positioning. |

### Specimens

| Project | Purpose |
|---|---|
| [specimens/pygemini](specimens/pygemini) | PyGeminiCLI — standalone Gemini CLI clone built as a learning exercise. Inspiration only. Not imported by JAC. |

> Specimens are workspace members of the root `pyproject.toml` — they keep their own dependency lists and run independently. Don't import from them into `src/jac/`.

## Running lab work

```bash
just sync                              # installs dev + lab groups by default
uv run python lab/scripts/<file>.py
```

If you want a leaner install without lab deps, use `uv sync --no-group lab`.
