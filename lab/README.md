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

Currently empty — drop ideas here as `.md` files. Use date prefixes (`2026-05-02-routing-heuristics.md`) so chronology is recoverable.

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
