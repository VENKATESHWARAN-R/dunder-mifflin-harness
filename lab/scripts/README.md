# Lab Script Index

This directory stores runnable exploratory scripts and spikes. These scripts are evidence and reference material for future implementation work; they are not production modules.

Before adding a new script, check this index for related experiments and update an existing entry when the new work extends an old idea.

## worker_observer.py

- Status: candidate
- Idea: Observe long-running Worker agent loops between graph steps instead of waiting for `agent.run()` to finish.
- Demonstrates: How `agent.iter()` can inspect `CallToolsNode` usage/tool calls, inject corrective `SystemPromptPart` guidance at `ModelRequestNode`, and abort when hard limits are reached.
- Run: `uv run python lab/scripts/worker_observer.py`
- Project relevance: Could inform V0 Worker supervision, token/step budgets, loop detection, and Architect escalation in the future workflow graph.
- Caveats: Uses private Pydantic AI internals from `pydantic_ai._agent_graph`; treat as a research spike until the API choice and integration boundary are revisited.

## pydantic_ai_script.py

- Status: candidate
- Idea: Try a minimal Pydantic AI tool-calling agent with global instrumentation enabled.
- Demonstrates: Creating an `Agent` with typed dependencies/output, registering an async tool, running the agent, and checking Langfuse client authentication before `Agent.instrument_all()`.
- Run: `uv run python lab/scripts/pydantic_ai_script.py`
- Project relevance: Useful as a small reference for tool registration, dependency injection, structured output, and observability experiments around Pydantic AI agents.
- Caveats: Depends on Langfuse configuration and a model gateway alias being available in the environment; the roulette example is illustrative rather than harness-specific.
