# Context Management Module

> **Date:** 2026-05-02 · **Status:** open
> **Related:** `docs/contracts/STATE_SCHEMA.md` (messages), `docs/contracts/MCP_INTEGRATION.md`

## Question

Where does the responsibility for shaping the message list passed into each
agent turn live? Right now Pydantic AI manages history per-agent
automatically; once we have multi-agent runs, long tasks, and cost pressure,
we need an explicit module that can:

- Inspect the current message list for an agent at any point.
- Decide what to keep verbatim, what to summarise, what to drop.
- Make direct LLM calls (cheap-tier) to produce summaries on demand.
- Be invoked at well-defined points in the agent loop, not as an
  afterthought.

## Why a Separate Module

Compaction is not a one-shot "cut messages when over budget" operation.
It's a recurring, configurable concern that a Worker agent will hit dozens
of times across a long task. Bolting it onto agent code per-role would
duplicate logic and tie message-shaping to model selection.

A `ContextManager` module gives us:

- A single interface (`prepare`, `compact`, `summarise_segment`) per
  concern.
- Per-role configuration — Workers compact aggressively; Architects rarely.
- Direct LLM calls for summarisation routed through the cost tracker like
  any other model call (`call_type = 'direct_llm'` in `attempts`).
- Tested independently of the agent loop.

## Pydantic AI Surface to Use

`Agent(history_processors=[...])` accepts a list of callables that
transform the message history before each model request. This is the
natural seam.

Each processor receives the current message list and returns a transformed
list. Multiple processors compose. Our `ContextManager` is the
implementation behind one or more such processors, parameterised by agent
config; the processor stays thin and the manager owns the policy.

## Decision Space

- **Trigger**: token-count threshold, message-count threshold, or
  semantic-trigger ("after every K tool calls")?
- **Strategy**: head/tail truncation, mid-section summary, segment rollup
  (compact every K messages older than the recent window), agent-fork
  (summarise → reset → reinject)?
- **Granularity**: compact per-agent, per-task, or per-run?
- **Direct-LLM cost model**: Scout-tier summariser by default; configurable
  per role.
- **Preservation**: tool-result messages tend to compact well; system
  prompts and recent assistant turns must be preserved verbatim.
- **Audit**: summarisation events should land in `attempts` (already
  supports `call_type = 'direct_llm'`) and emit a `ContextCompacted`
  event so the UI can show what happened.

## Sequencing

Tracked as roadmap component **C12**, scheduled after the Notes CLI
checkpoint (C11). The benchmark itself is too small to trigger
compaction, so building it earlier biases agent behaviour without
measurable benefit. Once multi-task runs land at C11, tasks start
running long enough to fill context — that's the natural moment.

## Next Step

Spike a minimal `history_processor` against the `worker_observer.py`
pattern in `lab/scripts/`, exercising token-threshold compaction on a
synthetic long conversation. Measure: does compaction actually preserve
task progress, or does the agent lose the thread? That evidence drives
the contract.
