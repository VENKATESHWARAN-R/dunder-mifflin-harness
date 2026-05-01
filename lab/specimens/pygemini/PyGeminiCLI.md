# PyGeminiCLI — Technical Reference

PyGeminiCLI is a terminal-based AI coding assistant that connects your shell to Google's Gemini API. You type in a `>>>` prompt; it reads your files, edits code, runs commands, searches the web, and streams back answers — all driven by a ReAct loop that keeps calling the model until it has nothing left to do.

This document is a technical reference for the codebase. It explains what every major component does, how the pieces connect, and why the design is shaped the way it is.

---

## Table of Contents

1. [The Big Picture: What Happens When You Type a Prompt](#1-the-big-picture)
2. [The ReAct Loop — The Heart of Every AI Agent](#2-the-react-loop)
3. [The Tool System — How Agents Interact with the World](#3-the-tool-system)
4. [The Event System — Decoupling UI from Logic](#4-the-event-system)
5. [Context Management — How Agents Stay Smart](#5-context-management)
6. [Sub-Agent Orchestration — Spawning and Managing Workers](#6-sub-agent-orchestration)
7. [Session and State Persistence — Surviving Across Conversations](#7-session-and-state-persistence)
8. [Safety and Control — Keeping Agents on a Leash](#8-safety-and-control)
9. [Practical Guide: Building This Yourself](#9-practical-guide)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. The Big Picture

When you type a message into PyGeminiCLI, here is everything that happens, in order:

```
User types: "refactor auth.py to use dataclasses"
                         |
                         v
              ┌─────────────────────┐
              │   InputHandler      │  Reads terminal input (readline, history)
              │   cli/input.py      │
              └─────────────────────┘
                         |
                         v
              ┌─────────────────────┐
              │   App.run() REPL    │  Outer while-True loop, slash command check
              │   cli/app.py        │
              └─────────────────────┘
                         |
                         v
              ┌─────────────────────┐
              │   AgentLoop.run()   │  THE REACT LOOP — inner while loop
              │   core/agent_loop.py│
              └─────────────────────┘
                    /         \
                   /           \
                  v             v
    ┌──────────────────┐  ┌──────────────────┐
    │ ContentGenerator │  │  ToolRegistry    │
    │ (calls Gemini    │  │  (dispatches     │
    │  streaming API)  │  │   tool calls)    │
    └──────────────────┘  └──────────────────┘
           |                      |
           v                      v
    ┌──────────────────┐  ┌──────────────────┐
    │ ConversationHist │  │  BaseTool.execute│
    │ (stores all      │  │  (does the real  │
    │  messages)       │  │   work)          │
    └──────────────────┘  └──────────────────┘
                   \           /
                    \         /
                     v       v
              ┌─────────────────────┐
              │   EventEmitter      │  Streams text & events to CLI without
              │   core/events.py    │  core knowing about the terminal
              └─────────────────────┘
                         |
                         v
              ┌─────────────────────┐
              │   Renderer          │  Displays Rich-formatted output
              │   cli/renderer.py   │
              └─────────────────────┘
                         |
                         v
              Terminal: model response streams in, tool outputs appear,
                        confirmation prompts block as needed
```

### The Two Loops

There are two nested loops, and understanding that they serve different purposes is the first key insight:

**Outer loop** (`App.run()` in `cli/app.py`, line 111): The REPL. Reads one user input, dispatches it, waits for the turn to finish, then reads the next input. This loop runs for the lifetime of the session.

**Inner loop** (`AgentLoop.run()` in `core/agent_loop.py`, line 98): The ReAct loop. Given a single user message, it keeps calling the model and executing tools until the model produces a text-only response. This loop runs for the duration of one *turn*.

The outer loop is trivial. The inner loop is everything.

---

## 2. The ReAct Loop — The Heart of Every AI Agent

### What ReAct Is

ReAct stands for **Reason + Act**. The pattern, introduced in a 2022 paper, interleaves model reasoning (text generation) with grounded actions (tool calls). The insight is that a model that can both think out loud *and* take actions is dramatically more capable than one that can only generate text.

The loop looks like this conceptually:

```
while True:
    response = model.generate(history + tools)

    if response.has_only_text:
        display(response.text)
        break                          # done!

    if response.has_tool_calls:
        display(response.text)         # partial reasoning shown to user
        for call in response.tool_calls:
            result = execute(call)
            history.append(result)
        # loop continues — model sees results and reasons further
```

The model controls when the loop ends. Your code never says "okay, we're done with tool use now." The model decides that by producing a response without any tool calls. This is the key insight that many people miss when they first encounter agentic systems.

### Our Implementation, Annotated

The entire ReAct loop lives in `src/pygemini/core/agent_loop.py`. Here it is with annotations explaining every decision:

```python
async def run(self, user_message: str) -> None:
    self._abort_signal.clear()              # reset from any previous Ctrl+C

    # Step 1: The user's message joins the conversation.
    # This is what the model sees as "what the user asked."
    self._history.add_user_message(user_message)

    # Step 2: System prompt is rebuilt every turn.
    # This lets context (GEMINI.md, memories) be injected freshly.
    tool_names = [t.name for t in self._tool_registry.get_all()]
    system_prompt = build_system_prompt(tool_names=tool_names)

    # Step 3: THE REACT LOOP.
    # This while loop is the entire intelligence of the agent.
    while not self._abort_signal.is_set():

        # 3a+3b: Build the tool declaration list.
        # This tells the model what tools exist and how to call them.
        # Sent with EVERY request so the model always knows its options.
        declarations = self._tool_registry.get_filtered_declarations()
        tools = [{"function_declarations": declarations}] if declarations else []

        # 3c: Stream the model's response.
        # The model may emit text, function calls, or both in one response.
        text_parts = []
        function_calls = []
        model_parts = []

        async for chunk in self._content_generator.generate_stream(
            self._history.get_messages(), tools, system_prompt
        ):
            # 3d: Each chunk is either text or a function call request.
            if chunk.text is not None:
                await self._event_emitter.emit(      # stream text to terminal
                    CoreEvent.STREAM_TEXT,
                    StreamTextEvent(text=chunk.text),
                )
                text_parts.append(chunk.text)

            if chunk.function_calls:
                function_calls.extend(chunk.function_calls)

        # 3e: Add the model's response to history.
        # The model's exact words (and tool call requests) are preserved.
        if text_parts:
            model_parts.append(types.Part.from_text(text="".join(text_parts)))
        for fc in function_calls:
            model_parts.append(types.Part(function_call=...))
        if model_parts:
            self._history.add_model_response(model_parts)

        # 3f: THE EXIT CONDITION.
        # If the model didn't ask to use any tools, we're done.
        if not function_calls:
            break

        # 3g: Execute all the tools the model requested.
        function_responses = []
        for fc in function_calls:
            response = await self._execute_function_call(fc)
            function_responses.append(response)

        # 3h: Tool results go back into history as a "user" turn.
        # On the next loop iteration, the model sees these results
        # and continues reasoning.
        self._history.add_tool_results(function_calls, function_responses)
        # ^ loop continues ^
```

**Why tool results are "user" turns:** The Gemini API (like most LLM APIs) uses a strict alternating user/model turn structure. When the model calls a tool, that's recorded as a model turn. The results of those tool calls are then wrapped in a "user" turn to hand back to the model. See `history.py` lines 58-103. This is an API convention, not a fundamental limitation — it means function responses appear as if the user is reporting results back to the model.

### How Streaming Works

The `ContentGenerator` (`src/pygemini/core/content_generator.py`) wraps the google-genai SDK's streaming endpoint. The SDK returns a sync iterator; we run it on the event loop by wrapping it in an `async for`:

```python
async def _stream(self, model, history, config) -> AsyncIterator[StreamChunk]:
    response = self._client.models.generate_content_stream(
        model=model,
        contents=history,
        config=config,
    )
    for server_chunk in response:          # sync iterator from SDK
        parts = server_chunk.candidates[0].content.parts
        for part in parts:
            if part.text:
                yield StreamChunk(text=part.text)
            elif part.function_call:
                fn_calls.append(FunctionCallData(...))
        if fn_calls:
            yield StreamChunk(function_calls=fn_calls)
```

A single streaming response can contain both text and function calls. The model might reason out loud ("I'll need to read the file first...") and then immediately emit a `read_file` tool call, all in the same stream. Our `StreamChunk` dataclass handles both cases:

```python
@dataclass(frozen=True, slots=True)
class StreamChunk:
    text: str | None = None
    function_calls: list[FunctionCallData] | None = None
```

Text chunks go directly to the renderer as they arrive (real-time streaming). Function call chunks are collected and processed after the stream completes.

### Framework Equivalents

**LangGraph:** Our `while not abort_signal.is_set()` loop is LangGraph's graph execution engine. In LangGraph, you define a `StateGraph` with nodes (agent, tools) and edges (conditional routing based on whether tool calls were made). The graph's built-in cycle detection and routing replaces our explicit while loop.

```python
# LangGraph equivalent of our ReAct loop
from langgraph.graph import StateGraph, MessagesState

def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: MessagesState):
    # LangGraph's ToolNode handles this for you
    ...

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # our function_calls branch
    return END              # our `if not function_calls: break`

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")   # tool results go back to agent
```

Our history = LangGraph's `MessagesState`. Our while loop with the break condition = LangGraph's conditional edges. The logic is identical; LangGraph just gives it a graph-theory framing.

**Pydantic AI:** Their `Agent.run()` method contains this same loop internally. You don't see it, but `run()` keeps calling the model and executing registered tools until the model stops requesting tools. Their API is higher-level — you register tools with `@agent.tool` and let the framework handle the loop. The tradeoff: less control, less boilerplate.

```python
# Pydantic AI equivalent
from pydantic_ai import Agent

agent = Agent('gemini-2.5-flash', system_prompt="...")

@agent.tool
async def read_file(ctx: RunContext, path: str) -> str:
    return Path(path).read_text()

result = await agent.run("refactor auth.py")
# The loop is hidden inside agent.run()
```

**Google ADK:** Their `Runner` class with an `LlmAgent` is the direct equivalent. `Runner.run_async()` drives the same generate → tools → generate cycle. ADK also exposes `SequentialAgent` and `ParallelAgent` for composing multiple agents, which is what our TASKS.md sub-agent pattern approximates manually.

**CrewAI:** Their `Agent.execute_task()` calls an LLM, parses tool requests from the text output (when using text-based tool parsing), executes them, and loops. Older versions used ReAct prompting to get the model to emit structured tool calls as text; newer versions use native function calling, which is what we do directly.

---

## 3. The Tool System — How Agents Interact with the World

Every capability the agent has beyond text generation is a tool. Understanding the tool contract deeply — what a tool must declare, how it's discovered, how it's called — is essential because this pattern appears in every agentic framework.

### The BaseTool Contract

Every tool in PyGeminiCLI extends `BaseTool` (`src/pygemini/tools/base.py`). The contract has four parts:

```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # How the model refers to this tool

    @property
    @abstractmethod
    def description(self) -> str: ...   # Prompt the model uses to decide WHEN to call it

    @property
    @abstractmethod
    def parameter_schema(self) -> dict: ...  # JSON Schema: what args the model passes

    def validate_params(self, params: dict) -> str | None: ...  # Pre-execution guard
    def should_confirm(self, params: dict) -> ToolConfirmation | None: ...  # Ask user?

    @abstractmethod
    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult: ...                # Do the actual work
```

This maps to how function calling works in the Gemini API. The `to_function_declaration()` method on `BaseTool` serializes the name, description, and parameter schema into the dict format the API expects:

```python
def to_function_declaration(self) -> dict:
    return {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameter_schema,
    }
```

The model uses `description` to decide *whether* to call the tool and `parameter_schema` to decide *what to pass*. Writing good descriptions is where most tool authors fail — it's prompt engineering for tool selection.

### The Validation → Confirmation → Execution Pipeline

Inside `AgentLoop._execute_function_call()` (lines 182-241), every tool call goes through three gates before execution:

```
Model requests tool call
         |
         v
  1. Lookup: Does this tool exist?
         |
         v (yes)
  2. Validate: Are the params structurally valid?
         |
         v (yes)
  3. Confirm: Does this tool require user approval?
         |  \
         |   \-- Yes: show prompt, wait for user input, abort if denied
         v
  4. Execute: Run the tool, catch exceptions
         |
         v
  5. Emit result to CLI display (display_content)
         |
         v
  6. Return result to model (llm_content)
```

This pipeline is explicit and ordered. Each gate can short-circuit the rest. Validation failures return an error to the model so it can try again with corrected params. Confirmation denials return "User denied execution" to the model, which typically causes the model to explain what it was going to do and ask how to proceed differently.

### The Dual-Output Pattern

The most subtle design decision in the tool system is `ToolResult`'s two content fields:

```python
@dataclass
class ToolResult:
    llm_content: str    # What the model sees — structured, complete, parseable
    display_content: str  # What the user sees — Rich markup, formatted, truncated
    is_error: bool = False
```

These serve fundamentally different audiences. The `EditFileTool` example makes this clear:

- `llm_content`: `"Edited /path/to/auth.py: replaced 45 chars with 82 chars\n--- auth.py\n+++ auth.py\n@@ -12,3 ..."`  — the model gets the full diff so it can verify its edit worked
- `display_content`: `"[bold]--- auth.py[/bold]\n[cyan]@@ -12,3 @@[/cyan]\n[red]-    def old_method[/red]\n[green]+    def new_method[/green]"` — the user gets colored Rich markup

Never conflate these. The model doesn't need Rich markup (it's noise). The user doesn't need machine-readable structured output (it's clutter). The separation is worth the extra field every time.

### Tool Registration and Discovery

`ToolRegistry` (`src/pygemini/tools/registry.py`) is a plain dictionary with extra methods. Registration happens in `App.__init__()` via `register_defaults()`:

```python
def register_defaults(self, event_emitter=None, memory_store=None) -> None:
    from pygemini.tools.filesystem import get_filesystem_tools
    for tool in get_filesystem_tools():
        self.register(tool)

    from pygemini.tools.shell import ShellTool
    self.register(ShellTool())
    # ...
```

The lazy imports here are intentional: they avoid circular import issues and keep startup fast. Tools are only imported when the registry is initialized.

The `get_filtered_declarations()` method (`registry.py` lines 80-107) applies three layers of filtering before sending the declaration list to the model: a core tools allowlist, an excluded tools denylist, and a per-call exclusion set. This allows you to expose different tool subsets in different contexts — useful for sandboxing, for restricting tools to relevant ones for a task, or for running the agent in read-only mode.

### Framework Equivalents

**LangGraph:** The `@tool` decorator from `langchain_core.tools` is the equivalent of our `BaseTool`. It auto-generates the JSON Schema from Python type hints and docstrings.

```python
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read a file and return its contents. Use for reading source code."""
    return Path(path).read_text()
```

The framework wires this into function calling for you. The tradeoff versus our explicit `BaseTool`: you lose explicit `validate_params`, `should_confirm`, and the dual-output pattern. You can add these back, but they're not built-in.

**Pydantic AI:** Their `@agent.tool` decorator is similar but adds `RunContext` for dependency injection. Their schema generation is Pydantic-native — you define params as a Pydantic model or use type annotations. The framework handles the serialization.

**Schema generation:** Our tools use hand-written JSON Schema (`parameter_schema` property). LangGraph and Pydantic AI generate schemas from type annotations automatically. Both approaches work; hand-written schemas give you explicit control over descriptions per-parameter (the part that influences the model's behavior most), while auto-generated schemas reduce boilerplate at the cost of some control.

---

## 4. The Event System — Decoupling UI from Logic

One of the cleanest design decisions in this codebase is the strict rule: **core never imports from CLI**. The `core/` package has no knowledge that a terminal exists. This is enforced architecturally: any import of `pygemini.cli.*` from `pygemini.core.*` is a bug.

The mechanism that makes this possible is the `EventEmitter` in `src/pygemini/core/events.py`.

### Why This Matters

The core layer emits events. The CLI layer listens to events. The core has no idea what happens on the other side of an event emission — it could be a terminal renderer, a JSON-over-HTTP adapter, a test mock, or nothing at all. This means:

- **Headless mode** (`App.run_headless()` in `cli/app.py` line 148) requires zero changes to the core
- **Integration tests** can swap the renderer for a mock that records events
- **A future web UI** would subscribe to the same events with different handlers

### The EventEmitter

The emitter is a simple pub/sub system with async support:

```python
class EventEmitter:
    def __init__(self):
        self._listeners: dict[CoreEvent, list[Callable]] = {
            event: [] for event in CoreEvent
        }
        self._confirmation_queue: asyncio.Queue[bool] | None = None

    def on(self, event: CoreEvent, callback: Callable) -> None:
        self._listeners[event].append(callback)

    async def emit(self, event: CoreEvent, data: Any = None) -> None:
        for callback in self._listeners[event]:
            result = callback(data)
            if asyncio.iscoroutine(result):
                await result
```

Both sync and async callbacks are supported. Async callbacks are awaited. This means the rendering layer can do async I/O (e.g., writing to a streaming HTTP response) without the core knowing.

The event types are:

```python
class CoreEvent(Enum):
    STREAM_TEXT = "stream_text"       # A chunk of model text arrived
    TOOL_EXECUTING = "tool_executing" # A tool is about to run
    TOOL_OUTPUT = "tool_output"       # A tool finished
    CONFIRM_REQUEST = "confirm_request"  # Need user approval
    ERROR = "error"                   # Something went wrong
    TURN_COMPLETE = "turn_complete"   # The agent loop finished
```

### The Confirmation Handshake — The Cleverest Pattern

The most interesting piece is the `request_confirmation` / `respond_confirmation` pair. This implements a **blocking handshake** across the core/CLI boundary using an `asyncio.Queue`:

```python
# In EventEmitter (core/events.py lines 139-167)

async def request_confirmation(self, description: str, details: dict) -> bool:
    # Create a single-slot queue that will carry the user's answer
    self._confirmation_queue = asyncio.Queue(maxsize=1)
    # Tell the CLI "please ask the user this question"
    await self.emit(CoreEvent.CONFIRM_REQUEST, ConfirmRequestEvent(...))
    # Block here until the CLI puts an answer in the queue
    approved = await self._confirmation_queue.get()
    self._confirmation_queue = None
    return approved

def respond_confirmation(self, approved: bool) -> None:
    # Called by the CLI after the user answers yes/no
    self._confirmation_queue.put_nowait(approved)
```

And in `App._wire_confirmation()` (`cli/app.py` lines 61-68):

```python
def _wire_confirmation(self) -> None:
    async def on_confirm(event: ConfirmRequestEvent) -> None:
        approved = self._renderer.render_confirmation(event)  # shows prompt, gets y/n
        self._event_emitter.respond_confirmation(approved)    # unblocks the queue

    self._event_emitter.on(CoreEvent.CONFIRM_REQUEST, on_confirm)
```

The flow:
1. Core calls `request_confirmation()`, emits `CONFIRM_REQUEST`, then `await`s the queue
2. The `on_confirm` listener fires (it's registered by the CLI layer)
3. Renderer shows the user the confirmation dialog
4. User says yes or no
5. `respond_confirmation()` puts the bool into the queue
6. Core's `await` unblocks, gets the answer, continues execution

This is async cooperative multitasking at its most elegant. The core loop is suspended during user input with no busy-waiting, no threads, no callbacks-on-callbacks. The queue is the only shared state, and it's bounded to one item by `maxsize=1`.

### Framework Equivalents

**LangGraph:** LangGraph's `interrupt()` function implements something similar — it halts graph execution and waits for human-in-the-loop input. Their streaming API with `stream_mode="updates"` is the equivalent of our event emission for streaming text.

**Pydantic AI:** Has message streaming via `agent.run_stream()`. Human-in-the-loop confirmation is not built-in — you'd implement it at the application layer using the same queue pattern we use.

**The general lesson:** Frameworks often handle streaming text well but are weak on the confirmation handshake. Our explicit queue-based approach is portable to any framework — if you need human-in-the-loop confirmation in LangGraph, Pydantic AI, or a custom system, this queue pattern is the right way to do it.

---

## 5. Context Management — How Agents Stay Smart

An agent's "intelligence" is largely determined by what it can see. Context management is the art of keeping the right information in the model's context window.

### Conversation History

`ConversationHistory` (`src/pygemini/core/history.py`) is an ordered list of `types.Content` objects. Every user message, every model response (including tool call requests), and every tool result gets appended to this list. The entire list is sent to the model on every API call.

The API conversation structure for a tool use sequence looks like:

```
[
    Content(role="user",  parts=[Part(text="refactor auth.py")]),
    Content(role="model", parts=[Part(text="I'll read the file first"),
                                  Part(function_call=FunctionCall(name="read_file", ...))]),
    Content(role="user",  parts=[Part(function_response=FunctionResponse(name="read_file", ...))]),
    Content(role="model", parts=[Part(text="Here's what I see: ..."),
                                  Part(function_call=FunctionCall(name="edit_file", ...))]),
    Content(role="user",  parts=[Part(function_response=FunctionResponse(name="edit_file", ...))]),
    Content(role="model", parts=[Part(text="Done. I made the following changes...")])
]
```

Notice the alternating user/model pattern. Tool responses appear as "user" turns — this is the Gemini API convention (`history.py` lines 73-103). The model wrote a function call as part of its response; the function response comes back as if the user is reporting the result.

### The Token Problem

Every API call sends the full history. As conversations grow, token costs grow and eventually you hit the context limit. The `token_count` property on `ConversationHistory` tracks an estimate:

```python
@property
def token_count(self) -> int:
    total_chars = 0
    for message in self._messages:
        for part in message.parts:
            if part.text is not None:
                total_chars += len(part.text)
            if part.function_call is not None:
                total_chars += len(fc.name) + len(str(fc.args))
            if part.function_response is not None:
                total_chars += len(fr.name) + len(str(fr.response))
    return estimate_tokens("x" * total_chars)  # ~1 token per 4 chars
```

This is a rough heuristic. A more accurate approach calls the `count_tokens` API endpoint, but that adds latency. For production use, the rough estimate is usually good enough for triggering compression.

### Context Compression

When history exceeds 50 messages, `ConversationCompressor` (`src/pygemini/session/compressor.py`) summarizes the oldest messages:

```python
async def compress(self, history: ConversationHistory) -> CompressionResult:
    messages = history.get_messages()
    split_idx = len(messages) - self._keep_recent  # keep last 10
    to_compress = messages[:split_idx]

    transcript = self._format_for_summarization(to_compress)
    summary = await self._call_llm_for_summary(transcript)
    # Returns CompressionResult — caller replaces the messages with the summary
```

The compressor uses a cheaper/faster model (`config.fallback_model`) for summarization since the task requires intelligence but not the full power of the primary model. The summary prompt explicitly instructs: "Preserve all important details: decisions made, files modified, errors encountered, user preferences, and task progress."

The `history.replace_messages(start, end, replacement_messages)` method (`history.py` line 176) performs the surgical replacement. The result: 40 messages become 1 summary message, the model keeps recent context intact, and the effective context window stays manageable.

### System Prompt and Context Injection

The system prompt is rebuilt on every turn in `AgentLoop.run()` (line 94). This is intentional: it allows live injection of project context. The `build_system_prompt()` function in `core/prompts.py` combines:

- Base behavior instructions ("be concise, read before writing")
- Tool usage guidelines ("prefer edit_file for targeted changes")
- Available tool names (for the model's reference)
- Project context (GEMINI.md files, memory entries)

This last piece — project context injection — is how you make an agent "know" your codebase. Place architectural decisions, coding conventions, and project facts in a `GEMINI.md` file (or equivalent). The agent reads it on every turn and adapts its behavior accordingly.

### Persistent Memory

`MemoryStore` (`src/pygemini/context/memory_store.py`) is an append-only JSON file at `~/.pygemini/memory.json`. When the model calls the `save_memory` tool, a fact gets written to disk. On the next session, the `get_formatted()` method returns those facts for injection into the system prompt:

```python
def get_formatted(self) -> str:
    entries = self.load()
    if not entries:
        return ""
    lines = ["Remembered facts and preferences:"]
    for entry in entries:
        lines.append(f"- [{entry['timestamp']}] {entry['content']}")
    return "\n".join(lines)
```

This is the simplest possible memory system. The entire memory is injected on every turn. For small fact sets this is fine; for larger memories you'd want retrieval-augmented lookup rather than dumping everything into the prompt.

### Framework Equivalents

**LangGraph:** Uses `MemorySaver` as a checkpointer for within-session state, and Postgres/Redis store backends for cross-session persistence. Their `MessageState` with automatic trimming handles the context window problem. The compression pattern is available via `langchain_community.chat_message_histories` with summarization chains.

**Pydantic AI:** Conversation history is managed via the `message_history` parameter to `agent.run()`. You hold the history externally and pass it back each time. Cross-session persistence is your responsibility.

**The real insight:** Every framework struggles with context management in the same ways. The problems — growing token costs, context limits, preserving important facts across compressions, cross-session memory — are not solved by choosing the right framework. They're solved by careful design. Understanding how we solve them here makes you better equipped to solve them in any system.

---

## 6. Sub-Agent Orchestration — Spawning and Managing Workers

Here's something meta: this project was *itself* built using sub-agents. The development followed a structured sub-agent orchestration pattern that you can observe in `TASKS.md`.

### The Pattern in Practice

Rather than one agent doing everything sequentially, work was broken into tasks with explicit:

- **IDs** (P1-01, P1-02, etc.) for unambiguous reference
- **Dependencies** (P1-03 depends on P1-01) for ordering
- **Model assignments** (opus for complex design, sonnet for standard implementation, haiku for boilerplate) for cost/capability optimization
- **Scope boundaries** (each task is one file or one coherent chunk) for parallel execution safety

The wave-based execution pattern:
```
Wave 1: P1-01, P1-02, P1-03 (no dependencies — all in parallel)
Wave 2: P1-04, P1-05 (depend only on Wave 1 tasks — parallel among themselves)
Wave 3: P1-06 (depends on P1-04 and P1-05 — must wait)
```

This is a manually-implemented version of what `ParallelAgent` and `SequentialAgent` do in Google ADK, or what parallel branches with `Join` nodes do in LangGraph.

### Why Model Assignment Matters

Not all tasks need the same model intelligence. The cost/capability matrix for common task types:

| Task Type | Recommended Model | Reason |
|---|---|---|
| Architecture decisions, complex refactoring | Opus/Pro | High reasoning demand |
| Standard feature implementation | Sonnet/Flash | Good balance |
| Boilerplate, config files, tests | Haiku/Flash-lite | Fast, cheap, sufficient |
| Summarization | Cheapest available | Low creativity needed |

The `ConversationCompressor` applies this principle by using `config.fallback_model` for summarization. You can apply it more broadly by routing different task types to different models.

### The Progress Tracking Problem

Long-running sub-agent work across multiple sessions has a fundamental problem: *where was I?* The solution in this project is `TASKS.md` — a markdown file with checkboxes that an agent can both read and update:

```markdown
- [x] P1-01: BaseTool ABC and ToolResult dataclasses   ← completed
- [x] P1-02: ToolRegistry                              ← completed
- [ ] P2-01: MCP client integration                    ← todo
```

An agent resuming work reads `TASKS.md` to understand current state, picks the next unchecked task with satisfied dependencies, and marks it done when complete. This pattern works across sessions, survives context resets, and provides a human-readable audit trail.

### Framework Equivalents

**LangGraph:** Subgraphs (nested `StateGraph` objects) for complex sub-agents. Parallel branches with a `fanout` + `JOIN` pattern for wave-based parallelism. Their persistence layer (checkpointers) handles the "where was I?" problem automatically.

**CrewAI:** Their `Crew` with multiple `Agent` instances and a `Process.hierarchical` mode is a formalized version of this pattern. A manager agent delegates to specialist agents. Their `Task` object maps almost directly to our task IDs with dependencies.

**Google ADK:** `SequentialAgent` for ordered pipelines, `ParallelAgent` for concurrent work, `LoopAgent` for iterative refinement. These are the building blocks of the wave pattern made explicit.

---

## 7. Session and State Persistence — Surviving Across Conversations

State management is consistently the hardest part of building real agent systems. The problems compound: conversation history is lossy when compressed, tool results reference external state that may have changed, and user context from three sessions ago is still relevant.

### Session Save/Load

`SessionManager` (`src/pygemini/session/manager.py`) serializes `ConversationHistory` to JSON:

```python
def save(self, name: str, history: ConversationHistory) -> SessionInfo:
    messages = [_content_to_dict(c) for c in history.get_messages()]
    payload = {
        "name": name,
        "created_at": created_at,
        "updated_at": now,
        "messages": messages,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

The serialization handles three part types: text, function_call, and function_response (`manager.py` lines 47-77). Loading returns raw dicts rather than reconstructed `types.Content` objects — this avoids a tight coupling to the SDK's internal deserialization and lets the caller reconstruct objects in whatever way suits their use.

### The Conversation State Problem

Saving and restoring conversation history is straightforward. The hard problem is that history contains references to external state that may have changed:

- History says "I wrote this function to auth.py" — but auth.py was later manually edited
- History says "the tests were passing" — but someone pushed a breaking change
- History references a file that was deleted

There's no universal solution. The pragmatic approaches:
1. **Fresh context on resume**: Re-read relevant files at the start of each session, don't trust history's statements about file contents
2. **Git checkpoints**: The `checkpointing_enabled` config flag (`config.py` line 101) enables git-based snapshots before destructive operations — creating a recovery point lets you "undo" agent actions
3. **Summarize and restart**: Rather than resuming a 50-message history, have the agent summarize what it accomplished, clear history, and start fresh with that summary as context

### Why State Is the Hard Part

Every agent framework sells a solution to the state problem, but the truth is that state is application-specific. LangGraph's `MemorySaver` checkpoints graph state — but the state schema is whatever you define. Pydantic AI returns `RunResult` with history — but what you *do* with that history between sessions is up to you. The primitives are there; the wisdom of how to use them comes from understanding the problem at this level of detail.

---

## 8. Safety and Control — Keeping Agents on a Leash

An agent that can execute shell commands, edit files, and fetch URLs can do real damage if it goes off-script. The safety layer provides multiple independent defense lines.

### Confirmation Gates

The `should_confirm()` method on `BaseTool` is the primary gate. It returns `None` for safe, read-only operations and a `ToolConfirmation` for anything destructive:

```python
# shell.py — ALWAYS requires confirmation
def should_confirm(self, params: dict) -> ToolConfirmation | None:
    command = params.get("command", "?")
    return ToolConfirmation(
        description=f"Run shell command: {command}",
        details={"command": command, "timeout": params.get("timeout", 30)},
    )

# read_file — no confirmation (safe read-only operation)
def should_confirm(self, params: dict) -> ToolConfirmation | None:
    return None
```

The `edit_file` tool (`filesystem/edit_file.py` lines 94-106) shows confirmation for any write operation. The confirmation includes what will change, enabling meaningful user review before approval.

### The Approval Mode Spectrum

The `ApprovalMode` in config (`config.py` line 25) defines how the confirmation gate behaves:

```python
ApprovalMode = Literal["interactive", "auto_edit", "yolo"]
```

- **interactive**: Every confirmation gate prompts the user. Default. Maximum safety.
- **auto_edit**: File edits auto-approve, shell commands still prompt. Useful for coding agents where you trust file changes but not arbitrary shell execution.
- **yolo**: Everything auto-approves. For trusted, well-scoped automation tasks where you've reviewed what the agent will do.

This is the trust spectrum. Move along it based on how well you understand and trust the agent's behavior in a given context. NEVER start at yolo for a new task.

### Abort Signals

The `_abort_signal` (`asyncio.Event`) in `AgentLoop` is how Ctrl+C works properly during agent execution:

```python
# In App.run():
except KeyboardInterrupt:
    if self._agent_loop:
        self._agent_loop.abort_signal.set()  # signal the inner loop
    self._console.print("\n[dim]Interrupted.[/dim]")

# In AgentLoop.run():
while not self._abort_signal.is_set():     # checked before each API call
    async for chunk in self._content_generator.generate_stream(...):
        if self._abort_signal.is_set():    # checked during streaming
            break
```

The signal is checked at multiple points: before entering the loop, during streaming, between tool executions. This means interruption is responsive without requiring any cancellation of running async tasks — the agent finishes its current atomic operation and then checks the signal.

The abort signal is also passed to tool implementations (`execute(params, abort_signal)`). Long-running tools like `ShellTool` check it after command completion:

```python
if abort_signal is not None and abort_signal.is_set():
    process.kill()
    return ToolResult(llm_content="Command aborted by user.", ...)
```

### Sandboxing

The `sandbox` config field (`config.py` line 97) supports `"none"` and `"docker"` modes. Docker sandboxing would run tool executions inside a container, preventing any filesystem or network access beyond what's explicitly mounted. This is the strongest safety primitive — even a compromised or hallucinating tool can't escape the container.

The architecture makes sandboxing clean to add: `ToolRegistry` returns tool declarations unchanged, but `AgentLoop._execute_function_call()` could route execution to a sandboxed subprocess rather than calling `tool.execute()` directly.

---

## 9. Practical Guide: Building This Yourself

### With LangGraph

LangGraph imposes a graph structure on what is fundamentally a loop. Here's how to think about the mapping:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# Our ConversationHistory == LangGraph's MessagesState
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Our BaseTool == LangGraph's @tool function
@tool
def read_file(path: str) -> str:
    """Read a file. Use for reading source code and configuration files."""
    return Path(path).read_text()

@tool
def run_shell_command(command: str) -> str:
    """Run a shell command. Returns stdout and stderr."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"Exit: {result.returncode}\n{result.stdout}\n{result.stderr}"

tools = [read_file, run_shell_command]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

# Our AgentLoop.run() while loop == this graph
def agent_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = (
    StateGraph(AgentState)
    .add_node("agent", agent_node)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue)
    .add_edge("tools", "agent")
    .compile()
)

# Equivalent of AgentLoop.run("your message")
for event in graph.stream({"messages": [("user", "refactor auth.py")]}):
    print(event)
```

What LangGraph gives you for free: streaming, state persistence (with checkpointers), human-in-the-loop via `interrupt()`, and visualization of the graph. What you lose: the explicit confirmation handshake, the dual-output pattern, and fine-grained tool validation. Add these back by wrapping tools with pre/post hooks.

### With Pydantic AI

Pydantic AI is the closest in spirit to our implementation — it's also built around the idea of owning the loop rather than hiding it:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import asyncio

model = GeminiModel("gemini-2.5-flash")
agent = Agent(
    model,
    system_prompt=(
        "You are a coding assistant. Read files before modifying them. "
        "Confirm destructive operations before executing."
    ),
)

# Our BaseTool.execute() == @agent.tool async function
@agent.tool
async def read_file(ctx: RunContext[None], path: str) -> str:
    """Read a file and return its contents."""
    return Path(path).read_text()

@agent.tool
async def edit_file(ctx: RunContext[None], path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string (must be unique)."""
    content = Path(path).read_text()
    count = content.count(old_string)
    if count != 1:
        return f"Error: old_string found {count} times, must be exactly 1"
    Path(path).write_text(content.replace(old_string, new_string, 1))
    return f"Edited {path}"

# Equivalent of AgentLoop.run() with streaming
async def main():
    async with agent.run_stream("refactor auth.py to use dataclasses") as result:
        async for text in result.stream():
            print(text, end="", flush=True)

asyncio.run(main())
```

What Pydantic AI gives you: Pydantic type safety throughout, clean dependency injection via `RunContext`, and a familiar Python-native API. What you need to add: the confirmation handshake (use the queue pattern from section 4), session persistence (pass `message_history` between runs), and the dual-output pattern (tools return strings, but you can return structured objects and display them separately).

### With Google ADK

Google ADK is the highest-level of the three and the most opinionated:

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

def read_file(path: str) -> dict:
    """Read a file and return its contents."""
    return {"content": Path(path).read_text()}

def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """Edit a file by replacing old_string with new_string."""
    content = Path(path).read_text()
    new_content = content.replace(old_string, new_string, 1)
    Path(path).write_text(new_content)
    return {"success": True, "path": path}

# Our AgentLoop == ADK's LlmAgent + Runner
agent = LlmAgent(
    name="coding_assistant",
    model="gemini-2.5-flash",
    instruction="You are a coding assistant...",
    tools=[FunctionTool(read_file), FunctionTool(edit_file)],
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="coding_assistant", session_service=session_service)

# Our App.run() REPL == driving the runner in a loop
import asyncio
from google.adk.types import Content, Part

async def run():
    session = await session_service.create_session(app_name="coding_assistant", user_id="user1")
    message = Content(role="user", parts=[Part(text="refactor auth.py")])
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="")

asyncio.run(run())
```

ADK's advantage: built-in session management, multi-agent composition (`SequentialAgent`, `ParallelAgent`), and deep Gemini integration. The tradeoff: less control over the inner loop, harder to customize the confirmation handshake, and the event model is different from our pub/sub approach.

### From Scratch — The Minimal Agent Loop

The most educational thing you can do is implement this from scratch. Here's the minimal agent in roughly 60 lines:

```python
import asyncio
from google import genai
from google.genai import types

API_KEY = "your-key-here"

# --- Tool definition ---
def read_file(path: str) -> str:
    try:
        return open(path).read()
    except Exception as e:
        return f"Error: {e}"

TOOLS = [{"function_declarations": [{
    "name": "read_file",
    "description": "Read a file and return its contents",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    },
}]}]

TOOL_FUNCTIONS = {"read_file": read_file}

# --- The loop ---
async def run_agent(user_message: str) -> None:
    client = genai.Client(api_key=API_KEY)
    history: list[types.Content] = []
    history.append(types.Content(role="user", parts=[types.Part.from_text(user_message)]))

    while True:
        # Generate
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(tools=[
                types.Tool(function_declarations=TOOLS[0]["function_declarations"])
            ]),
        )

        text_parts, function_calls, model_parts = [], [], []
        for chunk in response:
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
                    text_parts.append(part.text)
                    model_parts.append(types.Part.from_text(part.text))
                elif part.function_call:
                    fc = part.function_call
                    function_calls.append(fc)
                    model_parts.append(types.Part(function_call=fc))

        history.append(types.Content(role="model", parts=model_parts))

        if not function_calls:
            print()
            break  # Model is done

        # Execute tools
        response_parts = []
        for fc in function_calls:
            fn = TOOL_FUNCTIONS.get(fc.name)
            result = fn(**dict(fc.args)) if fn else f"Unknown tool: {fc.name}"
            print(f"\n[Tool: {fc.name}({dict(fc.args)})] -> {result[:100]}...")
            response_parts.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name,
                response={"result": result},
            )))

        history.append(types.Content(role="user", parts=response_parts))
        # loop continues

asyncio.run(run_agent("Read the file README.md and summarize it"))
```

This works. It's missing validation, confirmation gates, the event system, error handling, abort signals, and everything else — but it demonstrates that the core loop is simple. Every framework is this plus ~5,000 lines of the things that matter in production.

The "aha moment" progression as you build up from this:
1. This works. Wow.
2. Add error handling. The model should see tool errors, not crashes.
3. Add a second tool. Notice the tool dispatch logic grows.
4. Extract a `ToolRegistry`. First real abstraction.
5. Add confirmation before destructive tools. Learn about the async handshake.
6. Add the event system. Decouple rendering.
7. History is getting long. Add compression.
8. Sessions end. Add persistence.
9. You've re-implemented PyGeminiCLI.

---

## 10. Key Takeaways

**The loop is simple.** Look at `agent_loop.py` — the core ReAct while loop is about 50 lines. Every agent system, regardless of how sophisticated it appears, runs this same loop. The complexity lives in everything built around it.

**Every framework hides the same ~100 lines.** LangGraph's graph, Pydantic AI's `run()`, CrewAI's task execution, Google ADK's Runner — they all contain the generate → tools → generate loop. They differ in ergonomics, observability, and what they make easy. They agree on the fundamental mechanics.

**The model controls the loop.** This is not a minor implementation detail. The agent doesn't execute a plan; the model *generates* each step by looking at history. The model decides when to call tools and when to stop. Your code provides capabilities; the model provides strategy. If the agent behaves unexpectedly, the cause is almost always in the history it's seeing, the tool descriptions, or the system prompt.

**Context is everything.** The model can only reason about what's in its context window. Conversation history, system prompts, tool descriptions, injected project files, memories — these are the levers that control behavior. Perfect tool implementations with mediocre context get worse results than mediocre tool implementations with excellent context.

**The dual-output pattern is always worth it.** Every tool that shows something to a user has two audiences with different needs. Build `llm_content` for reasoning; build `display_content` for humans. Don't conflate them.

**The confirmation handshake is the hardest pattern to get right.** The async queue pattern in `EventEmitter` is what makes the agent pausable at arbitrary points without threads or callbacks. It's reusable verbatim in any framework that uses asyncio.

**Understanding the raw loop makes you better with every framework.** When LangGraph behaves unexpectedly, you can reason about which node is making which API call. When Pydantic AI's tool invocation order surprises you, you know it's because the model saw something unexpected in the context. The framework is always the loop plus abstractions; knowing the loop, you can debug any abstraction.

---

## Reference: File-to-Concept Map

| File | Primary Concept |
|---|---|
| `src/pygemini/core/agent_loop.py` | The ReAct loop — everything starts here |
| `src/pygemini/core/content_generator.py` | LLM API streaming, sync-to-async bridge |
| `src/pygemini/core/events.py` | Pub/sub event system, confirmation handshake |
| `src/pygemini/core/history.py` | Conversation state, token estimation |
| `src/pygemini/core/config.py` | Layered config, approval modes |
| `src/pygemini/core/prompts.py` | System prompt construction, context injection |
| `src/pygemini/tools/base.py` | Tool contract, dual-output pattern |
| `src/pygemini/tools/registry.py` | Tool registration, filtered declarations |
| `src/pygemini/tools/shell.py` | Complex tool: confirmation, output truncation |
| `src/pygemini/tools/filesystem/edit_file.py` | File mutation: validation, diff display |
| `src/pygemini/context/memory_store.py` | Cross-session persistence pattern |
| `src/pygemini/session/compressor.py` | Context compression via LLM summarization |
| `src/pygemini/session/manager.py` | Session save/load, serialization |
| `src/pygemini/cli/app.py` | Wiring: how all components connect |
