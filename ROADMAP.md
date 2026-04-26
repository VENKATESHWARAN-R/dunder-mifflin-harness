# Roadmap — dunder-mifflin-harness

One slice per weekend. Each slice ends with something runnable and demoable. Sunday night = ship + post a 30-second demo somewhere (even one friend counts).

**Rules:**
- Don't redesign earlier slices. W6 is the refactor weekend.
- Each slice is allowed to be embarrassing.
- Finish early → ship and stop. Don't roll into the next one.
- Found something to fix/improve along the way? Add it under "Backlog" below.

---

## Slices

### W0 — "It runs"
- [ ] CLI you can type into
- [ ] One LLM call, one provider
- [ ] `harness "say hi"` prints a response
- [ ] `main.py` is no longer a stub
- [ ] Demo posted

### W1 — "It remembers"
- [ ] Persistent conversation memory (sqlite or json — pick boring)
- [ ] `harness chat` enters a loop
- [ ] Quit and resume works
- [ ] Demo posted

### W2 — "It can do something"
- [ ] `read_file` tool wired
- [ ] `write_file` tool wired
- [ ] Tool-calling loop works end-to-end
- [ ] It can create a file on request
- [ ] Demo posted

### W3 — "It plans before it acts"
- [ ] Planner node (LLM → list of steps)
- [ ] Builder node (executes steps using tools)
- [ ] Hardcoded wiring is fine
- [ ] Small task runs end-to-end (e.g. fibonacci script)
- [ ] Demo posted

### W4 — "It has tiers"
- [ ] Tier-3 model for planner
- [ ] Tier-2 model for builder
- [ ] Tier-1 model for at least one cheap node
- [ ] Two providers minimum
- [ ] Per-call cost tracked in state store
- [ ] Cost breakdown printable per run
- [ ] Demo posted

### W5 — "It evaluates itself"
- [ ] `evaluate` node after build
- [ ] LLM grades pass/fail
- [ ] One retry loop on failure
- [ ] Ambiguous task fails eval, retries, passes
- [ ] Demo posted

### W6 — "It's a graph"
- [ ] Graph abstraction picked (LangGraph or homegrown)
- [ ] Uniform node state interface
- [ ] W5 functionality refactored onto the graph
- [ ] No new behavior — just composability
- [ ] Demo posted

### W7+ — "It picks up tasks"
- [ ] `task_router` node
- [ ] Reads a TASKS.md file
- [ ] Multi-task runs work
- [ ] Full V0 cycle: plan → router → loader → build → evaluate
- [ ] Demo posted

---

## Backlog

Stuff to fix/improve found along the way. Pull from here when a slice finishes early, or schedule into a future weekend.

- [ ] _(empty — add as you go)_

---

## Done

Move slices here once shipped, with the date.

- _(empty)_

---

## Notes

Anything worth remembering — a decision, a dead end, a trick that worked. Keep it short.

- _(empty)_
