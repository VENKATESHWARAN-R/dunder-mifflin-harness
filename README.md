# dunder-mifflin-harness

> A research and development project exploring the limits of autonomous, long-running agentic systems.

Inspired by Anthropic's engineering blog series on agentic harnesses:
- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Harness Design for Long-Running Application Development](https://www.anthropic.com/engineering/harness-design-long-running-apps)

Anthropic advocates shifting from prompt engineering to **system engineering** — using structured harnesses for external state management and context engineering for just-in-time data retrieval. Together, these patterns enable AI agents to handle long-running, complex tasks without suffering from "agent amnesia" or blowing their attention budget on noise.

This project takes that philosophy and puts it to the test.

Built on top of [dunder-mifflin-play](https://github.com/VENKATESHWARAN-R/dunder-mifflin-play) — a role-based multi-agent system modelled on an IT team — this repo reimagines that architecture using Google ADK's graph-based workflow primitives. The goal is a dynamic, application-agnostic agentic harness: one that can autonomously plan, implement, test, and iterate on real software tasks from a single high-level prompt.

**The core hypothesis:** a well-structured multi-agent harness with explicit role separation, persistent state, and disciplined context management can match or exceed the task completion benchmarks reported in Anthropic's internal harness work — at lower cost and with greater transparency into what the system is actually doing.

This is not a product. It is a deliberate experiment in understanding what agentic systems are genuinely capable of today.
