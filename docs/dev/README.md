> **Status:** Reference · **Last revised:** 2026-05-06 · **Type:** developer documentation index

# JAC Developer Documentation

These docs describe JAC as it exists — for contributors, debuggers, and AI agents who need to understand what was built, how the layers connect, and where to look when something breaks.

**These are NOT specs.** Specifications live in [`docs/contracts/`](../contracts/). This documentation describes what is currently implemented and why things are shaped the way they are.

**These are NOT a roadmap.** The roadmap lives in [`docs/ROADMAP.md`](../ROADMAP.md). These docs describe shipped behaviour through the current shipped frontier.

## Contents

| File | Description |
|---|---|
| [architecture.md](architecture.md) | Layer diagram, dependency rules, data flow for a single chat turn, key integration seams |
| [components.md](components.md) | C0–C6 component map — what each component shipped, key files, integration points |
| [state-layer.md](state-layer.md) | SQLite repos, table activation state, repository pattern, migration pattern, seeder |
| [agents-layer.md](agents-layer.md) | Agent factory, `config_loader` flow, tool resolution, SDK independence |
| [runtime-layer.md](runtime-layer.md) | `RunCoordinator`, `EventBus`, `SessionState`, approval and question primitives, model factory |
| [cli-layer.md](cli-layer.md) | Terminal adapter, Click commands, `ChatApp`, input parsing, slash command registry, renderer |

## Keeping these current

Update the relevant doc when a component ships or a layer changes substantially. If the update is large, delegate to a sub-agent (Sonnet-class is sufficient) — the source code is the ground truth. These docs should stay accurate enough that a fresh agent can orient quickly without reading all the source files.
