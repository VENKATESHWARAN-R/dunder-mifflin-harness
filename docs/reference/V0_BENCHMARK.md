# V0 Benchmark — Notes CLI

> **Status:** Locked · **Last revised:** 2026-05-02 · **Type:** test case spec

## Purpose

This is the test case for the V0 JAC run. The harness receives the prompt below and must
autonomously produce a working Python CLI app. Success is measured against the acceptance
criteria in this document.

Chosen because: every feature is evaluatable with shell commands and file checks — no browser,
no mocking, no ambiguity. Directory-based storage exercises filesystem tools realistically.
Search and tag filtering provide genuine Worker-tier logic alongside simpler Scout-tier CRUD.

---

## The Prompt

> Build a Python command-line note-taking app called `notes`.
>
> Features:
> - Create a note with a title and body
> - List all notes (plain and detailed formats)
> - View a specific note by ID
> - Search notes by keyword across title and body
> - Add tags to a note
> - Filter notes by tag
> - Delete a note
>
> Store each note as an individual Markdown file with YAML frontmatter in a `notes/` directory.
> The app should be installable and runnable as `notes <command>`.

---

## Expected Task Decomposition

The planner should decompose this into roughly 8–9 tasks. Expected breakdown and tier routing:

| # | Task | Expected tier |
|---|---|---|
| 1 | Project setup: `pyproject.toml`, CLI entry point, `notes/` directory init | Architect |
| 2 | `notes add <title> <body>` — create note, write markdown file with frontmatter | Scout |
| 3 | `notes list` — list all notes, plain format (ID + title) | Scout |
| 4 | `notes list --detailed` — detailed format with tags and body preview | Scout |
| 5 | `notes view <id>` — display full note by ID | Scout |
| 6 | `notes delete <id>` — remove note file | Scout |
| 7 | `notes search <keyword>` — search title and body across all notes | Worker |
| 8 | `notes tag <id> <tag>` — add tag to note frontmatter | Worker |
| 9 | `notes list --tag <tag>` — filter list output by tag | Worker |

---

## Acceptance Criteria

The `evaluate` node runs these shell commands after each task and checks output/exit codes.
All commands assume the app is installed and runnable as `notes`.

### Task 2 — Create

```bash
notes add "Meeting notes" "Discussed roadmap and priorities"
# Exit code: 0
# Stdout contains: note ID (e.g. "Created note #1")
# File exists: notes/1-meeting-notes.md (or similar slug)
```

### Task 3 — List (plain)

```bash
notes list
# Exit code: 0
# Stdout contains: "1" and "Meeting notes"
```

### Task 4 — List (detailed)

```bash
notes list --detailed
# Exit code: 0
# Stdout contains title, ID, and body preview
```

### Task 5 — View

```bash
notes view 1
# Exit code: 0
# Stdout contains: "Meeting notes" and "Discussed roadmap"
```

### Task 6 — Delete

```bash
notes add "To delete" "Temporary note"
notes delete 2
# Exit code: 0
notes list
# Stdout does NOT contain "To delete"
```

### Task 7 — Search

```bash
notes add "Planning session" "Reviewed the roadmap items"
notes search "roadmap"
# Exit code: 0
# Stdout contains both "Meeting notes" and "Planning session"
notes search "zzznomatch"
# Exit code: 0
# Stdout contains: no results message
```

### Task 8 — Tag

```bash
notes tag 1 "work"
notes view 1
# Stdout contains: "work" in tags section
```

### Task 9 — Filter by tag

```bash
notes tag 3 "planning"
notes list --tag work
# Stdout contains "Meeting notes", does NOT contain "Planning session"
notes list --tag planning
# Stdout contains "Planning session", does NOT contain "Meeting notes"
```

---

## V0 Success Criteria

| Criteria | Target |
|---|---|
| App runs without crashing | All commands exit 0 |
| All 7 features pass acceptance tests | 7/7 |
| Partial credit | ≥ 5/7 features passing |
| Total cost | < $15 |
| Harness completes autonomously | No human intervention |
| Cost log shows tier distribution | At least 2 tiers used |

A run scoring 5/7 features and under $15 is a passing V0. 7/7 is the goal.

---

## Out-of-Scope Extensions

Ideas noted during benchmark selection — not part of this checkpoint, but the data model
should not actively prevent them. These would be added by later roadmap components if and
when the project picks them up; see [`docs/ROADMAP.md`](../ROADMAP.md):

- **Topic grouping** — organise notes into named collections or folders
- **Connected notes** — graph-based linking between notes (Obsidian-style `[[note title]]`
  references), enabling a backlinks view and graph traversal
- **Full-text index** — faster search via an index file rather than scanning all files
- **Note history** — git-backed version history per note (the git tool is already available)
- **Export** — bundle notes into a single PDF or HTML file
