> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** guide index

# JAC User Guide

This guide covers everything you need to install, configure, and use JAC ("Just Another CLI") — an agentic coding harness built on Pydantic AI.

## Contents

| File | Description |
|---|---|
| [getting-started.md](getting-started.md) | Install JAC, run the onboarding wizard, and verify your setup |
| [usage.md](usage.md) | One-shot mode, chat mode, resume, file attachments, slash commands, approval modes |
| [configuration.md](configuration.md) | Workspace scopes, `settings.json` reference, environment variables, profiles |
| [sdk.md](sdk.md) | Use JAC as a library: agent factory, RunCoordinator, embedding in your own code |

## Quick start

```bash
# Install
uv tool install jac

# First-time setup
jac init --global

# Say hello
jac "say hello"

# Interactive mode
jac chat
```

See [getting-started.md](getting-started.md) for details.
