> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** user guide

# Getting Started

This guide walks you from zero to a working JAC session. By the end you will have JAC installed, a provider configured, and a smoke test passing.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) — the package manager used throughout

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell after installation, or source the updated profile.

## Install options

### As a tool (recommended for most users)

Installs JAC into an isolated environment and puts the `jac` binary on your `PATH`:

```bash
uv tool install jac
```

Verify:

```bash
jac --version
```

### Editable dev install (for contributors)

Clone the repo and sync the development environment:

```bash
git clone https://github.com/your-org/dunder-mifflin-harness
cd dunder-mifflin-harness
uv sync
```

Run JAC through `uv run` instead of the global binary:

```bash
uv run jac --version
uv run jac "say hello"
```

All examples in this guide use the global `jac` command. If you are on the dev install, substitute `uv run jac`.

## First-time setup

JAC needs to know which AI provider to use and how to authenticate with it. The `jac init` wizard handles this interactively.

### Global setup (recommended starting point)

Run the global wizard once to configure your default provider:

```bash
jac init --global
```

The wizard will:

1. Ask which provider to use (`gateway`, `anthropic`, `openai`, `google-gla`, `ollama`, `openrouter`, or `litellm`)
2. Prompt for the relevant API key
3. Ask which model tier to use by default (`scout`, `worker`, or `architect`)
4. Write `~/.jac/settings.json` with your choices
5. Write `~/.jac/.env` with your API key

The global config applies to every JAC session where no project-level config is present.

### Project-level setup

Inside a project repository you can create project-specific settings that override the global defaults:

```bash
cd /path/to/your/project
jac init
```

This writes `.agents/settings.json` and `.agents/.env` into the current directory. Project settings take precedence over global settings. Commit `.agents/settings.json` to share config with your team; add `.agents/.env` to `.gitignore` (the wizard does this automatically).

See [configuration.md](configuration.md) for full details on the workspace hierarchy and all available settings.

## Smoke test

After setup, verify JAC can reach your provider:

```bash
jac "say hello"
```

You should see a short response printed to the terminal. If you see an authentication error, recheck your API key with `jac doctor`.

## Interactive mode

For a multi-turn conversation, use chat mode:

```bash
jac chat
```

The interactive REPL starts. Type a message and press Enter to send it. Press Ctrl+C to cancel an in-flight request. Press Ctrl+D or type `/quit` to exit.

See [usage.md](usage.md) for the full feature set including file attachments, slash commands, and inline shell.

## Diagnostics

If something is not working, run the diagnostics command:

```bash
jac doctor
```

`jac doctor` checks:

- Workspace discovery (global vs. project paths)
- Settings file validity
- API key presence for the configured provider
- Database file accessibility
- Skills and MCP server counts

## Check your version

```bash
jac --version
```

JAC follows `0.x.y` versioning while in alpha. The version is read from the installed package metadata.
