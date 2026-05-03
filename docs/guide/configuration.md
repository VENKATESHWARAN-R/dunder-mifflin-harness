> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** user guide

# Configuration

JAC has a layered configuration system. Settings are read from multiple sources and merged in a defined order. This document covers workspace scopes, the `settings.json` format, all environment variables, and how to use profiles.

## Workspace scopes

JAC looks for configuration in two scopes: global and project.

### Global workspace (`~/.jac/`)

```
~/.jac/
  settings.json         # global defaults
  .env                  # API keys (gitignored by nature — your home dir)
  skills/               # globally available skills (markdown files)
  mcp/                  # globally available MCP server configs (JSON files)
  runs/
    <cwd-hash>/
      state.db          # state database for sessions outside a project repo
```

Global settings apply whenever no project-level config is found.

### Project workspace (`<repo>/.agents/`)

```
<repo>/.agents/
  settings.json         # project defaults (commit this)
  settings.local.json   # local overrides (gitignored)
  .env                  # project-scoped API keys (gitignored)
  skills/               # project-scoped skills
  mcp/                  # project-scoped MCP server configs
  state.db              # state database for this project
```

A project workspace is detected when JAC finds a `.agents/` directory in the current directory or any ancestor. Project settings override global settings.

### Merge order

Settings are applied from lowest to highest priority:

```
global settings.json
  < project settings.json
    < project settings.local.json
      < environment variables
        < CLI flags
```

A value at a higher level always wins. For example, an env var overrides anything written in `settings.json`.

## `settings.json` reference

Both global and project `settings.json` files use the same schema.

```json
{
  "default_provider": "gateway",
  "default_tier": "worker",
  "model_tiers": {
    "scout":    ["anthropic:claude-haiku-4-5"],
    "worker":   ["anthropic:claude-sonnet-4-6"],
    "architect": ["anthropic:claude-opus-4-5"]
  },
  "active_profile": null,
  "profiles": {
    "work": {
      "default_provider": "anthropic",
      "default_tier": "architect"
    }
  }
}
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `default_provider` | string | `"gateway"` | Provider to use when no model string is given. One of: `gateway`, `anthropic`, `openai`, `google-gla`, `ollama`, `openrouter`, `litellm` |
| `default_tier` | string | `"worker"` | Default model tier. One of: `scout`, `worker`, `architect` |
| `model_tiers` | object | — | Maps each tier name to a list of model ID strings. The first entry in the list is used. Format: `"<provider>:<model-id>"` |
| `active_profile` | string or null | `null` | Name of the profile to activate. Overridden by `JAC_ACTIVE_PROFILE` |
| `profiles` | object | `{}` | Named profile objects. Each profile can override any top-level field |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `JAC_MODEL` | — | Override the model string for the session (e.g. `anthropic:claude-sonnet-4-6`) |
| `JAC_ACTIVE_PROFILE` | — | Activate a named profile from `settings.json` |
| `JAC_CONFIG_DIR` | `~/.jac` | Override the global config directory |
| `JAC_MAX_ATTACHMENT_BYTES` | `200000` | Maximum file attachment size in bytes (~200 KB) |
| `JAC_SHELL_TIMEOUT_SECONDS` | `10.0` | Timeout for inline shell commands (seconds) |
| `JAC_SHELL_MAX_OUTPUT_CHARS` | `20000` | Maximum characters captured from shell command output |

### Provider API keys

| Provider | Environment variable(s) |
|---|---|
| `gateway` | `PYDANTIC_AI_GATEWAY_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google-gla` | `GEMINI_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL` (optional), `OLLAMA_API_KEY` (optional) |
| `openrouter` | `OPENROUTER_API_KEY` |
| `litellm` | `LITELLM_API_BASE`, `LITELLM_API_KEY` |

You can place these in `~/.jac/.env` (global) or `<repo>/.agents/.env` (project). JAC loads `.env` files automatically on startup.

## Profiles

Profiles let you maintain multiple named configurations — for example, a personal profile using one provider and a work profile using another.

### Defining profiles

Add a `profiles` key to your `settings.json`:

```json
{
  "default_provider": "gateway",
  "default_tier": "worker",
  "profiles": {
    "personal": {
      "default_provider": "gateway",
      "default_tier": "worker"
    },
    "work": {
      "default_provider": "anthropic",
      "default_tier": "architect"
    }
  }
}
```

### Activating a profile

Set `active_profile` in `settings.json`:

```json
{
  "active_profile": "work"
}
```

Or use the environment variable:

```bash
JAC_ACTIVE_PROFILE=work jac chat
```

### Profile-scoped environment variables

You can scope API keys to a specific profile using the pattern `JAC_PROFILE_<SLUG>_<VAR>`, where `<SLUG>` is the profile name uppercased:

```bash
JAC_PROFILE_WORK_ANTHROPIC_API_KEY=sk-ant-...
JAC_PROFILE_PERSONAL_PYDANTIC_AI_GATEWAY_API_KEY=gw-...
```

These are only loaded when the matching profile is active.

**Note:** Profile hot-swapping mid-session is not supported in C5. Set `JAC_ACTIVE_PROFILE` before launching JAC.

## Model tiers

Three tiers are available, each intended for a different cost/capability trade-off:

| Tier | Intended use | Speed | Cost |
|---|---|---|---|
| `scout` | File reading, boilerplate, formatting | Fastest | Cheapest |
| `worker` | Feature implementation, testing, evaluation | Balanced | Moderate |
| `architect` | Planning, architecture, complex debugging | Slower | Most expensive |

Map tiers to model IDs in your `settings.json` under `model_tiers`. Example mapping across providers:

**Anthropic:**
```json
{
  "model_tiers": {
    "scout":     ["anthropic:claude-haiku-4-5"],
    "worker":    ["anthropic:claude-sonnet-4-6"],
    "architect": ["anthropic:claude-opus-4-5"]
  }
}
```

**OpenAI:**
```json
{
  "model_tiers": {
    "scout":     ["openai:gpt-4o-mini"],
    "worker":    ["openai:gpt-4o"],
    "architect": ["openai:o1"]
  }
}
```

Switch the active tier mid-session with `/tier architect`.

## Example: two-profile setup

A common pattern is to have a personal profile using the gateway (cheaper) and a work profile using Anthropic directly:

**`~/.jac/settings.json`:**
```json
{
  "default_provider": "gateway",
  "default_tier": "worker",
  "active_profile": "personal",
  "profiles": {
    "personal": {
      "default_provider": "gateway",
      "default_tier": "worker",
      "model_tiers": {
        "scout":     ["gateway:claude-haiku-4-5"],
        "worker":    ["gateway:claude-sonnet-4-6"],
        "architect": ["gateway:claude-opus-4-5"]
      }
    },
    "work": {
      "default_provider": "anthropic",
      "default_tier": "architect",
      "model_tiers": {
        "scout":     ["anthropic:claude-haiku-4-5"],
        "worker":    ["anthropic:claude-sonnet-4-6"],
        "architect": ["anthropic:claude-opus-4-5"]
      }
    }
  }
}
```

**`~/.jac/.env`:**
```
PYDANTIC_AI_GATEWAY_API_KEY=gw-...
JAC_PROFILE_WORK_ANTHROPIC_API_KEY=sk-ant-...
```

Switch to the work profile for a session:

```bash
JAC_ACTIVE_PROFILE=work jac chat
```
