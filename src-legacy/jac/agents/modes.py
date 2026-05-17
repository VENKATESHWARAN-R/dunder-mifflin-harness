"""Slash-mode addendums for system prompts."""

from __future__ import annotations

INIT_MODE_ADDENDUM = """\
SLASH MODE: /init

You are surveying the user's workspace to produce an AGENTS.md file at
the project root. The file is read by future agent runs as project-level
instructions, so be precise and durable.

PROCESS:
1. List the top of the project tree (list_directory on cwd; one or two
   levels deep). Note the package manager, language, frameworks, build
   tool, test runner, and entrypoints.
2. Read 2-4 anchor files: README, top-level config (pyproject.toml /
   package.json), and the most central module if obvious.
3. If AGENTS.md already exists, read it first and merge useful details.
4. Compose AGENTS.md with these sections, in order:
     - Project name and one-line description
     - How to install / run / test (commands, not prose)
     - Repo layout (one bullet per top-level dir)
     - Conventions worth knowing (lint, format, type-check, naming)
5. Write AGENTS.md to the project root using write_file.

RULES:
- Do not invent commands. If you cannot tell, say "TODO: confirm with user".
- Keep it under 120 lines. Future agents skim it; do not write a tutorial.
- Do not commit, push, or modify any other file.
"""

PLAN_MODE_ADDENDUM = """\
SLASH MODE: /plan

You are producing a one-shot Plan for the user's request. Output the
structured Plan only - no narration. The user will review the plan
before any builder is summoned.
"""

MODE_PROMPTS: dict[str, str] = {
    "init": INIT_MODE_ADDENDUM,
    "plan": PLAN_MODE_ADDENDUM,
}


def build_instructions(base: str, mode: str | None) -> str:
    """Compose a base system prompt with the slash-mode addendum, if any."""
    if mode is None:
        return base
    addendum = MODE_PROMPTS.get(mode)
    if not addendum:
        return base
    return f"{base}\n\n---\n\n{addendum}"
