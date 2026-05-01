"""PyGeminiCLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import click

from pygemini.cli.app import App
from pygemini.core.config import load_config
from pygemini.core.events import CoreEvent, StreamTextEvent, ToolExecutingEvent


@click.command()
@click.option("--model", "-m", default=None, help="Override the model name.")
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Run a single prompt in headless mode (non-interactive).",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for headless mode.",
)
@click.option(
    "--approval-mode",
    type=click.Choice(["interactive", "auto_edit", "yolo"]),
    default=None,
    help="Approval mode for tool execution.",
)
@click.option(
    "--sandbox",
    type=click.Choice(["none", "docker"]),
    default=None,
    help="Sandbox mode for shell commands.",
)
@click.option(
    "--yolo",
    is_flag=True,
    default=False,
    help="Shorthand for --approval-mode yolo (skip all confirmations).",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
@click.version_option(version="0.1.0", prog_name="pygemini")
def cli(
    model: str | None,
    prompt: str | None,
    output_format: str,
    approval_mode: str | None,
    sandbox: str | None,
    yolo: bool,
    debug: bool,
) -> None:
    """PyGeminiCLI — AI coding assistant in your terminal."""
    # Configure logging
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # --yolo is a shorthand; explicit --approval-mode takes precedence
    effective_approval_mode = approval_mode or ("yolo" if yolo else None)

    # Build config with CLI overrides (None values are ignored inside load_config)
    config = load_config(
        model=model,
        approval_mode=effective_approval_mode,
        sandbox=sandbox,
    )

    app = App(config=config)

    if prompt:
        if output_format == "json":
            asyncio.run(_run_headless_json(app, prompt))
        else:
            asyncio.run(app.run_headless(prompt))
    else:
        asyncio.run(app.run())


async def _run_headless_json(app: App, prompt: str) -> None:
    """Run headless mode and emit a single JSON object to stdout.

    Collects all text chunks emitted by the agent loop and writes:

        {"prompt": "...", "response": "...", "tool_calls": [...]}
    """
    collected_text: list[str] = []
    tool_calls: list[dict[str, object]] = []

    # Subscribe to events before running so we capture everything.
    emitter = app._event_emitter  # noqa: SLF001

    async def on_stream_text(event: object) -> None:
        if isinstance(event, StreamTextEvent):
            collected_text.append(event.text)

    async def on_tool_executing(event: object) -> None:
        if isinstance(event, ToolExecutingEvent):
            tool_calls.append({"name": event.tool_name, "params": event.params})

    emitter.on(CoreEvent.STREAM_TEXT, on_stream_text)
    emitter.on(CoreEvent.TOOL_EXECUTING, on_tool_executing)

    try:
        agent_loop = app._ensure_agent_loop()  # noqa: SLF001
        await agent_loop.run(prompt)
    except ValueError as exc:
        output = {
            "prompt": prompt,
            "error": str(exc),
            "response": "",
            "tool_calls": [],
        }
        click.echo(json.dumps(output))
        sys.exit(1)

    output = {
        "prompt": prompt,
        "response": "".join(collected_text),
        "tool_calls": tool_calls,
    }
    click.echo(json.dumps(output))


if __name__ == "__main__":
    cli()
