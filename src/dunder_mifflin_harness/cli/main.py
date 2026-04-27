"""Click entrypoints for the harness CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import click

from dunder_mifflin_harness.cli.app import ChatApp
from dunder_mifflin_harness.config import ConfigurationError, Settings
from dunder_mifflin_harness.runtime.approvals import ApprovalMode
from dunder_mifflin_harness.runtime.runner import RunCoordinator, UserMessage
from dunder_mifflin_harness.runtime.session import RunMode, SessionConfig, SessionState


async def run_prompt(
    prompt: str,
    *,
    settings: Settings | None = None,
    model: str | None = None,
) -> str:
    """Run a single prompt through the runtime coordinator."""
    resolved_settings = settings or Settings()
    session = SessionState(
        config=SessionConfig(
            model=model or resolved_settings.model,
            max_attachment_bytes=resolved_settings.max_attachment_bytes,
            shell_timeout_seconds=resolved_settings.shell_timeout_seconds,
            shell_max_output_chars=resolved_settings.shell_max_output_chars,
        )
    )
    coordinator = RunCoordinator(settings=resolved_settings, session=session)
    return await coordinator.submit_message(UserMessage(text=prompt))


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--model", "-m", default=None, help="Override the model name.")
@click.option(
    "--mode",
    type=click.Choice([item.value for item in RunMode]),
    default=None,
    help="Run mode for interactive sessions.",
)
@click.option(
    "--approval",
    "approval_mode",
    type=click.Choice([item.value for item in ApprovalMode]),
    default=None,
    help="Approval mode for agent-requested actions.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def _command(
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
    args: tuple[str, ...],
) -> None:
    """Run a prompt or start interactive chat."""
    settings = Settings()
    command_args = list(args)
    if command_args and command_args[0] in {"chat", "run"}:
        command_name = command_args[0]
        remaining, model, mode, approval_mode = _extract_inline_options(
            command_args[1:],
            model=model,
            mode=mode,
            approval_mode=approval_mode,
        )
        command_args = [command_name, *remaining]

    if command_args and command_args[0] == "chat":
        app = ChatApp(settings=settings)
        _apply_overrides(app, model=model, mode=mode, approval_mode=approval_mode)
        asyncio.run(app.run())
        return

    if command_args and command_args[0] == "run":
        command_args = command_args[1:]

    if not command_args:
        app = ChatApp(settings=settings)
        _apply_overrides(app, model=model, mode=mode, approval_mode=approval_mode)
        asyncio.run(app.run())
        return

    prompt = " ".join(command_args)
    output = asyncio.run(run_prompt(prompt, settings=settings, model=model))
    click.echo(output)


def _apply_overrides(
    app: ChatApp,
    *,
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
) -> None:
    if model:
        app.session.config.model = model
        app.coordinator.reset_agent()
    if mode:
        app.session.config.mode = RunMode(mode)
    if approval_mode:
        approval = ApprovalMode(approval_mode)
        app.session.config.approval_mode = approval
        app.approvals.mode = approval


def _extract_inline_options(
    args: list[str],
    *,
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
) -> tuple[list[str], str | None, str | None, str | None]:
    """Accept simple options after `chat` or `run` for ergonomic commands."""
    remaining: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in {"--model", "-m"} and index + 1 < len(args):
            model = args[index + 1]
            index += 2
            continue
        if item.startswith("--model="):
            model = item.split("=", 1)[1]
            index += 1
            continue
        if item == "--mode" and index + 1 < len(args):
            mode = args[index + 1]
            index += 2
            continue
        if item.startswith("--mode="):
            mode = item.split("=", 1)[1]
            index += 1
            continue
        if item == "--approval" and index + 1 < len(args):
            approval_mode = args[index + 1]
            index += 2
            continue
        if item.startswith("--approval="):
            approval_mode = item.split("=", 1)[1]
            index += 1
            continue
        remaining.append(item)
        index += 1
    return remaining, model, mode, approval_mode


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entrypoint that returns process-style exit codes."""
    try:
        _command.main(
            args=list(argv) if argv is not None else None,
            prog_name="harness",
            standalone_mode=False,
        )
    except ConfigurationError as exc:
        click.echo(str(exc), err=True)
        return 2
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
