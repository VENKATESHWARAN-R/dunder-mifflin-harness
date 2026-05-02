"""Click entrypoints for the JAC CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import click

from jac import __version__
from jac.cli.app import ChatApp
from jac.config import ConfigurationError, Settings
from jac.onboarder import (
    DEFAULT_MODEL,
    doctor_report,
    init_global_workspace,
    init_project_workspace,
)
from jac.runtime.approvals import ApprovalMode
from jac.runtime.coordinator import RunCoordinator, UserMessage
from jac.runtime.session import RunMode, SessionConfig, SessionState
from jac.workspace import default_user_dir


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
@click.version_option(version=__version__, prog_name="jac")
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

    if command_args and command_args[0] == "init":
        _run_init(command_args[1:], settings=settings)
        return

    if command_args and command_args[0] in {"doctor", "config"}:
        click.echo(doctor_report(settings))
        return

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


def _run_init(args: list[str], *, settings: Settings) -> None:
    global_scope = "--global" in args
    yes = "--yes" in args or "-y" in args
    create_env_local = "--env-local" in args

    if global_scope:
        model = settings.model or DEFAULT_MODEL
        gateway_key: str | None = None
        if not yes:
            model = click.prompt("Default model", default=model)
            gateway_key = click.prompt(
                "PYDANTIC_AI_GATEWAY_API_KEY",
                default="",
                hide_input=True,
                show_default=False,
            )
        result = init_global_workspace(
            user_dir=default_user_dir(),
            model=model,
            gateway_api_key=gateway_key,
        )
        _print_init_result("Initialized user workspace", result.created, result.updated)
        return

    if not yes:
        create_env_local = click.confirm(
            "Create project-local .agents/.env.local?",
            default=create_env_local,
        )

    result = init_project_workspace(
        cwd=Path.cwd(),
        create_env_local=create_env_local,
        model=settings.model or DEFAULT_MODEL,
    )
    _print_init_result("Initialized project workspace", result.created, result.updated)


def _print_init_result(title: str, created: list[Path], updated: list[Path]) -> None:
    click.echo(title)
    for path in created:
        click.echo(f"created: {path}")
    for path in updated:
        click.echo(f"updated: {path}")
    if not created and not updated:
        click.echo("already up to date")


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entrypoint that returns process-style exit codes."""
    try:
        _command.main(
            args=list(argv) if argv is not None else None,
            prog_name="jac",
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
