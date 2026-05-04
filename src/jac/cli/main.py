"""Click entrypoints for the JAC CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import click

from jac import __version__
from jac.cli.app import ChatApp
from jac.config import ConfigurationError, Settings
from jac.config import (
    DEFAULT_PROVIDER,
    PROVIDER_DEFINITIONS,
    TIER_NAMES,
    default_model_tiers,
    provider_definition,
)
from jac.onboarder import (
    DEFAULT_MODEL,
    activate_global_profile,
    configure_global_profile,
    doctor_report,
    init_global_workspace,
    init_project_workspace,
)
from jac.runtime.approvals import ApprovalMode
from jac.runtime.coordinator import RunCoordinator, UserMessage
from jac.runtime.session import RunMode, SessionConfig, SessionState
from jac.state import open_state_store, seed_workspace
from jac.workspace import default_user_dir, discover_workspace


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
            model=model,
            max_attachment_bytes=resolved_settings.max_attachment_bytes,
            shell_timeout_seconds=resolved_settings.shell_timeout_seconds,
            shell_max_output_chars=resolved_settings.shell_max_output_chars,
        )
    )
    workspace = discover_workspace(session.config.cwd)
    state = await open_state_store(workspace.state_db_path)
    await seed_workspace(workspace, state)
    try:
        coordinator = RunCoordinator(
            settings=resolved_settings, session=session, state=state
        )
        output = await coordinator.submit_message(UserMessage(text=prompt))
        await state.runs.update_status(session.run_id, "done")
        return output
    finally:
        await state.close()


HELP_TEXT = "Run prompts, interactive chat, and workspace/profile tooling."
HELP_EPILOG = """\
\b
Usage patterns:
  jac
  jac run "write tests for parser"
  jac resume <run-id>
  jac init [--global] [--yes]
  jac profile [current|list|use|add]
  jac doctor
  jac config

\b
Interactive chat shortcuts:
  /help, /model, /tier, /mode, /approval, /params, /context, /cost, /quit

\b
Runtime override examples:
  jac --model <name> --mode hitl --approval interactive
  jac run --model <name> --mode autopilot "summarize this file"
"""

KNOWN_TOP_LEVEL_COMMANDS = {"run", "chat", "resume", "init", "profile", "doctor", "config"}


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help=HELP_TEXT,
    epilog=HELP_EPILOG,
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Override the model name for this invocation.",
)
@click.version_option(version=__version__, prog_name="jac")
@click.option(
    "--mode",
    type=click.Choice([item.value for item in RunMode]),
    default=None,
    help="Run mode override (useful with chat/resume).",
)
@click.option(
    "--approval",
    "approval_mode",
    type=click.Choice([item.value for item in ApprovalMode]),
    default=None,
    help="Approval mode override for agent-requested actions.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def _command(
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
    args: tuple[str, ...],
) -> None:
    """Run a prompt or dispatch CLI subcommands."""
    settings = Settings()
    command_args = list(args)
    if command_args and command_args[0] in {"chat", "run", "resume"}:
        command_name = command_args[0]
        remaining, model, mode, approval_mode = _extract_inline_options(
            command_args[1:],
            model=model,
            mode=mode,
            approval_mode=approval_mode,
        )
        command_args = [command_name, *remaining]

    if not command_args:
        asyncio.run(
            _run_chat(
                settings=settings,
                model=model,
                mode=mode,
                approval_mode=approval_mode,
            )
        )
        return

    command = command_args[0]

    if command == "init":
        _run_init(command_args[1:], settings=settings)
        return

    if command == "profile":
        _run_profile(command_args[1:], settings=settings)
        return

    if command in {"doctor", "config"}:
        click.echo(doctor_report(settings))
        return

    if command == "chat":
        asyncio.run(
            _run_chat(
                settings=settings,
                model=model,
                mode=mode,
                approval_mode=approval_mode,
            )
        )
        return

    if command == "resume":
        if len(command_args) < 2:
            raise click.ClickException("Usage: jac resume <run-id>")
        run_id = command_args[1]
        try:
            asyncio.run(
                _run_resume(
                    run_id=run_id,
                    settings=settings,
                    model=model,
                    mode=mode,
                    approval_mode=approval_mode,
                )
            )
        except LookupError as exc:
            raise click.ClickException(str(exc)) from exc
        return

    if command != "run":
        raise click.ClickException(_unknown_command_message(command_args))
    prompt_args = command_args[1:]
    if not prompt_args:
        raise click.ClickException('Usage: jac run "<prompt>"')
    prompt = " ".join(prompt_args)
    output = asyncio.run(run_prompt(prompt, settings=settings, model=model))
    click.echo(output)


def _unknown_command_message(args: list[str]) -> str:
    command = args[0]
    if command == "list" and len(args) >= 2 and args[1] == "profiles":
        return 'Unknown command "list profiles". Did you mean `jac profile list`?'
    known = ", ".join(sorted(KNOWN_TOP_LEVEL_COMMANDS))
    return (
        f'Unknown command "{command}". '
        f"Top-level commands: {known}. "
        'Use `jac run "<prompt>"` for one-shot prompts.'
    )


async def _run_chat(
    *,
    settings: Settings,
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
) -> None:
    app = await ChatApp.open(settings=settings)
    _apply_overrides(app, model=model, mode=mode, approval_mode=approval_mode)
    try:
        await app.run()
    finally:
        await app.aclose()


async def _run_resume(
    *,
    run_id: str,
    settings: Settings,
    model: str | None,
    mode: str | None,
    approval_mode: str | None,
) -> None:
    app = await ChatApp.from_resumed(run_id, settings=settings)
    _apply_overrides(app, model=model, mode=mode, approval_mode=approval_mode)
    try:
        await app.run()
    finally:
        await app.aclose()


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
        provider = DEFAULT_PROVIDER
        model_tiers = default_model_tiers(provider)
        env_values: dict[str, str] | None = None
        if not yes:
            provider = click.prompt(
                "Provider",
                default=settings.default_provider or DEFAULT_PROVIDER,
                type=click.Choice(list(PROVIDER_DEFINITIONS)),
            )
            model_tiers = _prompt_model_tiers(provider)
            env_values = _prompt_provider_env(provider)
        result = init_global_workspace(
            user_dir=default_user_dir(),
            provider=provider,
            model_tiers=model_tiers,
            env_values=env_values,
            model=settings.model or DEFAULT_MODEL,
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
        provider=settings.default_provider or DEFAULT_PROVIDER,
        model_tiers=settings.model_tiers
        or default_model_tiers(settings.default_provider or DEFAULT_PROVIDER),
        model=settings.model or DEFAULT_MODEL,
    )
    _print_init_result("Initialized project workspace", result.created, result.updated)


def _run_profile(args: list[str], *, settings: Settings) -> None:
    subcommand = args[0] if args else "current"
    if subcommand in {"current", "show"}:
        active = settings.active_profile or "(none)"
        click.echo(f"Active profile: {active}")
        click.echo(f"Provider: {settings.default_provider}")
        click.echo(f"Default tier: {settings.default_tier}")
        return

    if subcommand == "list":
        if not settings.profiles:
            click.echo("No profiles configured. Run `jac profile add <name>`.")
            return
        for name, profile in settings.profiles.items():
            marker = "*" if name == settings.active_profile else " "
            provider = profile.get("default_provider", "(unknown)")
            tier = profile.get("default_tier", "(unknown)")
            click.echo(f"{marker} {name}: provider={provider} tier={tier}")
        return

    if subcommand == "use":
        if len(args) < 2:
            raise click.ClickException("Usage: jac profile use <name>")
        try:
            result = activate_global_profile(
                user_dir=default_user_dir(),
                profile=args[1],
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        _print_init_result(
            f"Activated profile: {args[1]}", result.created, result.updated
        )
        return

    if subcommand == "add":
        if len(args) < 2:
            raise click.ClickException(
                "Usage: jac profile add <name> [--provider <id>]"
            )
        profile = args[1]
        remaining = args[2:]
        provider = _option_value(remaining, "--provider") or DEFAULT_PROVIDER
        activate = "--activate" in remaining
        yes = "--yes" in remaining or "-y" in remaining
        if not yes:
            provider = click.prompt(
                "Provider",
                default=provider,
                type=click.Choice(list(PROVIDER_DEFINITIONS)),
            )
            activate = click.confirm("Activate this profile now?", default=activate)
        model_tiers = (
            default_model_tiers(provider) if yes else _prompt_model_tiers(provider)
        )
        env_values = None if yes else _prompt_provider_env(provider)
        result = configure_global_profile(
            user_dir=default_user_dir(),
            profile=profile,
            provider=provider,
            model_tiers=model_tiers,
            env_values=env_values,
            activate=activate,
        )
        _print_init_result(
            f"Configured profile: {profile}", result.created, result.updated
        )
        return

    raise click.ClickException(
        "Usage: jac profile [current|list|use <name>|add <name>]"
    )


def _print_init_result(title: str, created: list[Path], updated: list[Path]) -> None:
    click.echo(title)
    for path in created:
        click.echo(f"created: {path}")
    for path in updated:
        click.echo(f"updated: {path}")
    if not created and not updated:
        click.echo("already up to date")


def _prompt_model_tiers(provider: str) -> dict[str, list[str]]:
    definition = provider_definition(provider)
    click.echo(f"Suggested tier models for {definition.label}:")
    tiers: dict[str, list[str]] = {}
    defaults = default_model_tiers(provider)
    for tier in TIER_NAMES:
        default = ", ".join(defaults[tier])
        click.echo(f"- {tier}: {default}")
        raw_value = click.prompt(
            f"{tier.capitalize()} models (comma-separated)",
            default=default,
            show_default=False,
        )
        tiers[tier] = _split_model_list(raw_value)
    return tiers


def _prompt_provider_env(provider: str) -> dict[str, str]:
    values: dict[str, str] = {}
    definition = provider_definition(provider)
    for item in definition.env:
        values[item.name] = click.prompt(
            item.prompt,
            default=item.default,
            hide_input=item.secret,
            show_default=bool(item.default and not item.secret),
        )
    return values


def _split_model_list(raw_value: str) -> list[str]:
    models = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not models:
        raise click.ClickException("Each tier must include at least one model.")
    return models


def _option_value(args: list[str], name: str) -> str | None:
    for index, item in enumerate(args):
        if item == name and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


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
