import asyncio

from dunder_mifflin_harness.cli.commands import SlashCommandRegistry


def test_slash_command_registry_dispatches_handler() -> None:
    async def scenario() -> list[str]:
        seen: list[str] = []
        registry = SlashCommandRegistry()

        async def handler(args: str) -> None:
            seen.append(args)

        registry.register("model", handler, "Set model")
        dispatched = await registry.dispatch("model", "test")

        assert dispatched
        return seen

    assert asyncio.run(scenario()) == ["test"]


def test_slash_command_registry_reports_unknown() -> None:
    async def scenario() -> bool:
        registry = SlashCommandRegistry()
        return await registry.dispatch("missing")

    assert asyncio.run(scenario()) is False
