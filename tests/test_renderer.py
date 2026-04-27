import asyncio

from rich.console import Console

from dunder_mifflin_harness.cli.renderer import Renderer
from dunder_mifflin_harness.runtime.events import (
    AgentMessageCompleted,
    AgentTextDelta,
    EventBus,
    WarningRaised,
)


def test_renderer_buffers_agent_text_until_message_complete() -> None:
    async def scenario() -> str:
        console = Console(record=True, force_terminal=False, width=80)
        renderer = Renderer(console)
        bus = EventBus()
        renderer.wire(bus)

        await bus.emit(AgentTextDelta(text="hello"))
        assert console.export_text() == ""

        await bus.emit(AgentMessageCompleted(message="hello"))
        return console.export_text()

    output = asyncio.run(scenario())

    assert "hello" in output


def test_renderer_prints_warnings() -> None:
    async def scenario() -> str:
        console = Console(record=True, force_terminal=False, width=80)
        renderer = Renderer(console)
        bus = EventBus()
        renderer.wire(bus)

        await bus.emit(WarningRaised(message="careful"))
        return console.export_text()

    output = asyncio.run(scenario())

    assert "warning:" in output
    assert "careful" in output
