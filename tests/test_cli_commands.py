import asyncio
import io
from pathlib import Path

from rich.console import Console

from dunder_mifflin_harness.cli import app as app_module
from dunder_mifflin_harness.cli.commands import SlashCommandRegistry
from dunder_mifflin_harness.cli.renderer import Renderer
from dunder_mifflin_harness.config import Settings
from dunder_mifflin_harness.runtime.coordinator import UserMessage


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


class _FakeInput:
    def __init__(self, values: list[str | None]) -> None:
        self._values = values
        self.calls = 0

    async def read(self) -> str | None:
        self.calls += 1
        if not self._values:
            raise AssertionError("chat loop continued after input ended")
        return self._values.pop(0)


def _make_chat_app(
    monkeypatch,
    tmp_path: Path,
    input_session: _FakeInput | None = None,
) -> app_module.ChatApp:
    if input_session is None:
        input_session = _FakeInput([])
    monkeypatch.setattr(app_module, "InputSession", lambda _history_path: input_session)
    console = Console(file=io.StringIO(), force_terminal=False, width=80)
    return app_module.ChatApp(
        settings=Settings(config_dir=tmp_path),
        renderer=Renderer(console),
    )


def test_chat_app_exits_on_input_eof(monkeypatch, tmp_path: Path) -> None:
    input_session = _FakeInput([None])
    app = _make_chat_app(monkeypatch, tmp_path, input_session)

    asyncio.run(app.run())

    assert input_session.calls == 1


def test_chat_app_keeps_running_after_model_failure(monkeypatch, tmp_path: Path) -> None:
    class FailingCoordinator:
        def __init__(self) -> None:
            self.messages: list[UserMessage] = []

        async def submit_message(self, message: UserMessage) -> str:
            self.messages.append(message)
            raise RuntimeError("backend unavailable")

    app = _make_chat_app(monkeypatch, tmp_path)
    coordinator = FailingCoordinator()
    app.coordinator = coordinator

    asyncio.run(app.handle_input("hello"))

    assert [message.text for message in coordinator.messages] == ["hello"]


def test_params_rejects_invalid_temperature(monkeypatch, tmp_path: Path) -> None:
    app = _make_chat_app(monkeypatch, tmp_path)

    asyncio.run(app.handle_input("/params temperature nope"))

    assert "temperature" not in app.session.config.model_params
