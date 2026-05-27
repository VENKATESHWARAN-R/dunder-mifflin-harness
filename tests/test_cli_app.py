import asyncio
import io
from pathlib import Path

from rich.console import Console

from dunder_mifflin_harness.cli.app import ChatApp
from dunder_mifflin_harness.cli.renderer import Renderer
from dunder_mifflin_harness.config import Settings
from dunder_mifflin_harness.runtime.coordinator import UserMessage


def _app(tmp_path: Path) -> ChatApp:
    renderer = Renderer(Console(file=io.StringIO()))
    return ChatApp(settings=Settings(config_dir=tmp_path), renderer=renderer)


def test_chat_loop_exits_on_eof(tmp_path: Path) -> None:
    class EofInput:
        calls = 0

        async def read(self) -> str | None:
            self.calls += 1
            return None

    app = _app(tmp_path)
    input_session = EofInput()
    app.input = input_session  # type: ignore[assignment]

    asyncio.run(app.run())

    assert input_session.calls == 1
    assert app._should_exit


def test_chat_input_keeps_session_alive_after_run_failure(tmp_path: Path) -> None:
    class FailingCoordinator:
        submitted: UserMessage | None = None

        async def submit_message(self, message: UserMessage) -> str:
            self.submitted = message
            raise RuntimeError("model unavailable")

    app = _app(tmp_path)
    coordinator = FailingCoordinator()
    app.coordinator = coordinator  # type: ignore[assignment]

    asyncio.run(app.handle_input("hello"))

    assert coordinator.submitted == UserMessage(text="hello")
    assert not app._should_exit
