import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest

from dunder_mifflin_harness.cli import main as cli_main
from dunder_mifflin_harness.cli import main as cli_public_main
from dunder_mifflin_harness.config import ConfigurationError


def test_cli_prints_agent_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}

    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        seen["prompt"] = prompt
        return "hi from the model"

    cli_module = importlib.import_module("dunder_mifflin_harness.cli.main")

    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["say", "hi"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["prompt"] == "say hi"
    assert captured.out == "hi from the model\n"
    assert captured.err == ""


def test_cli_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        raise ConfigurationError("GEMINI_API_KEY is required to call Gemini.")

    cli_module = importlib.import_module("dunder_mifflin_harness.cli.main")

    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["say hi"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "GEMINI_API_KEY is required" in captured.err


def test_run_accepts_mode_after_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}

    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        seen["prompt"] = prompt
        return "ok"

    cli_module = importlib.import_module("dunder_mifflin_harness.cli.main")
    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["run", "--mode", "autopilot", "say", "hi"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["prompt"] == "say hi"
    assert captured.out == "ok\n"


def test_run_prompt_loads_file_attachments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hello from attachment", encoding="utf-8")
    seen: dict[str, str] = {}

    class FakeCoordinator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, message: Any) -> str:
            seen["prompt"] = message.as_prompt()
            return "ok"

    cli_module = importlib.import_module("dunder_mifflin_harness.cli.main")
    monkeypatch.setattr(cli_module, "RunCoordinator", FakeCoordinator)
    monkeypatch.chdir(tmp_path)

    output = asyncio.run(cli_module.run_prompt("summarize @note.txt"))

    assert output == "ok"
    assert "Attached files:" in seen["prompt"]
    assert "hello from attachment" in seen["prompt"]


def test_public_cli_exports_main() -> None:
    assert cli_public_main is cli_main
