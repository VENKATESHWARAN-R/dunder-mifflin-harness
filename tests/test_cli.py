import importlib

import pytest

from jac.cli import main as cli_main
from jac.cli import main as cli_public_main
from jac.config import ConfigurationError


def test_cli_prints_agent_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}

    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        seen["prompt"] = prompt
        return "hi from the model"

    cli_module = importlib.import_module("jac.cli.main")

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

    cli_module = importlib.import_module("jac.cli.main")

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

    cli_module = importlib.import_module("jac.cli.main")
    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["run", "--mode", "autopilot", "say", "hi"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["prompt"] == "say hi"
    assert captured.out == "ok\n"


def test_public_cli_exports_main() -> None:
    assert cli_public_main is cli_main
