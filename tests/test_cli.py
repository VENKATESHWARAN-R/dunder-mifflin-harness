import pytest

from dunder_mifflin_harness import cli
from dunder_mifflin_harness.config import ConfigurationError


def test_cli_prints_agent_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}

    async def fake_run_prompt(prompt: str) -> str:
        seen["prompt"] = prompt
        return "hi from the model"

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    exit_code = cli.main(["say", "hi"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["prompt"] == "say hi"
    assert captured.out == "hi from the model\n"
    assert captured.err == ""


def test_cli_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_prompt(prompt: str) -> str:
        raise ConfigurationError("GEMINI_API_KEY is required to call Gemini.")

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    exit_code = cli.main(["say hi"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "GEMINI_API_KEY is required" in captured.err
