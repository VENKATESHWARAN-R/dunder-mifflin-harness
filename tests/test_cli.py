import importlib
from pathlib import Path

import pytest

from jac.cli import main as cli_main
from jac.cli import main as cli_public_main
from jac.config import ConfigurationError


def test_cli_prints_agent_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))

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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))

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


def test_cli_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["--version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("jac, version ")


def test_init_global_creates_user_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / ".jac"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("JAC_MODEL", raising=False)

    exit_code = cli_main(["init", "--global", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Initialized user workspace" in captured.out
    assert (config_dir / "settings.json").is_file()
    assert (config_dir / ".env").is_file()
    assert "PYDANTIC_AI_GATEWAY_API_KEY=" in (
        config_dir / ".env"
    ).read_text(encoding="utf-8")
    assert (config_dir / "history").is_dir()


def test_init_project_creates_agents_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))

    exit_code = cli_main(["init", "--yes", "--env-local"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Initialized project workspace" in captured.out
    assert (tmp_path / ".agents" / "settings.json").is_file()
    assert (tmp_path / ".agents" / ".env.local").is_file()
    assert ".agents/state.db" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_doctor_reports_missing_credentials_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("JAC_MODEL", "gateway/google-vertex:gemini")
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)

    exit_code = cli_main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "JAC Doctor" in captured.out
    assert "credential PYDANTIC_AI_GATEWAY_API_KEY: missing" in captured.out
