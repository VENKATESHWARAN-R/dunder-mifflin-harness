import importlib
import json
import asyncio
from pathlib import Path

import pytest

from jac.cli import main as cli_main
from jac.cli import main as cli_public_main
from jac.cli.app import ChatApp
from jac.config import ConfigurationError
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.runtime.coordinator import UserMessage
from jac.runtime.session import SessionConfig, SessionState


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

    exit_code = cli_main(["run", "say", "hi"])

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

    exit_code = cli_main(["run", "say hi"])

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


def test_prompt_requires_run_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_module = importlib.import_module("jac.cli.main")
    called = {"run_prompt": False}

    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        called["run_prompt"] = True
        return prompt

    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["say", "hi"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert 'Use `jac run "<prompt>"`' in captured.err
    assert called["run_prompt"] is False


def test_list_profiles_guides_to_profile_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_module = importlib.import_module("jac.cli.main")
    called = {"run_prompt": False}

    async def fake_run_prompt(prompt: str, **_kwargs: object) -> str:
        called["run_prompt"] = True
        return prompt

    monkeypatch.setattr(cli_module, "run_prompt", fake_run_prompt)

    exit_code = cli_main(["list", "profiles"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Did you mean `jac profile list`?" in captured.err
    assert called["run_prompt"] is False


def test_public_cli_exports_main() -> None:
    assert cli_public_main is cli_main


def test_resume_unknown_run_id_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))

    exit_code = cli_main(["resume", "no-such-run"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "no run with id no-such-run" in captured.err


def test_resume_with_no_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))

    # With no prior sessions, resume should fail with a clear message
    exit_code = cli_main(["resume"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "No prior sessions found" in captured.err


def test_cli_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["--version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("jac, version ")


def test_cli_help_lists_real_usage_patterns(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Usage patterns:" in captured.out
    assert "jac run " in captured.out
    assert "jac resume <run-id>" in captured.out
    assert "jac init [--global] [--yes]" in captured.out
    assert "Interactive chat shortcuts:" in captured.out
    assert 'jac "explain this repo"' not in captured.out


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
    assert "PYDANTIC_AI_GATEWAY_API_KEY=" in (config_dir / ".env").read_text(
        encoding="utf-8"
    )
    settings = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["active_profile"] == "default"
    assert "default" in settings["profiles"]
    assert settings["default_provider"] == "gateway"
    assert settings["model_tiers"]["worker"][0] == "gateway/anthropic:claude-sonnet-4-6"
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
    assert "provider: gateway" in captured.out
    assert "credential PYDANTIC_AI_GATEWAY_API_KEY: missing" in captured.out


def test_init_global_interactive_writes_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / ".jac"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cli_module = importlib.import_module("jac.cli.main")
    answers = iter(
        [
            "anthropic",
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-opus-4-6",
            "test-anthropic-key",
        ]
    )

    monkeypatch.setattr(cli_module.click, "prompt", lambda *_, **__: next(answers))

    exit_code = cli_main(["init", "--global"])

    captured = capsys.readouterr()
    settings = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    env_text = (config_dir / ".env").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Initialized user workspace" in captured.out
    assert settings["default_provider"] == "anthropic"
    assert settings["profiles"]["default"]["default_provider"] == "anthropic"
    assert settings["model_tiers"]["architect"] == ["anthropic:claude-opus-4-6"]
    assert "JAC_PROFILE_DEFAULT_ANTHROPIC_API_KEY=test-anthropic-key" in env_text
    assert "ANTHROPIC_API_KEY=test-anthropic-key" in env_text


def test_profile_use_switches_active_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / ".jac"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        """
{
  "active_profile": "home",
  "profiles": {
    "home": {
      "default_provider": "ollama",
      "default_tier": "worker",
      "model_tiers": {
        "worker": ["ollama:kimi-k2.6:cloud"]
      }
    },
    "office": {
      "default_provider": "litellm",
      "default_tier": "worker",
      "model_tiers": {
        "worker": ["litellm:openai/gpt-5.4-mini"]
      }
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))

    exit_code = cli_main(["profile", "use", "office"])

    captured = capsys.readouterr()
    settings = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Activated profile: office" in captured.out
    assert settings["active_profile"] == "office"
    assert settings["default_provider"] == "litellm"
    assert settings["model_tiers"]["worker"] == ["litellm:openai/gpt-5.4-mini"]


def test_profile_add_writes_profile_scoped_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / ".jac"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    cli_module = importlib.import_module("jac.cli.main")
    answers = iter(
        [
            "litellm",
            True,
            "litellm:openai/gpt-5.4-nano",
            "litellm:openai/gpt-5.4-mini",
            "litellm:openai/gpt-5.4",
            "https://litellm.example/v1",
            "office-key",
        ]
    )

    def fake_prompt(*_args: object, **_kwargs: object) -> object:
        return next(answers)

    def fake_confirm(*_args: object, **_kwargs: object) -> object:
        return next(answers)

    monkeypatch.setattr(cli_module.click, "prompt", fake_prompt)
    monkeypatch.setattr(cli_module.click, "confirm", fake_confirm)

    exit_code = cli_main(["profile", "add", "office"])

    captured = capsys.readouterr()
    settings = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    env_text = (config_dir / ".env").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Configured profile: office" in captured.out
    assert settings["active_profile"] == "office"
    assert settings["profiles"]["office"]["default_provider"] == "litellm"
    assert "JAC_PROFILE_OFFICE_LITELLM_API_BASE=https://litellm.example/v1" in env_text
    assert "JAC_PROFILE_OFFICE_LITELLM_API_KEY=office-key" in env_text


def test_chatapp_from_resumed_runs_seed_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class DummyState:
        async def close(self) -> None:
            return None

    class DummyCoordinator:
        def __init__(self) -> None:
            self.session = object()
            self.events = object()

    async def fake_open_state_store(_path: Path) -> DummyState:
        calls.append("open_state_store")
        return DummyState()

    async def fake_seed_workspace(_workspace: object, _state: object) -> None:
        calls.append("seed_workspace")

    async def fake_resume_run(**_kwargs: object) -> DummyCoordinator:
        calls.append("resume_run")
        return DummyCoordinator()

    monkeypatch.setattr("jac.cli.app.open_state_store", fake_open_state_store)
    monkeypatch.setattr("jac.cli.app.seed_workspace", fake_seed_workspace)
    monkeypatch.setattr("jac.cli.app.resume_run", fake_resume_run)
    monkeypatch.setattr(
        "jac.cli.app.discover_workspace",
        lambda _cwd: type("W", (), {"state_db_path": Path("/tmp/state.db")})(),
    )

    original_init = ChatApp.__init__
    seen: dict[str, object] = {}

    def fake_init(self, **kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(ChatApp, "__init__", fake_init)
    try:
        asyncio.run(ChatApp.from_resumed("run-1"))
    finally:
        monkeypatch.setattr(ChatApp, "__init__", original_init)

    assert calls == ["open_state_store", "seed_workspace", "resume_run"]
    assert seen["state"] is not None


def test_submit_message_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCoordinator:
        def __init__(self) -> None:
            self.calls = 0
            self.approval_policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

        async def submit_message(self, _message: UserMessage) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return "ok"

    class DummyPrompts:
        async def ask_yn(self, _prompt: str) -> bool:
            return True

    session = SessionState(config=SessionConfig())
    coordinator = DummyCoordinator()
    app = ChatApp(session=session, coordinator=coordinator)
    app.prompts = DummyPrompts()

    asyncio.run(app._submit_message(UserMessage(text="hello")))
    assert coordinator.calls == 2


def test_destructive_shell_command_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyPrompts:
        async def ask_yn(self, _prompt: str) -> bool:
            return False

    class DummyRenderer:
        def __init__(self) -> None:
            self.warnings: list[str] = []
            self.info: list[str] = []
            self.console = type("C", (), {"clear": lambda self: None})()

        def wire(self, _events: object) -> None:
            return None

        def print_warning(self, message: str) -> None:
            self.warnings.append(message)

        def print_info(self, message: str) -> None:
            self.info.append(message)

    called = {"run_shell": 0}

    async def fake_run_shell(**_kwargs: object):
        called["run_shell"] += 1
        raise AssertionError("should not execute destructive command")

    session = SessionState(config=SessionConfig())
    app = ChatApp(session=session, renderer=DummyRenderer())
    app.prompts = DummyPrompts()
    monkeypatch.setattr("jac.cli.app.run_shell", fake_run_shell)

    asyncio.run(app._handle_shell("rm -rf /tmp/demo"))
    assert called["run_shell"] == 0


def test_undo_command_restores_file(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("original", encoding="utf-8")

    session = SessionState(config=SessionConfig())
    app = ChatApp(session=session)
    app._undo_stack.append((path, b"original"))
    path.write_text("changed", encoding="utf-8")

    dispatched = asyncio.run(app.commands.dispatch("undo", ""))

    assert dispatched is True
    assert path.read_text(encoding="utf-8") == "original"
