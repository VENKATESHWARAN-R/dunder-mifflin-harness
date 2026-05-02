from pathlib import Path

import pytest

from jac.config import ConfigurationError, Settings


def test_settings_use_w0_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("JAC_MODEL", raising=False)

    settings = Settings()

    assert settings.model == "gateway/google-vertex:gemini-3.1-flash-lite-preview"


def test_require_gemini_api_key_returns_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = Settings()

    assert settings.require_gemini_api_key() == "test-key"


def test_require_gemini_api_key_fails_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)

    settings = Settings()

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        settings.require_gemini_api_key()


def test_cli_settings_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JAC_MAX_ATTACHMENT_BYTES", "100")
    monkeypatch.setenv("JAC_SHELL_TIMEOUT_SECONDS", "2.5")

    settings = Settings()

    assert settings.max_attachment_bytes == 100
    assert settings.shell_timeout_seconds == 2.5


def test_settings_load_user_global_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".jac"
    config_dir.mkdir()
    (config_dir / ".env").write_text(
        "JAC_MODEL=ollama:llama3\nPYDANTIC_AI_GATEWAY_API_KEY=user-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("JAC_MODEL", raising=False)
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)

    settings = Settings()

    assert settings.model == "ollama:llama3"
    assert settings.require_api_key() == "user-key"


def test_project_env_overrides_user_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".jac"
    project_dir = tmp_path / "project"
    agents_dir = project_dir / ".agents"
    config_dir.mkdir()
    agents_dir.mkdir(parents=True)
    (config_dir / ".env").write_text(
        "JAC_MODEL=gateway/google-vertex:global\n"
        "PYDANTIC_AI_GATEWAY_API_KEY=user-key\n",
        encoding="utf-8",
    )
    (agents_dir / ".env.local").write_text(
        "JAC_MODEL=ollama:project\nPYDANTIC_AI_GATEWAY_API_KEY=project-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("JAC_MODEL", raising=False)
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)

    settings = Settings()

    assert settings.model == "ollama:project"
    assert settings.require_api_key() == "project-key"


def test_process_env_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".jac"
    config_dir.mkdir()
    (config_dir / ".env").write_text("JAC_MODEL=ollama:file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("JAC_MODEL", "ollama:process")

    settings = Settings()

    assert settings.model == "ollama:process"


def test_require_model_credentials_reports_missing_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("JAC_MODEL", "gateway/google-vertex:gemini")

    settings = Settings()

    with pytest.raises(ConfigurationError, match="PYDANTIC_AI_GATEWAY_API_KEY"):
        settings.require_model_credentials()


def test_ollama_model_requires_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("JAC_MODEL", "ollama:llama3")
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    Settings().require_model_credentials()
