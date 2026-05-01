from pathlib import Path

import pytest

from jac.config import ConfigurationError, Settings


def test_settings_use_w0_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
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
