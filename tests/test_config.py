from pathlib import Path

import pytest

from dunder_mifflin_harness.config import ConfigurationError, Settings


def test_settings_use_w0_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_MODEL", raising=False)

    settings = Settings()

    assert settings.model == "google-gla:gemini-2.5-flash"


def test_require_gemini_api_key_returns_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = Settings()

    assert settings.require_gemini_api_key() == "test-key"


def test_require_gemini_api_key_fails_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings()

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        settings.require_gemini_api_key()
