from pathlib import Path

import pytest

from jac.config import ConfigurationError, ModelSelection, Settings
from jac.runtime.models import build_pydantic_model


def test_gateway_model_uses_model_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "gateway-key")
    selection = ModelSelection(
        model_ref="gateway/anthropic:claude-sonnet-4-6",
        provider="gateway",
        source="test",
    )

    assert build_pydantic_model(selection, Settings()) == selection.model_ref


def test_openai_model_factory_builds_provider_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    selection = ModelSelection(
        model_ref="openai:gpt-5.4-mini",
        provider="openai",
        source="test",
    )

    model = build_pydantic_model(selection, Settings())

    assert type(model).__name__ == "OpenAIChatModel"
    assert model.model_name == "gpt-5.4-mini"


def test_ollama_model_factory_uses_openai_compatible_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    selection = ModelSelection(
        model_ref="ollama:qwen3.5:397b-cloud",
        provider="ollama",
        source="test",
    )

    model = build_pydantic_model(selection, Settings())

    assert type(model).__name__ == "OpenAIChatModel"
    assert model.model_name == "qwen3.5:397b-cloud"


def test_litellm_model_factory_strips_jac_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.setenv("LITELLM_API_BASE", "http://litellm.test/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")
    selection = ModelSelection(
        model_ref="litellm:openai/gpt-5.4",
        provider="litellm",
        source="test",
    )

    model = build_pydantic_model(selection, Settings())

    assert type(model).__name__ == "OpenAIChatModel"
    assert model.model_name == "openai/gpt-5.4"


def test_model_factory_reports_missing_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAC_CONFIG_DIR", str(tmp_path / ".jac"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selection = ModelSelection(
        model_ref="openai:gpt-5.4",
        provider="openai",
        source="test",
    )

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_pydantic_model(selection, Settings())
