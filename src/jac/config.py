"""Configuration for JAC."""

import os
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from jac.workspace import load_workspace_env


class ConfigurationError(RuntimeError):
    """Raised when the CLI cannot be configured for a live LLM call."""


def credential_candidates_for_model(model: str) -> tuple[str, ...]:
    """Return credential env vars required by a model id."""
    normalized = model.lower()
    if normalized.startswith("ollama:"):
        return ()
    if normalized.startswith("gateway/"):
        return ("PYDANTIC_AI_GATEWAY_API_KEY",)
    if normalized.startswith(("openai:", "gpt-")):
        return ("OPENAI_API_KEY",)
    if normalized.startswith("anthropic:"):
        return ("ANTHROPIC_API_KEY",)
    if normalized.startswith("openrouter:"):
        return ("OPENROUTER_API_KEY",)
    if normalized.startswith(("google-gla:", "google-vertex:")) or "gemini" in normalized:
        return ("GEMINI_API_KEY",)
    if normalized.startswith("litellm:"):
        return ("LITELLM_API_KEY",)
    return ("PYDANTIC_AI_GATEWAY_API_KEY", "GEMINI_API_KEY")


class Settings(BaseSettings):
    """Settings for JAC."""

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    def __init__(self, **values: Any) -> None:
        """Load workspace dotenv files before BaseSettings reads env vars."""
        load_workspace_env()
        super().__init__(**values)

    model: str = Field(
        description="The model to use for JAC.",
        default="gateway/google-vertex:gemini-3.1-flash-lite-preview",
        validation_alias="JAC_MODEL",
    )

    api_key: SecretStr | None = Field(
        description="The API key used by the configured Pydantic AI provider.",
        default=None,
        validation_alias=AliasChoices(
            "PYDANTIC_AI_GATEWAY_API_KEY",
            "GEMINI_API_KEY",
        ),
    )

    config_dir: Path = Field(
        description="Directory for CLI-local state such as prompt history.",
        default=Path.home() / ".jac",
        validation_alias="JAC_CONFIG_DIR",
    )

    max_attachment_bytes: int = Field(
        description="Maximum size for a single @ file attachment.",
        default=200_000,
        validation_alias="JAC_MAX_ATTACHMENT_BYTES",
    )

    shell_timeout_seconds: float = Field(
        description="Timeout for user-triggered shell commands.",
        default=10.0,
        validation_alias="JAC_SHELL_TIMEOUT_SECONDS",
    )

    shell_max_output_chars: int = Field(
        description="Maximum captured stdout/stderr characters shown per stream.",
        default=20_000,
        validation_alias="JAC_SHELL_MAX_OUTPUT_CHARS",
    )

    def require_api_key(self, name: str = "PYDANTIC_AI_GATEWAY_API_KEY") -> str:
        """Return the configured API key or raise a CLI-friendly error."""
        value = os.getenv(name)
        if value:
            return value
        if self.api_key is None:
            raise ConfigurationError(f"{name} is required to call the configured model.")
        return self.api_key.get_secret_value()

    def require_gemini_api_key(self) -> str:
        """Backward-compatible helper for the original W0 Gemini tests."""
        if key := os.getenv("GEMINI_API_KEY"):
            return key
        return self.require_api_key("GEMINI_API_KEY")

    def require_model_credentials(self, model: str | None = None) -> None:
        """Raise a CLI-friendly error if the selected model needs missing credentials."""
        selected_model = model or self.model
        candidates = credential_candidates_for_model(selected_model)
        if not candidates:
            return
        if any(os.getenv(name) for name in candidates):
            return

        names = " or ".join(candidates)
        raise ConfigurationError(
            f"Model '{selected_model}' requires {names}. "
            "Run `jac init --global` or set the env var before making an LLM call."
        )
