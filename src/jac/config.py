"""Configuration for JAC."""

import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when the CLI cannot be configured for a live LLM call."""


class Settings(BaseSettings):
    """Settings for JAC."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

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
        if self.api_key is None:
            raise ConfigurationError(f"{name} is required to call the configured model.")
        return self.api_key.get_secret_value()

    def require_gemini_api_key(self) -> str:
        """Backward-compatible helper for the original W0 Gemini tests."""
        if key := os.getenv("GEMINI_API_KEY"):
            return key
        return self.require_api_key("GEMINI_API_KEY")
