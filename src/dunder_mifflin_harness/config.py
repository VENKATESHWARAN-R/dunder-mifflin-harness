"""Configuration for the dunder-mifflin-harness."""
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when the CLI cannot be configured for a live LLM call."""


class Settings(BaseSettings):
    """Settings for the dunder-mifflin-harness."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    model: str = Field(
        default="google-gla:gemini-2.5-flash",
        validation_alias="HARNESS_MODEL",
    )

    def require_gemini_api_key(self) -> str:
        """Require the Gemini API key."""
        key: SecretStr | None = self.gemini_api_key
        if key is None:
            raise ConfigurationError("GEMINI_API_KEY is required to call Gemini.")
        return key.get_secret_value() # pylint: disable=no-member
