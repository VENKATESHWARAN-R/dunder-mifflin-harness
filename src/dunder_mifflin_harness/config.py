"""Configuration for the dunder-mifflin-harness."""

from pydantic import Field
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

    model: str = Field(
        default="google-gla:gemini-2.5-flash",
        validation_alias="HARNESS_MODEL",
    )
