"""Settings for JAC — env-backed, minimal for M1 Slice B.

Slice B only resolves provider credentials and a default tier. The per-run
SQLite override branch of the three-tier model resolution is stubbed in
`agents/base.py`; user-scope JSON settings live at `~/.jac/settings.json`
and are read by the factory directly. This module is the floor everything
else depends on, so it stays small and pure — no Pydantic AI, no SQLite,
no `jac.*` imports beyond `workspace`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


Tier = Literal["scout", "worker", "architect"]


class ConfigurationError(RuntimeError):
    """Raised when a required setting is missing for a live LLM call."""


class Settings(BaseSettings):
    """JAC environment-backed settings."""

    model_config = SettingsConfigDict(
        env_prefix="JAC_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    default_tier: Tier = Field(default="worker", validation_alias="JAC_DEFAULT_TIER")

    def require_anthropic_api_key(self) -> str:
        if self.anthropic_api_key is None:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell or .env "
                "before running JAC against Anthropic models."
            )
        return self.anthropic_api_key.get_secret_value()
