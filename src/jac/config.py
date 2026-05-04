"""Configuration for JAC."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from jac.workspace import discover_workspace, load_workspace_env


class ConfigurationError(RuntimeError):
    """Raised when the CLI cannot be configured for a live LLM call."""


TIER_NAMES = ("scout", "worker", "architect")
DEFAULT_PROVIDER = "gateway"
DEFAULT_TIER = "worker"
DEFAULT_MODEL = "gateway/google-vertex:gemini-3.1-flash-lite-preview"


@dataclass(frozen=True, slots=True)
class EnvRequirement:
    """Provider environment variable requested by onboarding and runtime."""

    name: str
    prompt: str
    required: bool = True
    secret: bool = True
    default: str = ""


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Static metadata for a supported model provider."""

    provider_id: str
    label: str
    tier_defaults: dict[str, tuple[str, ...]]
    env: tuple[EnvRequirement, ...]
    custom_models: bool = False


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Resolved provider/model pair used to build a live Pydantic AI model."""

    model_ref: str
    provider: str
    source: str


PROFILE_CONFIG_KEYS = (
    "default_provider",
    "default_tier",
    "model_tiers",
    "model_overrides",
)


PROVIDER_DEFINITIONS: dict[str, ProviderDefinition] = {
    "gateway": ProviderDefinition(
        provider_id="gateway",
        label="Pydantic AI Gateway",
        tier_defaults={
            "scout": (
                "gateway/google-vertex:gemini-3.1-flash-lite-preview",
                "gateway/openai:gpt-5.4-nano",
                "gateway/anthropic:claude-haiku-4-5",
            ),
            "worker": (
                "gateway/anthropic:claude-sonnet-4-6",
                "gateway/openai:gpt-5.4-mini",
                "gateway/google-vertex:gemini-3-flash-preview",
            ),
            "architect": (
                "gateway/anthropic:claude-opus-4-6",
                "gateway/openai:gpt-5.4",
                "gateway/google-vertex:gemini-3.1-pro-preview",
            ),
        },
        env=(
            EnvRequirement(
                "PYDANTIC_AI_GATEWAY_API_KEY",
                "Pydantic AI Gateway API key",
            ),
        ),
    ),
    "anthropic": ProviderDefinition(
        provider_id="anthropic",
        label="Anthropic",
        tier_defaults={
            "scout": ("anthropic:claude-haiku-4-5",),
            "worker": ("anthropic:claude-sonnet-4-6",),
            "architect": ("anthropic:claude-opus-4-6",),
        },
        env=(EnvRequirement("ANTHROPIC_API_KEY", "Anthropic API key"),),
    ),
    "openai": ProviderDefinition(
        provider_id="openai",
        label="OpenAI",
        tier_defaults={
            "scout": ("openai:gpt-5.4-nano",),
            "worker": ("openai:gpt-5.4-mini",),
            "architect": ("openai:gpt-5.4",),
        },
        env=(EnvRequirement("OPENAI_API_KEY", "OpenAI API key"),),
    ),
    "google-gla": ProviderDefinition(
        provider_id="google-gla",
        label="Google AI Studio",
        tier_defaults={
            "scout": ("google-gla:gemini-3.1-flash-lite-preview",),
            "worker": ("google-gla:gemini-3-flash-preview",),
            "architect": ("google-gla:gemini-3.1-pro-preview",),
        },
        env=(EnvRequirement("GEMINI_API_KEY", "Google API key"),),
    ),
    "ollama": ProviderDefinition(
        provider_id="ollama",
        label="Ollama",
        tier_defaults={
            "scout": ("ollama:gemma4:31b-cloud",),
            "worker": ("ollama:qwen3.5:397b-cloud",),
            "architect": ("ollama:kimi-k2.6:cloud",),
        },
        env=(
            EnvRequirement(
                "OLLAMA_BASE_URL",
                "Ollama base URL",
                required=False,
                secret=False,
                default="http://localhost:11434/v1",
            ),
            EnvRequirement(
                "OLLAMA_API_KEY",
                "Ollama API key",
                required=False,
                default="",
            ),
        ),
    ),
    "openrouter": ProviderDefinition(
        provider_id="openrouter",
        label="OpenRouter",
        tier_defaults={
            "scout": ("openrouter:google/gemini-3-flash-preview",),
            "worker": ("openrouter:anthropic/claude-sonnet-4-6",),
            "architect": ("openrouter:anthropic/claude-opus-4-6",),
        },
        env=(EnvRequirement("OPENROUTER_API_KEY", "OpenRouter API key"),),
        custom_models=True,
    ),
    "litellm": ProviderDefinition(
        provider_id="litellm",
        label="LiteLLM",
        tier_defaults={
            "scout": ("litellm:openai/gpt-5.4-nano",),
            "worker": ("litellm:openai/gpt-5.4-mini",),
            "architect": ("litellm:openai/gpt-5.4",),
        },
        env=(
            EnvRequirement("LITELLM_API_BASE", "LiteLLM API base URL", secret=False),
            EnvRequirement("LITELLM_API_KEY", "LiteLLM API key"),
        ),
        custom_models=True,
    ),
}


def default_model_tiers(provider: str = DEFAULT_PROVIDER) -> dict[str, list[str]]:
    """Return editable default tier buckets for a provider."""
    definition = provider_definition(provider)
    return {
        tier: list(definition.tier_defaults[tier])
        for tier in TIER_NAMES
        if tier in definition.tier_defaults
    }


def provider_definition(provider: str) -> ProviderDefinition:
    """Return provider metadata or raise a CLI-friendly configuration error."""
    try:
        return PROVIDER_DEFINITIONS[provider]
    except KeyError as exc:
        supported = ", ".join(PROVIDER_DEFINITIONS)
        raise ConfigurationError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        ) from exc


def infer_provider(model: str, default_provider: str | None = None) -> str:
    """Infer the provider family for a model reference."""
    normalized = model.lower()
    if normalized.startswith("gateway/"):
        return "gateway"
    if normalized.startswith("anthropic:"):
        return "anthropic"
    if normalized.startswith(("openai:", "gpt-")):
        return "openai"
    if normalized.startswith(("google-gla:", "google-vertex:")) or (
        "gemini" in normalized and normalized.startswith("google")
    ):
        return "google-gla"
    if normalized.startswith("ollama:"):
        return "ollama"
    if normalized.startswith("openrouter:"):
        return "openrouter"
    if normalized.startswith("litellm:"):
        return "litellm"
    if default_provider in PROVIDER_DEFINITIONS:
        return default_provider
    return DEFAULT_PROVIDER


def provider_model_name(model: str, provider: str) -> str:
    """Strip JAC provider prefixes before provider-specific constructors run."""
    normalized = model.lower()
    if provider == "gateway":
        return model
    prefixes = {
        "anthropic": "anthropic:",
        "openai": "openai:",
        "google-gla": "google-gla:",
        "ollama": "ollama:",
        "openrouter": "openrouter:",
        "litellm": "litellm:",
    }
    prefix = prefixes.get(provider)
    if prefix and normalized.startswith(prefix):
        return model.split(":", 1)[1]
    if provider == "google-gla" and normalized.startswith("google-vertex:"):
        return model.split(":", 1)[1]
    return model


def profile_env_name(profile: str, name: str) -> str:
    """Return the profile-scoped env var name for a provider setting."""
    slug = re.sub(r"[^A-Z0-9]+", "_", profile.upper()).strip("_")
    return f"JAC_PROFILE_{slug}_{name}"


def credential_requirements_for_model(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return credential groups required by a model.

    Each inner tuple is an OR group. Every group must be satisfied.
    """
    selected_provider = provider or infer_provider(model)
    if selected_provider == "ollama":
        return ()
    if selected_provider in PROVIDER_DEFINITIONS:
        required = tuple(
            (item.name,)
            for item in PROVIDER_DEFINITIONS[selected_provider].env
            if item.required
        )
        if required:
            return required

    normalized = model.lower()
    if normalized.startswith("ollama:"):
        return ()
    if normalized.startswith("gateway/"):
        return (("PYDANTIC_AI_GATEWAY_API_KEY",),)
    if normalized.startswith(("openai:", "gpt-")):
        return (("OPENAI_API_KEY",),)
    if normalized.startswith("anthropic:"):
        return (("ANTHROPIC_API_KEY",),)
    if normalized.startswith("openrouter:"):
        return (("OPENROUTER_API_KEY",),)
    if (
        normalized.startswith(("google-gla:", "google-vertex:"))
        or "gemini" in normalized
    ):
        return (("GEMINI_API_KEY",),)
    if normalized.startswith("litellm:"):
        return (("LITELLM_API_KEY",), ("LITELLM_API_BASE",))
    return (("PYDANTIC_AI_GATEWAY_API_KEY", "GEMINI_API_KEY"),)


def credential_candidates_for_model(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[str, ...]:
    """Return credential env vars associated with a model id."""
    names: list[str] = []
    for group in credential_requirements_for_model(model, provider=provider):
        for name in group:
            if name not in names:
                names.append(name)
    return tuple(names)


class Settings(BaseSettings):
    """Settings for JAC."""

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    def __init__(self, **values: Any) -> None:
        """Load workspace dotenv files before BaseSettings reads env vars."""
        load_workspace_env()
        settings_values = _load_workspace_settings()
        settings_values.update(values)
        values = settings_values
        super().__init__(**values)

    model: str = Field(
        description="The model to use for JAC.",
        default=DEFAULT_MODEL,
        validation_alias="JAC_MODEL",
    )

    default_provider: str = Field(
        description="Provider family used when tier model ids are provider-local.",
        default=DEFAULT_PROVIDER,
    )

    default_tier: str = Field(
        description="Default tier used when no session tier or model override is set.",
        default=DEFAULT_TIER,
    )

    model_tiers: dict[str, list[str]] = Field(
        description="Tiered model buckets in scout/worker/architect order.",
        default_factory=dict,
    )

    active_profile: str | None = Field(
        description="Named provider/model profile currently selected.",
        default=None,
        validation_alias=AliasChoices("JAC_ACTIVE_PROFILE", "active_profile"),
    )

    profiles: dict[str, dict[str, Any]] = Field(
        description="Named provider/model configurations.",
        default_factory=dict,
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
            raise ConfigurationError(
                f"{name} is required to call the configured model."
            )
        return self.api_key.get_secret_value()

    def require_env_var(self, name: str) -> str:
        """Return a provider-specific env var or raise a CLI-friendly error."""
        if value := self._env_value(name):
            return value
        raise ConfigurationError(f"{name} is required to call the configured model.")

    def optional_env_var(self, name: str) -> str | None:
        """Return a provider-specific env var when present."""
        return self._env_value(name)

    @property
    def ollama_base_url(self) -> str:
        """Return the configured Ollama OpenAI-compatible endpoint."""
        return self._env_value("OLLAMA_BASE_URL") or "http://localhost:11434/v1"

    def candidate_env_names(self, name: str) -> tuple[str, ...]:
        """Return profile-scoped then global env names for a provider variable."""
        if self.active_profile:
            return (profile_env_name(self.active_profile, name), name)
        return (name,)

    def require_gemini_api_key(self) -> str:
        """Backward-compatible helper for the original W0 Gemini tests."""
        if key := os.getenv("GEMINI_API_KEY"):
            return key
        return self.require_api_key("GEMINI_API_KEY")

    def require_model_credentials(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        """Raise a CLI-friendly error if the selected model needs missing credentials."""
        selected_model = model or self.model
        requirements = credential_requirements_for_model(
            selected_model,
            provider=provider,
        )
        missing_groups = [
            group
            for group in requirements
            if not any(self._env_value(name) for name in group)
        ]
        if not missing_groups:
            return

        names = " and ".join(" or ".join(group) for group in missing_groups)
        raise ConfigurationError(
            f"Model '{selected_model}' requires {names}. "
            "Run `jac init --global` or set the env var before making an LLM call."
        )

    def resolve_model_selection(
        self,
        *,
        model_override: str | None = None,
        tier: str | None = None,
    ) -> ModelSelection:
        """Resolve the model to use for a session turn."""
        if model_override:
            provider = infer_provider(model_override, self.default_provider)
            return ModelSelection(
                model_ref=model_override,
                provider=provider,
                source="model_override",
            )

        selected_tier = tier or self.default_tier
        if self.model_tiers:
            models = self.model_tiers.get(selected_tier)
            if models:
                model_ref = models[0]
                provider = infer_provider(model_ref, self.default_provider)
                return ModelSelection(
                    model_ref=model_ref,
                    provider=provider,
                    source=f"tier:{selected_tier}",
                )

        provider = infer_provider(self.model, self.default_provider)
        return ModelSelection(
            model_ref=self.model, provider=provider, source="JAC_MODEL"
        )

    def _env_value(self, name: str) -> str | None:
        for candidate in self.candidate_env_names(name):
            if value := os.getenv(candidate):
                return value
        return None


def _load_workspace_settings() -> dict[str, Any]:
    """Load non-secret JSON settings from workspace scopes."""
    workspace = discover_workspace()
    paths = [workspace.user_dir / "settings.json"]
    if workspace.project_dir is not None:
        paths.append(workspace.project_dir / "settings.json")
        paths.append(workspace.project_dir / "settings.local.json")

    merged: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON settings file: {path}") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError(
                f"Settings file must contain a JSON object: {path}"
            )
        merged.update(_normalize_settings_payload(payload))
    return _apply_active_profile(merged)


def _normalize_settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if isinstance(normalized.get("profiles"), dict):
        normalized["profiles"] = {
            name: _normalize_profile_payload(profile)
            for name, profile in normalized["profiles"].items()
            if isinstance(name, str) and isinstance(profile, dict)
        }
    if "model_tiers" not in normalized and isinstance(
        normalized.get("model_overrides"),
        dict,
    ):
        normalized["model_tiers"] = {
            tier: [model]
            for tier, model in normalized["model_overrides"].items()
            if tier in TIER_NAMES and isinstance(model, str) and model
        }
    if isinstance(normalized.get("model_tiers"), dict):
        normalized["model_tiers"] = _normalize_model_tiers(normalized["model_tiers"])
    return normalized


def _normalize_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value for key, value in payload.items() if key in PROFILE_CONFIG_KEYS
    }
    if "model_tiers" not in normalized and isinstance(
        normalized.get("model_overrides"),
        dict,
    ):
        normalized["model_tiers"] = {
            tier: [model]
            for tier, model in normalized["model_overrides"].items()
            if tier in TIER_NAMES and isinstance(model, str) and model
        }
    if isinstance(normalized.get("model_tiers"), dict):
        normalized["model_tiers"] = _normalize_model_tiers(normalized["model_tiers"])
    return normalized


def _apply_active_profile(settings: dict[str, Any]) -> dict[str, Any]:
    active_profile = os.getenv("JAC_ACTIVE_PROFILE") or settings.get("active_profile")
    profiles = settings.get("profiles")
    if not isinstance(active_profile, str) or not isinstance(profiles, dict):
        return settings
    profile = profiles.get(active_profile)
    if not isinstance(profile, dict):
        return settings
    resolved = dict(settings)
    resolved.update(
        {key: value for key, value in profile.items() if key in PROFILE_CONFIG_KEYS}
    )
    resolved["active_profile"] = active_profile
    return resolved


def _normalize_model_tiers(value: dict[str, Any]) -> dict[str, list[str]]:
    tiers: dict[str, list[str]] = {}
    for tier in TIER_NAMES:
        models = value.get(tier)
        if isinstance(models, str):
            models = [models]
        if not isinstance(models, list):
            continue
        cleaned = [
            item.strip() for item in models if isinstance(item, str) and item.strip()
        ]
        if cleaned:
            tiers[tier] = cleaned
    return tiers
