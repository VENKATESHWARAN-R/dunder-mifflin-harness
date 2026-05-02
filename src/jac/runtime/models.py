"""Provider-aware model construction for Pydantic AI agents."""

from __future__ import annotations

from typing import Any

from jac.config import (
    ConfigurationError,
    ModelSelection,
    Settings,
    provider_model_name,
)


def build_pydantic_model(selection: ModelSelection, settings: Settings) -> Any:
    """Build the Pydantic AI model value for a resolved provider/model pair."""
    settings.require_model_credentials(
        selection.model_ref,
        provider=selection.provider,
    )
    model_name = provider_model_name(selection.model_ref, selection.provider)

    if selection.provider == "gateway":
        return selection.model_ref

    if selection.provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=settings.require_env_var("OPENAI_API_KEY")),
        )

    if selection.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(
                api_key=settings.require_env_var("ANTHROPIC_API_KEY")
            ),
        )

    if selection.provider == "google-gla":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=settings.require_env_var("GEMINI_API_KEY")),
        )

    if selection.provider == "ollama":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        return OpenAIChatModel(
            model_name,
            provider=OllamaProvider(
                base_url=settings.ollama_base_url,
                api_key=settings.optional_env_var("OLLAMA_API_KEY"),
            ),
        )

    if selection.provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(
                api_key=settings.require_env_var("OPENROUTER_API_KEY"),
            ),
        )

    if selection.provider == "litellm":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.litellm import LiteLLMProvider

        return OpenAIChatModel(
            model_name,
            provider=LiteLLMProvider(
                api_base=settings.require_env_var("LITELLM_API_BASE"),
                api_key=settings.require_env_var("LITELLM_API_KEY"),
            ),
        )

    raise ConfigurationError(f"Unsupported provider '{selection.provider}'.")
