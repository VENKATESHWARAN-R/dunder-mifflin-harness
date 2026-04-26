from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from dunder_mifflin_harness.config import Settings


INSTRUCTIONS = (
    "You are the W0 dunder-mifflin-harness agent. "
    "Answer directly and keep responses concise."
)
GOOGLE_GLA_PREFIX = "google-gla:"


def _google_model_name(model: str) -> str:
    if model.startswith(GOOGLE_GLA_PREFIX):
        return model.removeprefix(GOOGLE_GLA_PREFIX)
    return model


def build_agent(settings: Settings) -> Agent[None, str]:
    provider = GoogleProvider(api_key=settings.require_gemini_api_key())
    model = GoogleModel(_google_model_name(settings.model), provider=provider)
    return Agent(model, instructions=INSTRUCTIONS, output_type=str)


async def run_prompt(prompt: str, settings: Settings | None = None) -> str:
    runtime_settings = settings or Settings()
    result = await build_agent(runtime_settings).run(prompt)
    return result.output
