"""CLI for the dunder-mifflin-harness."""

from pydantic_ai import Agent
import logfire

from dunder_mifflin_harness.config import Settings

logfire.configure()
logfire.instrument_pydantic_ai()

settings = Settings()

agent = Agent(
    settings.model,
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
    output_type=str,
    model_settings={"temperature": 0},
)


def main() -> None:
    """Main function for the dunder-mifflin-harness."""
    print("Starting the dunder-mifflin-harness...")
    agent.to_cli_sync()
    print("Dunder-mifflin-harness started successfully.")


if __name__ == "__main__":
    main()
