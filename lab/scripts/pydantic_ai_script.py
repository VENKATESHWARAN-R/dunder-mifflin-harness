import asyncio
from pydantic_ai import Agent, RunContext
from langfuse import get_client

langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

Agent.instrument_all()

# Define the agent
roulette_agent = Agent(
    "gateway/openai:gpt-5.4-nano",
    deps_type=int,
    output_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to see if the "
        "customer has won based on the number they provide."
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """Check if the square is the winner"""
    return "Winner" if square == ctx.deps else "loser"


async def main():
    # Run the agent
    success_number = 19
    print(f"Running agent with winning number: {success_number}")

    # Using the standard asyncio.run to execute the async agent.run method
    result = await roulette_agent.run(
        "Put my money on square eighteen", deps=success_number
    )

    print(f"Agent Response: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
