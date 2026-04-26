import argparse
import asyncio
import sys
from collections.abc import Sequence

from dunder_mifflin_harness.agents.chat import run_prompt
from dunder_mifflin_harness.config import ConfigurationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Run one prompt through the W0 dunder-mifflin harness.",
    )
    parser.add_argument("prompt", nargs="+", help="Prompt to send to the agent.")
    return parser


async def _run(prompt: str) -> str:
    return await run_prompt(prompt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt)

    try:
        response = asyncio.run(_run(prompt))
    except ConfigurationError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("harness: interrupted", file=sys.stderr)
        return 130

    print(response)
    return 0
