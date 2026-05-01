"""Compatibility fallback for direct execution of the old W0 CLI module."""

from dunder_mifflin_harness.cli.main import main, run_prompt

__all__ = ["main", "run_prompt"]


if __name__ == "__main__":
    raise SystemExit(main())
