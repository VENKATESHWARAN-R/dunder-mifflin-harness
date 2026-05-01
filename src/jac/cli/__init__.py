"""Public CLI API.

This package intentionally exposes the same `main` and `run_prompt` names that
the original single-file W0 CLI exposed to tests and script entrypoints.
"""

from dunder_mifflin_harness.cli.main import main, run_prompt

__all__ = ["main", "run_prompt"]
