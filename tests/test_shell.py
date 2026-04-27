import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.shell import run_shell_command, truncate_output


def test_run_shell_command_captures_output(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell_command(
            command="printf hello",
            cwd=tmp_path,
            timeout_seconds=2,
            max_output_chars=1000,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert result.stderr == ""


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
