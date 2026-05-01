"""run_shell_command tool — executes shell commands via subprocess with confirmation."""

from __future__ import annotations

import asyncio

from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult

_MAX_OUTPUT_LEN = 100_000
_KEEP_HEAD = 50_000
_KEEP_TAIL = 50_000
_TRUNCATION_MARKER = "\n\n... [truncated — output too long] ...\n\n"


def _truncate(text: str) -> str:
    """Truncate extremely long output, keeping head and tail."""
    if len(text) <= _MAX_OUTPUT_LEN:
        return text
    return text[:_KEEP_HEAD] + _TRUNCATION_MARKER + text[-_KEEP_TAIL:]


class ShellTool(BaseTool):
    """Run a shell command and return stdout/stderr."""

    @property
    def name(self) -> str:
        return "run_shell_command"

    @property
    def description(self) -> str:
        return (
            "Run a shell command and return its stdout and stderr. "
            "Use this for running tests, installing packages, git operations, "
            "build commands, and other shell tasks."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default: 30.",
                },
            },
            "required": ["command"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "command" not in params:
            return "Missing required parameter: command"
        if not params["command"].strip():
            return "Command must not be empty"
        timeout = params.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            return "Timeout must be a positive integer"
        return None

    def get_description(self, params: dict) -> str:
        command = params.get("command", "?")
        if len(command) > 80:
            command = command[:77] + "..."
        return f"Run: {command}"

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        command = params.get("command", "?")
        timeout = params.get("timeout", 30)
        return ToolConfirmation(
            description=f"Run shell command: {command}",
            details={
                "command": command,
                "timeout": timeout,
            },
        )

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        command = params["command"]
        timeout = params.get("timeout", 30)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    llm_content=f"Error: Command timed out after {timeout} seconds: {command}",
                    display_content=(
                        f"[red]Timeout:[/red] Command exceeded {timeout}s limit"
                    ),
                    is_error=True,
                )

            # Check abort signal after completion
            if abort_signal is not None and abort_signal.is_set():
                process.kill()
                return ToolResult(
                    llm_content="Command aborted by user.",
                    display_content="[yellow]Aborted[/yellow]",
                    is_error=True,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode

            # Truncate extremely long output
            stdout = _truncate(stdout)
            stderr = _truncate(stderr)

            # Build LLM content
            llm_parts = [f"Exit code: {exit_code}"]
            if stdout:
                llm_parts.append(f"stdout:\n{stdout}")
            else:
                llm_parts.append("stdout: (empty)")
            if stderr:
                llm_parts.append(f"stderr:\n{stderr}")
            llm_content = "\n".join(llm_parts)

            # Build display content with Rich markup
            if exit_code == 0:
                status = f"[green]Exit code: {exit_code}[/green]"
            else:
                status = f"[red]Exit code: {exit_code}[/red]"

            display_parts = [status]
            if stdout:
                display_parts.append(stdout.rstrip())
            if stderr:
                display_parts.append(f"[dim]stderr:[/dim]\n{stderr.rstrip()}")
            display_content = "\n".join(display_parts)

            return ToolResult(
                llm_content=llm_content,
                display_content=display_content,
                is_error=exit_code != 0,
            )

        except OSError as e:
            return ToolResult(
                llm_content=f"Error executing command: {e}",
                display_content=f"[red]Execution error:[/red] {e}",
                is_error=True,
            )
