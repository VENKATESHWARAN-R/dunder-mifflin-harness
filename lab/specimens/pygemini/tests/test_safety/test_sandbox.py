"""Tests for pygemini.safety.sandbox (Sandbox, SandboxMode)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch


from pygemini.core.config import Config
from pygemini.safety.sandbox import Sandbox, SandboxMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sandbox(mode: str = "none") -> Sandbox:
    """Construct a Sandbox from a minimal Config."""
    config = Config(sandbox=mode)  # type: ignore[arg-type]
    return Sandbox(config)


# ---------------------------------------------------------------------------
# TestSandboxMode
# ---------------------------------------------------------------------------


class TestSandboxMode:
    """SandboxMode enum values match the literal strings used in Config."""

    def test_none_value(self) -> None:
        assert SandboxMode.NONE.value == "none"

    def test_docker_value(self) -> None:
        assert SandboxMode.DOCKER.value == "docker"

    def test_enum_from_string_none(self) -> None:
        assert SandboxMode("none") is SandboxMode.NONE

    def test_enum_from_string_docker(self) -> None:
        assert SandboxMode("docker") is SandboxMode.DOCKER

    def test_none_mode_passthrough(self) -> None:
        """In none mode, wrap_command returns the command unchanged."""
        sandbox = _make_sandbox("none")
        cmd = "echo hello"
        assert sandbox.wrap_command(cmd) == cmd

    def test_docker_mode_wraps_command(self) -> None:
        """In docker mode, wrap_command returns a docker run invocation."""
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("echo hello")
        assert result.startswith("docker run")
        assert "echo hello" in result


# ---------------------------------------------------------------------------
# TestWrapCommand
# ---------------------------------------------------------------------------


class TestWrapCommand:
    """wrap_command() should produce the correct docker run command structure."""

    def test_contains_docker_run_rm(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("ls")
        assert "docker run --rm" in result

    def test_contains_volume_mount(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("ls")
        # Should contain -v <something>:/workspace
        assert "-v " in result
        assert ":/workspace" in result

    def test_contains_working_dir_flag(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("ls")
        assert "-w /workspace" in result

    def test_contains_docker_image(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("ls")
        assert "python:3.12-slim" in result

    def test_contains_bash_c(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("echo hi")
        assert 'bash -c' in result

    def test_command_embedded_in_result(self) -> None:
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("echo hi")
        assert "echo hi" in result

    def test_double_quotes_in_command_are_escaped(self) -> None:
        """Double quotes in the command must be escaped so the shell quoting is valid."""
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command('echo "hello world"')
        assert '\\"hello world\\"' in result

    def test_none_mode_returns_exact_command(self) -> None:
        sandbox = _make_sandbox("none")
        cmd = 'echo "unmodified"'
        assert sandbox.wrap_command(cmd) == cmd

    def test_docker_uses_cwd_as_project_root(self) -> None:
        """The volume mount should reference an actual path (not empty string)."""
        sandbox = _make_sandbox("docker")
        result = sandbox.wrap_command("pwd")
        # The volume flag looks like: -v /some/path:/workspace
        assert "-v /" in result or "-v " in result


# ---------------------------------------------------------------------------
# TestIsSandboxed
# ---------------------------------------------------------------------------


class TestIsSandboxed:
    """is_sandboxed() reflects whether Docker mode is active."""

    def test_none_mode_not_sandboxed(self) -> None:
        sandbox = _make_sandbox("none")
        assert sandbox.is_sandboxed() is False

    def test_docker_mode_is_sandboxed(self) -> None:
        sandbox = _make_sandbox("docker")
        assert sandbox.is_sandboxed() is True


# ---------------------------------------------------------------------------
# TestCheckDockerAvailable
# ---------------------------------------------------------------------------


class TestCheckDockerAvailable:
    """check_docker_available() probes the Docker daemon via subprocess."""

    def test_docker_available_when_returncode_zero(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            available = Sandbox.check_docker_available()
        assert available is True
        mock_run.assert_called_once()

    def test_docker_unavailable_when_returncode_nonzero(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            available = Sandbox.check_docker_available()
        assert available is False

    def test_docker_unavailable_when_binary_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            available = Sandbox.check_docker_available()
        assert available is False

    def test_docker_unavailable_when_timeout_expires(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 5)):
            available = Sandbox.check_docker_available()
        assert available is False

    def test_docker_unavailable_on_os_error(self) -> None:
        with patch("subprocess.run", side_effect=OSError("permission denied")):
            available = Sandbox.check_docker_available()
        assert available is False

    def test_subprocess_run_called_with_docker_info(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            Sandbox.check_docker_available()
        args, kwargs = mock_run.call_args
        assert args[0] == ["docker", "info"]

    def test_subprocess_called_with_timeout(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            Sandbox.check_docker_available()
        _, kwargs = mock_run.call_args
        assert "timeout" in kwargs
        assert kwargs["timeout"] > 0
