"""Docker sandbox module — wraps shell commands for sandboxed execution."""

from __future__ import annotations

import logging
import subprocess
from enum import Enum
from pathlib import Path

from pygemini.core.config import Config

logger = logging.getLogger(__name__)

_DOCKER_IMAGE = "python:3.12-slim"
_WORKSPACE = "/workspace"


class SandboxMode(Enum):
    """Supported sandbox execution modes."""

    NONE = "none"
    DOCKER = "docker"


class Sandbox:
    """Wraps shell command execution in a sandbox.

    When sandbox mode is "docker", commands are executed inside a Docker
    container with the project directory mounted as a volume.
    When mode is "none", commands pass through directly.
    """

    def __init__(self, config: Config) -> None:
        self._mode = SandboxMode(config.sandbox)
        self._project_root = Path.cwd()

    def wrap_command(self, command: str) -> str:
        """Wrap a command for sandboxed execution.

        For Docker mode, produces:
            docker run --rm -v <project_root>:/workspace -w /workspace
                python:3.12-slim bash -c "<command>"

        The project directory is mounted read-write at /workspace and set as
        the working directory inside the container. --rm ensures the container
        is removed automatically after the command exits.

        For none mode, the command is returned unchanged.
        """
        if self._mode is SandboxMode.NONE:
            return command

        # Escape any double-quotes in the command so it nests safely inside
        # the outer bash -c "..." quoting.
        escaped = command.replace('"', '\\"')

        return (
            f'docker run --rm'
            f' -v {self._project_root}:{_WORKSPACE}'
            f' -w {_WORKSPACE}'
            f' {_DOCKER_IMAGE}'
            f' bash -c "{escaped}"'
        )

    def is_sandboxed(self) -> bool:
        """Return True if Docker sandboxing is active."""
        return self._mode is SandboxMode.DOCKER

    @staticmethod
    def check_docker_available() -> bool:
        """Return True if Docker is installed and the daemon is reachable.

        Runs ``docker info`` with a short timeout.  Any failure — missing
        binary, daemon not running, permission error — returns False and logs
        a warning rather than raising.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            available = result.returncode == 0
            if not available:
                logger.warning(
                    "docker info returned non-zero exit code %d; "
                    "Docker may not be running.",
                    result.returncode,
                )
            return available
        except FileNotFoundError:
            logger.warning("Docker binary not found; sandbox mode unavailable.")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("docker info timed out; Docker daemon may be unresponsive.")
            return False
        except OSError as exc:
            logger.warning("Error checking Docker availability: %s", exc)
            return False
