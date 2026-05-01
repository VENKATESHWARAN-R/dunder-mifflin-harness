"""Git-based checkpoint manager for file-modifying operations.

On every file-modifying tool execution the manager:
1. Copies tracked project files into a *shadow* git repository
   (``~/.pygemini/history/<project_hash>/``).
2. Commits the snapshot so it can be restored later.
3. Persists checkpoint metadata and a conversation-history snapshot.

Restoring a checkpoint reverses all three steps: the shadow commit is
checked out, files are copied back into the project, and the saved
history snapshot is returned so the caller can rewind the conversation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pygemini.core.config import Config

__all__ = [
    "CheckpointInfo",
    "CheckpointManager",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File extensions that are eligible for checkpointing
# ---------------------------------------------------------------------------

_TRACKED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini"}
)

# Directories to skip when copying project files into the shadow repo.
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata about a single checkpoint."""

    checkpoint_id: str  # UUID
    created_at: str  # ISO 8601
    tool_name: str  # Tool that triggered the checkpoint
    tool_args: dict[str, Any]  # Tool arguments
    description: str  # Human-readable description
    commit_hash: str | None  # Git commit hash in shadow repo


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Git-based checkpointing for file-modifying operations.

    On every file-modifying tool execution:
    1. Create a git commit in a shadow repo (``~/.pygemini/history/<project_hash>/``).
    2. Save conversation history snapshot.
    3. Save the tool call that triggered the checkpoint.

    Restore reverses all three.
    """

    def __init__(self, config: Config, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root.resolve()
        # Shadow repo path based on a short hash of the project root.
        project_hash = hashlib.md5(str(self._project_root).encode()).hexdigest()[:12]
        self._shadow_dir = config.config_dir / "history" / project_hash
        self._checkpoints_file = self._shadow_dir / "checkpoints.json"
        self._enabled = config.checkpointing_enabled

    # -- Public API ----------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return whether checkpointing is enabled in the current config."""
        return self._enabled

    def create(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        history_snapshot: list[dict[str, Any]],
    ) -> CheckpointInfo | None:
        """Create a checkpoint before a file-modifying tool executes.

        Steps:
        1. If shadow git repo doesn't exist, initialize it.
        2. Copy tracked files from project to shadow repo.
        3. ``git add -A && git commit`` in shadow repo.
        4. Save checkpoint metadata (id, timestamp, tool info, commit hash).
        5. Save history snapshot alongside.

        Returns:
            :class:`CheckpointInfo` or ``None`` if checkpointing is disabled
            or an error occurred.
        """
        if not self._enabled:
            return None

        try:
            self._init_shadow_repo()
            self._sync_project_to_shadow()

            description = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})"
            commit_hash = self._git_commit(f"checkpoint: {description}")

            info = CheckpointInfo(
                checkpoint_id=uuid.uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),
                tool_name=tool_name,
                tool_args=tool_args,
                description=description,
                commit_hash=commit_hash,
            )

            self._save_checkpoint_metadata(info, history_snapshot)
            logger.info("Checkpoint created: %s (%s)", info.checkpoint_id, description)
            return info
        except Exception:
            logger.exception("Failed to create checkpoint")
            return None

    def restore(self, checkpoint_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        """Restore project files and conversation from a checkpoint.

        Steps:
        1. Find checkpoint by ID.
        2. ``git checkout`` the corresponding commit in the shadow repo.
        3. Copy files back from shadow repo to the project.
        4. Load the saved history snapshot.
        5. Return ``(history_messages, tool_call_info)``.

        Returns:
            A tuple of ``(history_messages, tool_call_info)`` or ``None`` if
            the checkpoint was not found or an error occurred.
        """
        try:
            all_checkpoints = self._load_checkpoint_metadata()
            target: dict[str, Any] | None = None
            for cp in all_checkpoints:
                if cp["checkpoint_id"] == checkpoint_id:
                    target = cp
                    break

            if target is None:
                logger.warning("Checkpoint not found: %s", checkpoint_id)
                return None

            commit_hash = target.get("commit_hash")
            if commit_hash is None:
                logger.warning("Checkpoint %s has no commit hash", checkpoint_id)
                return None

            # Checkout the snapshot in the shadow repo.
            self._git_run(["git", "checkout", commit_hash, "--", "."], cwd=self._shadow_dir)

            # Copy files from shadow back into the project.
            self._sync_shadow_to_project()

            # Load the history snapshot.
            history = self._load_history_snapshot(checkpoint_id)

            tool_call_info: dict[str, Any] = {
                "tool_name": target.get("tool_name", ""),
                "tool_args": target.get("tool_args", {}),
            }

            logger.info("Restored checkpoint %s", checkpoint_id)
            return history, tool_call_info
        except Exception:
            logger.exception("Failed to restore checkpoint %s", checkpoint_id)
            return None

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints, newest first."""
        try:
            raw = self._load_checkpoint_metadata()
            infos = [
                CheckpointInfo(
                    checkpoint_id=cp["checkpoint_id"],
                    created_at=cp["created_at"],
                    tool_name=cp["tool_name"],
                    tool_args=cp["tool_args"],
                    description=cp["description"],
                    commit_hash=cp.get("commit_hash"),
                )
                for cp in raw
            ]
            # Newest first (ISO timestamps sort lexicographically).
            infos.sort(key=lambda c: c.created_at, reverse=True)
            return infos
        except Exception:
            logger.exception("Failed to list checkpoints")
            return []

    # -- Shadow repo helpers -------------------------------------------------

    def _init_shadow_repo(self) -> None:
        """Initialize the shadow git repo if it doesn't exist."""
        if (self._shadow_dir / ".git").is_dir():
            return

        self._shadow_dir.mkdir(parents=True, exist_ok=True)
        self._git_run(["git", "init"], cwd=self._shadow_dir)
        # Create an initial empty commit so that later operations work even
        # when the first real commit has no changes.
        self._git_run(
            ["git", "commit", "--allow-empty", "-m", "init shadow repo"],
            cwd=self._shadow_dir,
        )
        logger.debug("Initialized shadow repo at %s", self._shadow_dir)

    def _git_commit(self, message: str) -> str | None:
        """Stage all changes and commit in the shadow repo.

        Returns the commit hash, or ``None`` if there was nothing to commit.
        """
        self._git_run(["git", "add", "-A"], cwd=self._shadow_dir)

        # Check if there are staged changes.
        result = self._git_run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self._shadow_dir,
            check=False,
        )
        if result.returncode == 0:
            # Nothing to commit — still return the HEAD hash.
            head = self._git_run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._shadow_dir,
                capture=True,
            )
            return head.stdout.strip() if head.stdout else None

        self._git_run(["git", "commit", "-m", message], cwd=self._shadow_dir)

        head = self._git_run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._shadow_dir,
            capture=True,
        )
        return head.stdout.strip() if head.stdout else None

    # -- File synchronization ------------------------------------------------

    def _sync_project_to_shadow(self) -> None:
        """Copy eligible project files into the shadow repo."""
        # Clear existing tracked-file copies (but keep .git and metadata).
        for child in self._shadow_dir.iterdir():
            if child.name in {".git", "checkpoints.json"} or child.name.endswith(".history.json"):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        for src in self._iter_tracked_files():
            rel = src.relative_to(self._project_root)
            dst = self._shadow_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _sync_shadow_to_project(self) -> None:
        """Copy files from the shadow repo back into the project."""
        for src in self._shadow_dir.rglob("*"):
            if not src.is_file():
                continue
            # Skip shadow-repo metadata.
            rel = src.relative_to(self._shadow_dir)
            parts = rel.parts
            if parts[0] == ".git":
                continue
            if rel.name == "checkpoints.json" or rel.name.endswith(".history.json"):
                continue

            dst = self._project_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _iter_tracked_files(self) -> list[Path]:
        """Return a list of eligible files in the project root."""
        files: list[Path] = []
        for path in self._project_root.rglob("*"):
            if not path.is_file():
                continue
            # Skip directories we should never touch.
            if any(part in _SKIP_DIRS for part in path.relative_to(self._project_root).parts):
                continue
            if path.suffix in _TRACKED_EXTENSIONS:
                files.append(path)
        return files

    # -- Metadata persistence ------------------------------------------------

    def _save_checkpoint_metadata(
        self, info: CheckpointInfo, history: list[dict[str, Any]]
    ) -> None:
        """Persist checkpoint info and history to disk."""
        # Append to checkpoints.json
        all_checkpoints = self._load_checkpoint_metadata()
        all_checkpoints.append(asdict(info))
        self._checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoints_file.write_text(
            json.dumps(all_checkpoints, indent=2), encoding="utf-8"
        )

        # Write history snapshot.
        history_path = self._shadow_dir / f"{info.checkpoint_id}.history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def _load_checkpoint_metadata(self) -> list[dict[str, Any]]:
        """Load all checkpoint metadata from disk."""
        if not self._checkpoints_file.is_file():
            return []
        try:
            data = json.loads(self._checkpoints_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt checkpoints.json — starting fresh")
            return []

    def _load_history_snapshot(self, checkpoint_id: str) -> list[dict[str, Any]]:
        """Load a saved history snapshot for a given checkpoint ID."""
        path = self._shadow_dir / f"{checkpoint_id}.history.json"
        if not path.is_file():
            logger.warning("History snapshot not found for %s", checkpoint_id)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt history snapshot for %s", checkpoint_id)
            return []

    # -- Git subprocess wrapper ----------------------------------------------

    @staticmethod
    def _git_run(
        cmd: list[str],
        *,
        cwd: Path,
        check: bool = True,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the given directory.

        Args:
            cmd: Command and arguments (e.g. ``["git", "init"]``).
            cwd: Working directory for the command.
            check: Whether to raise on non-zero exit code.
            capture: Whether to capture stdout/stderr.

        Returns:
            The :class:`subprocess.CompletedProcess` result.
        """
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=capture,
            text=True,
            env={
                # Minimal env so git doesn't pick up user-level config that
                # could interfere (e.g. hooks, signing).
                "HOME": str(Path.home()),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "GIT_AUTHOR_NAME": "pygemini",
                "GIT_AUTHOR_EMAIL": "pygemini@checkpoint",
                "GIT_COMMITTER_NAME": "pygemini",
                "GIT_COMMITTER_EMAIL": "pygemini@checkpoint",
            },
        )
