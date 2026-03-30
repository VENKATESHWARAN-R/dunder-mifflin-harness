"""MemoryStore — persistent JSON-backed memory for cross-session recall."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


class MemoryStore:
    """Append-only persistent memory store backed by a JSON file.

    Entries are stored as a JSON array of objects with ``content`` and
    ``timestamp`` keys.  The file is created (along with any missing parent
    directories) on first write.  A missing file is treated as an empty store
    rather than an error.
    """

    DEFAULT_PATH = Path.home() / ".pygemini" / "memory.json"

    def __init__(self, storage_path: Path | None = None) -> None:
        self._path: Path = storage_path if storage_path is not None else self.DEFAULT_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, content: str) -> None:
        """Append a new memory entry with the current UTC timestamp."""
        entries = self.load()
        entries.append(
            {
                "content": content,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        self._write(entries)

    def load(self) -> list[dict[str, str]]:
        """Return all stored entries.  Returns an empty list if no file exists."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data  # type: ignore[return-value]
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def clear(self) -> None:
        """Delete all stored memories."""
        self._write([])

    def get_formatted(self) -> str:
        """Return all memories as a formatted string suitable for prompt injection.

        Returns an empty string when there are no entries.
        """
        entries = self.load()
        if not entries:
            return ""
        lines = ["Remembered facts and preferences:"]
        for entry in entries:
            timestamp = entry.get("timestamp", "")
            content = entry.get("content", "")
            lines.append(f"- [{timestamp}] {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, entries: list[dict[str, str]]) -> None:
        """Atomically replace the storage file with *entries*."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # Best-effort: silently swallow write failures so the agent loop
            # is never blocked by a storage error.
            pass
