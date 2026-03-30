"""Session manager — save and load named conversation sessions.

Sessions are stored as JSON files under ``~/.pygemini/sessions/<name>.json``.
Each file contains metadata (name, timestamps, message count) alongside the
full serialised conversation history.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import types

from pygemini.core.config import Config
from pygemini.core.history import ConversationHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Metadata about a saved session."""

    name: str
    created_at: str       # ISO 8601
    updated_at: str       # ISO 8601
    message_count: int
    file_path: Path


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _part_to_dict(part: types.Part) -> dict[str, Any]:
    """Convert a single ``types.Part`` to a plain dict.

    Handles the three part variants that appear in a Gemini conversation:
    text, function_call, and function_response.  Unknown / empty parts are
    represented as ``{"text": ""}``.
    """
    if part.text is not None:
        return {"text": part.text}

    if part.function_call is not None:
        fc = part.function_call
        return {
            "function_call": {
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            }
        }

    if part.function_response is not None:
        fr = part.function_response
        return {
            "function_response": {
                "name": fr.name,
                "response": dict(fr.response) if fr.response else {},
            }
        }

    # Fallback — represent as empty text so the round-trip stays lossless
    # for the common case and never silently drops a part.
    return {"text": ""}


def _content_to_dict(content: types.Content) -> dict[str, Any]:
    """Serialise a ``types.Content`` object to a plain dict."""
    return {
        "role": content.role,
        "parts": [_part_to_dict(p) for p in (content.parts or [])],
    }


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Save, load, and list conversation sessions.

    Sessions are stored as JSON files in ``~/.pygemini/sessions/<name>.json``.
    The file layout is::

        {
            "name": "my-session",
            "created_at": "2026-03-28T12:00:00+00:00",
            "updated_at": "2026-03-28T12:05:00+00:00",
            "messages": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                ...
            ]
        }

    On load the raw message dicts are returned so that the caller (typically
    ``AgentLoop`` or ``ContentGenerator``) can reconstruct ``types.Content``
    objects as needed — this avoids a hard dependency on the SDK's internal
    deserialisation API inside this module.
    """

    def __init__(self, config: Config) -> None:
        self._sessions_dir: Path = config.config_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, name: str, history: ConversationHistory) -> SessionInfo:
        """Save the current conversation to a named session file.

        If a session with *name* already exists it is overwritten.  The
        ``created_at`` timestamp is preserved across overwrites.

        Args:
            name: Human-readable session name.  Non-alphanumeric characters
                (except hyphens) are replaced with underscores in the
                filename but the original *name* is stored in the JSON.
            history: The conversation history to persist.

        Returns:
            A :class:`SessionInfo` describing the saved session.
        """
        path = self._session_path(name)
        now = datetime.datetime.now(datetime.UTC).isoformat()

        # Preserve original created_at when overwriting an existing session.
        created_at = now
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                pass  # Treat a corrupt file as a new session.

        messages = [_content_to_dict(c) for c in history.get_messages()]

        payload: dict[str, Any] = {
            "name": name,
            "created_at": created_at,
            "updated_at": now,
            "messages": messages,
        }

        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to write session file: %s", path)
            raise

        logger.debug("Saved session %r to %s (%d messages)", name, path, len(messages))

        return SessionInfo(
            name=name,
            created_at=created_at,
            updated_at=now,
            message_count=len(messages),
            file_path=path,
        )

    def load(self, name: str) -> list[dict[str, Any]]:
        """Load a saved session and return the raw message dicts.

        The dicts have the shape ``{"role": str, "parts": list[dict]}`` and
        can be fed back to the Gemini API or used to reconstruct
        ``types.Content`` objects.

        Args:
            name: The session name to load.

        Returns:
            List of message dicts.

        Raises:
            FileNotFoundError: If no session with *name* exists.
        """
        path = self._session_path(name)
        if not path.exists():
            raise FileNotFoundError(f"No session named {name!r} found at {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise OSError(f"Failed to read session file {path}: {exc}") from exc

        messages: list[dict[str, Any]] = data.get("messages", [])
        logger.debug("Loaded session %r from %s (%d messages)", name, path, len(messages))
        return messages

    def list_sessions(self) -> list[SessionInfo]:
        """List all saved sessions, sorted by ``updated_at`` (newest first).

        Returns:
            Sorted list of :class:`SessionInfo` objects.  Returns an empty
            list if no sessions have been saved yet.
        """
        infos: list[SessionInfo] = []

        for json_file in self._sessions_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable session file: %s", json_file)
                continue

            infos.append(
                SessionInfo(
                    name=data.get("name", json_file.stem),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    message_count=len(data.get("messages", [])),
                    file_path=json_file,
                )
            )

        infos.sort(key=lambda s: s.updated_at, reverse=True)
        return infos

    def delete(self, name: str) -> bool:
        """Delete a saved session.

        Args:
            name: The session name to delete.

        Returns:
            ``True`` if the session was deleted, ``False`` if it was not found.
        """
        path = self._session_path(name)
        if not path.exists():
            return False

        try:
            path.unlink()
        except OSError:
            logger.exception("Failed to delete session file: %s", path)
            raise

        logger.debug("Deleted session %r (%s)", name, path)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_path(self, name: str) -> Path:
        """Return the filesystem path for a named session.

        Non-alphanumeric characters (except hyphens and underscores) in
        *name* are replaced with underscores so that the filename is safe on
        all platforms.

        Args:
            name: Session name.

        Returns:
            Absolute path to the ``.json`` session file.
        """
        safe_name = re.sub(r"[^\w\-]", "_", name)
        return self._sessions_dir / f"{safe_name}.json"
