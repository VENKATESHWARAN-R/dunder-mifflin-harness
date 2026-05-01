"""Tests for pygemini.session.manager (SessionManager)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.genai import types

from pygemini.core.config import Config
from pygemini.core.history import ConversationHistory
from pygemini.session.manager import SessionInfo, SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Config that stores sessions under tmp_path."""
    import os
    # Override PYGEMINI_HOME so config_dir points to tmp_path
    monkeypatch_env = {"PYGEMINI_HOME": str(tmp_path)}
    original = {k: os.environ.get(k) for k in monkeypatch_env}
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    yield Config()
    # Restore
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def manager(config: Config) -> SessionManager:
    return SessionManager(config)


def _make_history(*messages: str) -> ConversationHistory:
    """Create a history with alternating user/model messages."""
    history = ConversationHistory()
    for i, text in enumerate(messages):
        if i % 2 == 0:
            history.add_user_message(text)
        else:
            history.add_model_response([types.Part.from_text(text=text)])
    return history


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------


class TestSave:
    """save() should write a JSON file with the correct structure."""

    def test_save_creates_file(self, manager: SessionManager) -> None:
        history = _make_history("hello")
        info = manager.save("my-session", history)
        assert info.file_path.exists()

    def test_save_returns_session_info(self, manager: SessionManager) -> None:
        history = _make_history("hi")
        info = manager.save("test-session", history)
        assert isinstance(info, SessionInfo)
        assert info.name == "test-session"

    def test_save_correct_message_count(self, manager: SessionManager) -> None:
        history = _make_history("msg1", "msg2", "msg3")
        info = manager.save("count-test", history)
        assert info.message_count == 3

    def test_save_json_structure(self, manager: SessionManager, tmp_path: Path) -> None:
        history = _make_history("hello", "world")
        info = manager.save("struct-test", history)
        data = json.loads(info.file_path.read_text(encoding="utf-8"))
        assert "name" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_save_name_stored_in_json(self, manager: SessionManager) -> None:
        history = _make_history("hello")
        info = manager.save("my-session", history)
        data = json.loads(info.file_path.read_text(encoding="utf-8"))
        assert data["name"] == "my-session"

    def test_save_messages_stored(self, manager: SessionManager) -> None:
        history = _make_history("test message")
        info = manager.save("msg-test", history)
        data = json.loads(info.file_path.read_text(encoding="utf-8"))
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"

    def test_save_overwrites_existing_preserves_created_at(
        self, manager: SessionManager
    ) -> None:
        history = _make_history("first")
        info1 = manager.save("overwrite-test", history)
        created_at_1 = info1.created_at

        history2 = _make_history("second")
        info2 = manager.save("overwrite-test", history2)

        assert info2.created_at == created_at_1


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------


class TestLoad:
    """load() should return the raw message dicts from a saved session."""

    def test_round_trip_message_count(self, manager: SessionManager) -> None:
        history = _make_history("hello", "world", "foo")
        manager.save("round-trip", history)
        messages = manager.load("round-trip")
        assert len(messages) == 3

    def test_round_trip_roles(self, manager: SessionManager) -> None:
        history = _make_history("user turn", "model turn")
        manager.save("roles-test", history)
        messages = manager.load("roles-test")
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "model"

    def test_round_trip_text_content(self, manager: SessionManager) -> None:
        history = _make_history("remember this")
        manager.save("content-test", history)
        messages = manager.load("content-test")
        parts = messages[0]["parts"]
        assert any(p.get("text") == "remember this" for p in parts)

    def test_returns_list_of_dicts(self, manager: SessionManager) -> None:
        history = _make_history("hi")
        manager.save("type-test", history)
        messages = manager.load("type-test")
        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)

    def test_load_empty_history(self, manager: SessionManager) -> None:
        manager.save("empty-session", ConversationHistory())
        messages = manager.load("empty-session")
        assert messages == []


# ---------------------------------------------------------------------------
# TestLoadMissing
# ---------------------------------------------------------------------------


class TestLoadMissing:
    """load() should raise FileNotFoundError for unknown sessions."""

    def test_raises_file_not_found(self, manager: SessionManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent-session")

    def test_error_message_contains_name(self, manager: SessionManager) -> None:
        with pytest.raises(FileNotFoundError, match="ghost-session"):
            manager.load("ghost-session")


# ---------------------------------------------------------------------------
# TestListSessions
# ---------------------------------------------------------------------------


class TestListSessions:
    """list_sessions() should enumerate saved sessions."""

    def test_empty_returns_empty_list(self, manager: SessionManager) -> None:
        assert manager.list_sessions() == []

    def test_lists_saved_sessions(self, manager: SessionManager) -> None:
        manager.save("session-a", _make_history("a"))
        manager.save("session-b", _make_history("b"))
        sessions = manager.list_sessions()
        names = {s.name for s in sessions}
        assert "session-a" in names
        assert "session-b" in names

    def test_sorted_newest_first(self, manager: SessionManager) -> None:
        import time

        manager.save("older", _make_history("old"))
        time.sleep(0.01)
        manager.save("newer", _make_history("new"))
        sessions = manager.list_sessions()
        assert sessions[0].name == "newer"

    def test_each_item_is_session_info(self, manager: SessionManager) -> None:
        manager.save("info-test", _make_history("msg"))
        sessions = manager.list_sessions()
        assert all(isinstance(s, SessionInfo) for s in sessions)

    def test_message_count_in_list(self, manager: SessionManager) -> None:
        manager.save("count-list", _make_history("a", "b", "c"))
        sessions = manager.list_sessions()
        match = next(s for s in sessions if s.name == "count-list")
        assert match.message_count == 3


# ---------------------------------------------------------------------------
# TestDelete
# ---------------------------------------------------------------------------


class TestDelete:
    """delete() should remove session files."""

    def test_delete_returns_true(self, manager: SessionManager) -> None:
        manager.save("to-delete", _make_history("x"))
        assert manager.delete("to-delete") is True

    def test_deleted_file_gone(self, manager: SessionManager) -> None:
        info = manager.save("gone", _make_history("x"))
        manager.delete("gone")
        assert not info.file_path.exists()

    def test_deleted_session_not_listed(self, manager: SessionManager) -> None:
        manager.save("bye", _make_history("x"))
        manager.delete("bye")
        sessions = manager.list_sessions()
        assert all(s.name != "bye" for s in sessions)

    def test_missing_returns_false(self, manager: SessionManager) -> None:
        assert manager.delete("does-not-exist") is False


# ---------------------------------------------------------------------------
# TestSessionPath
# ---------------------------------------------------------------------------


class TestSessionPath:
    """_session_path() should sanitize session names into safe filenames."""

    def test_plain_name_no_change(self, manager: SessionManager) -> None:
        path = manager._session_path("my-session")
        assert path.stem == "my-session"

    def test_spaces_replaced(self, manager: SessionManager) -> None:
        path = manager._session_path("my session name")
        assert " " not in path.stem

    def test_special_chars_replaced(self, manager: SessionManager) -> None:
        path = manager._session_path("session!@#$%")
        # Should not contain special characters in the stem
        assert "!" not in path.stem
        assert "@" not in path.stem

    def test_json_extension(self, manager: SessionManager) -> None:
        path = manager._session_path("test")
        assert path.suffix == ".json"

    def test_hyphen_and_underscore_preserved(self, manager: SessionManager) -> None:
        path = manager._session_path("my-session_test")
        assert "my-session_test" in path.stem
