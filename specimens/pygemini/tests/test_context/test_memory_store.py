"""Tests for pygemini.context.memory_store (MemoryStore)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pygemini.context.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """MemoryStore backed by a temp file that starts empty."""
    return MemoryStore(storage_path=tmp_path / "memory.json")


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------


class TestSave:
    """save() should append entries and persist them to the JSON file."""

    def test_creates_file_on_first_save(self, store: MemoryStore, tmp_path: Path) -> None:
        storage_file = tmp_path / "memory.json"
        assert not storage_file.exists()
        store.save("first entry")
        assert storage_file.exists()

    def test_saved_file_is_valid_json(self, store: MemoryStore, tmp_path: Path) -> None:
        store.save("a fact")
        data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_file_contains_content(self, store: MemoryStore, tmp_path: Path) -> None:
        store.save("User likes dark mode.")
        data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
        assert any(entry.get("content") == "User likes dark mode." for entry in data)

    def test_multiple_saves_append(self, store: MemoryStore, tmp_path: Path) -> None:
        store.save("fact one")
        store.save("fact two")
        data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_order_preserved(self, store: MemoryStore, tmp_path: Path) -> None:
        store.save("first")
        store.save("second")
        store.save("third")
        data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
        contents = [entry["content"] for entry in data]
        assert contents == ["first", "second", "third"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "memory.json"
        s = MemoryStore(storage_path=nested)
        s.save("nested fact")
        assert nested.exists()


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------


class TestLoad:
    """load() should return all stored entries."""

    def test_save_and_load_round_trip(self, store: MemoryStore) -> None:
        store.save("Remember this.")
        entries = store.load()
        assert len(entries) == 1
        assert entries[0]["content"] == "Remember this."

    def test_multiple_entries_round_trip(self, store: MemoryStore) -> None:
        store.save("alpha")
        store.save("beta")
        store.save("gamma")
        entries = store.load()
        assert len(entries) == 3
        contents = [e["content"] for e in entries]
        assert "alpha" in contents
        assert "gamma" in contents

    def test_returns_list(self, store: MemoryStore) -> None:
        store.save("x")
        assert isinstance(store.load(), list)

    def test_each_entry_has_content_key(self, store: MemoryStore) -> None:
        store.save("some info")
        entries = store.load()
        assert all("content" in e for e in entries)


# ---------------------------------------------------------------------------
# TestLoadMissing
# ---------------------------------------------------------------------------


class TestLoadMissing:
    """load() from a non-existent file should return an empty list."""

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        store = MemoryStore(storage_path=tmp_path / "nonexistent.json")
        assert store.load() == []

    def test_corrupted_file_returns_empty_list(self, tmp_path: Path) -> None:
        corrupted = tmp_path / "bad.json"
        corrupted.write_text("not valid json {{{", encoding="utf-8")
        store = MemoryStore(storage_path=corrupted)
        assert store.load() == []

    def test_non_list_json_returns_empty_list(self, tmp_path: Path) -> None:
        wrong_type = tmp_path / "wrong.json"
        wrong_type.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        store = MemoryStore(storage_path=wrong_type)
        assert store.load() == []


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------


class TestClear:
    """clear() should remove all stored entries."""

    def test_clear_empties_store(self, store: MemoryStore) -> None:
        store.save("to be cleared")
        store.clear()
        assert store.load() == []

    def test_clear_creates_empty_file(self, store: MemoryStore, tmp_path: Path) -> None:
        store.save("entry")
        store.clear()
        data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
        assert data == []

    def test_save_after_clear_works(self, store: MemoryStore) -> None:
        store.save("before clear")
        store.clear()
        store.save("after clear")
        entries = store.load()
        assert len(entries) == 1
        assert entries[0]["content"] == "after clear"

    def test_clear_on_empty_store_is_safe(self, store: MemoryStore) -> None:
        store.clear()  # Should not raise
        assert store.load() == []


# ---------------------------------------------------------------------------
# TestGetFormatted
# ---------------------------------------------------------------------------


class TestGetFormatted:
    """get_formatted() should return a human-readable string."""

    def test_empty_store_returns_empty_string(self, store: MemoryStore) -> None:
        assert store.get_formatted() == ""

    def test_formatted_contains_content(self, store: MemoryStore) -> None:
        store.save("User prefers Python 3.12.")
        formatted = store.get_formatted()
        assert "User prefers Python 3.12." in formatted

    def test_formatted_contains_header(self, store: MemoryStore) -> None:
        store.save("a fact")
        formatted = store.get_formatted()
        assert "Remembered" in formatted or "facts" in formatted.lower()

    def test_formatted_contains_all_entries(self, store: MemoryStore) -> None:
        store.save("fact one")
        store.save("fact two")
        formatted = store.get_formatted()
        assert "fact one" in formatted
        assert "fact two" in formatted

    def test_formatted_is_multiline_for_multiple_entries(self, store: MemoryStore) -> None:
        store.save("entry A")
        store.save("entry B")
        formatted = store.get_formatted()
        assert "\n" in formatted


# ---------------------------------------------------------------------------
# TestTimestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    """Saved entries should include a timestamp field."""

    def test_entry_has_timestamp_key(self, store: MemoryStore) -> None:
        store.save("timestamped fact")
        entries = store.load()
        assert "timestamp" in entries[0]

    def test_timestamp_is_non_empty_string(self, store: MemoryStore) -> None:
        store.save("another fact")
        entries = store.load()
        ts = entries[0]["timestamp"]
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_timestamp_is_iso_format(self, store: MemoryStore) -> None:
        """Timestamps should be ISO 8601 (contain 'T' and a date)."""
        store.save("iso timestamp check")
        entries = store.load()
        ts = entries[0]["timestamp"]
        assert "T" in ts  # ISO 8601: 2024-01-01T12:00:00+00:00

    def test_multiple_entries_have_timestamps(self, store: MemoryStore) -> None:
        store.save("first")
        store.save("second")
        entries = store.load()
        assert all("timestamp" in e for e in entries)

    def test_formatted_output_includes_timestamps(self, store: MemoryStore) -> None:
        store.save("fact with timestamp")
        formatted = store.get_formatted()
        # Timestamps appear in brackets in the formatted output
        assert "[" in formatted and "]" in formatted
