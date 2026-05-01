"""Tests for pygemini.session.checkpoint (CheckpointManager)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pygemini.core.config import Config
from pygemini.session.checkpoint import CheckpointInfo, CheckpointManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A minimal project directory with a tracked Python file."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')", encoding="utf-8")
    return project


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Config with checkpointing enabled; sessions stored in tmp_path."""
    import os
    monkeypatch_env = {"PYGEMINI_HOME": str(tmp_path / "config")}
    original = {k: os.environ.get(k) for k in monkeypatch_env}
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    yield Config(checkpointing_enabled=True)
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def disabled_config(tmp_path: Path) -> Config:
    import os
    monkeypatch_env = {"PYGEMINI_HOME": str(tmp_path / "config")}
    original = {k: os.environ.get(k) for k in monkeypatch_env}
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    yield Config(checkpointing_enabled=False)
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def manager(config: Config, project_root: Path) -> CheckpointManager:
    return CheckpointManager(config, project_root)


@pytest.fixture
def disabled_manager(disabled_config: Config, project_root: Path) -> CheckpointManager:
    return CheckpointManager(disabled_config, project_root)


def _fake_git_run(cmd: list[str], *, cwd: Path, check: bool = True, capture: bool = False):
    """Fake _git_run that simulates successful git operations."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = 0
    if "rev-parse" in cmd:
        result.stdout = "abc1234567890abcdef"
    elif "diff" in cmd and "--cached" in cmd and "--quiet" in cmd:
        # Simulate "there are staged changes" by returning non-zero
        result.returncode = 1
    else:
        result.stdout = ""
    return result


# ---------------------------------------------------------------------------
# TestIsEnabled
# ---------------------------------------------------------------------------


class TestIsEnabled:
    """is_enabled() should reflect the config setting."""

    def test_enabled_when_config_true(self, manager: CheckpointManager) -> None:
        assert manager.is_enabled() is True

    def test_disabled_when_config_false(self, disabled_manager: CheckpointManager) -> None:
        assert disabled_manager.is_enabled() is False


# ---------------------------------------------------------------------------
# TestCreate
# ---------------------------------------------------------------------------


class TestCreate:
    """create() should save a checkpoint with mocked git operations."""

    def test_disabled_returns_none(self, disabled_manager: CheckpointManager) -> None:
        result = disabled_manager.create("write_file", {"path": "/foo.py"}, [])
        assert result is None

    def test_create_returns_checkpoint_info(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {"path": "/foo.py"}, [])

        assert isinstance(info, CheckpointInfo)

    def test_create_sets_tool_name(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {"path": "/foo.py"}, [])

        assert info is not None
        assert info.tool_name == "write_file"

    def test_create_sets_tool_args(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("edit_file", {"path": "/bar.py"}, [])

        assert info is not None
        assert info.tool_args == {"path": "/bar.py"}

    def test_create_sets_checkpoint_id(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {}, [])

        assert info is not None
        assert len(info.checkpoint_id) > 0

    def test_create_sets_created_at(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {}, [])

        assert info is not None
        assert "T" in info.created_at  # ISO 8601

    def test_create_saves_metadata_to_disk(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {}, [{"role": "user", "parts": []}])

        assert info is not None
        # The checkpoints.json file should exist in the shadow dir
        assert manager._checkpoints_file.exists()

    def test_create_saves_history_snapshot(self, manager: CheckpointManager) -> None:
        history_snapshot = [{"role": "user", "parts": [{"text": "hello"}]}]
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {}, history_snapshot)

        assert info is not None
        history_file = manager._shadow_dir / f"{info.checkpoint_id}.history.json"
        assert history_file.exists()
        stored = json.loads(history_file.read_text(encoding="utf-8"))
        assert stored == history_snapshot

    def test_create_returns_none_on_exception(self, manager: CheckpointManager) -> None:
        """If _init_shadow_repo raises, create() should return None (not crash)."""
        with patch.object(
            manager, "_init_shadow_repo", side_effect=RuntimeError("git unavailable")
        ):
            info = manager.create("write_file", {}, [])

        assert info is None


# ---------------------------------------------------------------------------
# TestListCheckpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    """list_checkpoints() should enumerate saved checkpoints newest-first."""

    def test_empty_returns_empty_list(self, manager: CheckpointManager) -> None:
        assert manager.list_checkpoints() == []

    def test_lists_created_checkpoints(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            manager.create("write_file", {"path": "/a.py"}, [])
            manager.create("edit_file", {"path": "/b.py"}, [])

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2

    def test_checkpoints_sorted_newest_first(self, manager: CheckpointManager) -> None:
        import time

        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            manager.create("write_file", {"path": "/first.py"}, [])
            time.sleep(0.01)
            manager.create("write_file", {"path": "/second.py"}, [])

        checkpoints = manager.list_checkpoints()
        # Newest first — second checkpoint's created_at > first
        assert checkpoints[0].tool_args == {"path": "/second.py"}

    def test_returns_checkpoint_info_objects(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            manager.create("write_file", {}, [])

        checkpoints = manager.list_checkpoints()
        assert all(isinstance(c, CheckpointInfo) for c in checkpoints)


# ---------------------------------------------------------------------------
# TestRestore
# ---------------------------------------------------------------------------


class TestRestore:
    """restore() should load history and file state from a checkpoint."""

    def test_restore_missing_id_returns_none(self, manager: CheckpointManager) -> None:
        result = manager.restore("nonexistent-id")
        assert result is None

    def test_restore_returns_tuple(self, manager: CheckpointManager) -> None:
        history_snapshot = [{"role": "user", "parts": [{"text": "hello"}]}]
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {"path": "/test.py"}, history_snapshot)

        assert info is not None

        # Now restore — patch git checkout and sync
        def fake_restore_git(cmd, *, cwd, check=True, capture=False):
            r = MagicMock(spec=subprocess.CompletedProcess)
            r.returncode = 0
            r.stdout = ""
            return r

        with patch.object(CheckpointManager, "_git_run", side_effect=fake_restore_git):
            with patch.object(manager, "_sync_shadow_to_project"):
                result = manager.restore(info.checkpoint_id)

        assert result is not None
        history, tool_call_info = result
        assert isinstance(history, list)
        assert isinstance(tool_call_info, dict)

    def test_restore_returns_correct_history(self, manager: CheckpointManager) -> None:
        history_snapshot = [{"role": "user", "parts": [{"text": "remember me"}]}]
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {}, history_snapshot)

        assert info is not None

        def fake_restore_git(cmd, *, cwd, check=True, capture=False):
            r = MagicMock(spec=subprocess.CompletedProcess)
            r.returncode = 0
            r.stdout = ""
            return r

        with patch.object(CheckpointManager, "_git_run", side_effect=fake_restore_git):
            with patch.object(manager, "_sync_shadow_to_project"):
                result = manager.restore(info.checkpoint_id)

        assert result is not None
        history, _ = result
        assert history == history_snapshot

    def test_restore_returns_tool_call_info(self, manager: CheckpointManager) -> None:
        with patch.object(
            CheckpointManager, "_git_run", side_effect=_fake_git_run
        ):
            info = manager.create("write_file", {"path": "/x.py"}, [])

        assert info is not None

        def fake_restore_git(cmd, *, cwd, check=True, capture=False):
            r = MagicMock(spec=subprocess.CompletedProcess)
            r.returncode = 0
            r.stdout = ""
            return r

        with patch.object(CheckpointManager, "_git_run", side_effect=fake_restore_git):
            with patch.object(manager, "_sync_shadow_to_project"):
                result = manager.restore(info.checkpoint_id)

        assert result is not None
        _, tool_call_info = result
        assert tool_call_info["tool_name"] == "write_file"
        assert tool_call_info["tool_args"] == {"path": "/x.py"}
