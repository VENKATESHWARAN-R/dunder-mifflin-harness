"""Tests for pygemini.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygemini.core.config import (
    Config,
    HookConfig,
    MCPServerConfig,
    ensure_config_dir,
    load_config,
)


# ---------------------------------------------------------------------------
# Config model defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Config should have sane defaults out of the box."""

    def test_default_model(self) -> None:
        cfg = Config()
        assert cfg.model == "gemini-2.5-flash"

    def test_default_fallback_model(self) -> None:
        cfg = Config()
        assert cfg.fallback_model == "gemini-2.5-flash"

    def test_api_key_none_by_default(self) -> None:
        cfg = Config()
        assert cfg.api_key is None

    def test_default_approval_mode(self) -> None:
        cfg = Config()
        assert cfg.approval_mode == "interactive"

    def test_default_sandbox(self) -> None:
        cfg = Config()
        assert cfg.sandbox == "none"

    def test_core_tools_populated(self) -> None:
        cfg = Config()
        assert len(cfg.core_tools) > 0
        assert "read_file" in cfg.core_tools
        assert "write_file" in cfg.core_tools

    def test_empty_collections_by_default(self) -> None:
        cfg = Config()
        assert cfg.allowed_tools == []
        assert cfg.excluded_tools == []
        assert cfg.gemini_md_paths == []
        assert cfg.include_directories == []
        assert cfg.mcp_servers == {}
        assert cfg.hooks == {}

    def test_hooks_enabled_by_default(self) -> None:
        cfg = Config()
        assert cfg.hooks_enabled is True

    def test_checkpointing_disabled_by_default(self) -> None:
        cfg = Config()
        assert cfg.checkpointing_enabled is False

    def test_default_theme(self) -> None:
        cfg = Config()
        assert cfg.theme == "default"


# ---------------------------------------------------------------------------
# Config construction with explicit values
# ---------------------------------------------------------------------------


class TestConfigExplicit:
    """Config should accept explicit field values."""

    def test_set_api_key(self) -> None:
        cfg = Config(api_key="test-key-123")
        assert cfg.api_key == "test-key-123"

    def test_set_model(self) -> None:
        cfg = Config(model="gemini-2.0-pro")
        assert cfg.model == "gemini-2.0-pro"

    def test_set_approval_mode_yolo(self) -> None:
        cfg = Config(approval_mode="yolo")
        assert cfg.approval_mode == "yolo"

    def test_set_sandbox_docker(self) -> None:
        cfg = Config(sandbox="docker")
        assert cfg.sandbox == "docker"

    def test_invalid_approval_mode_rejected(self) -> None:
        with pytest.raises(Exception):
            Config(approval_mode="invalid")  # type: ignore[arg-type]

    def test_invalid_sandbox_rejected(self) -> None:
        with pytest.raises(Exception):
            Config(sandbox="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class TestNestedModels:
    """MCPServerConfig and HookConfig should work correctly."""

    def test_mcp_server_config_defaults(self) -> None:
        mcp = MCPServerConfig(command="node")
        assert mcp.command == "node"
        assert mcp.args == []
        assert mcp.env == {}

    def test_mcp_server_config_full(self) -> None:
        mcp = MCPServerConfig(
            command="npx",
            args=["-y", "@mcp/server"],
            env={"TOKEN": "abc"},
        )
        assert mcp.args == ["-y", "@mcp/server"]
        assert mcp.env["TOKEN"] == "abc"

    def test_hook_config_defaults(self) -> None:
        hook = HookConfig(command="echo hello")
        assert hook.command == "echo hello"
        assert hook.timeout == 10

    def test_hook_config_custom_timeout(self) -> None:
        hook = HookConfig(command="run.sh", timeout=30)
        assert hook.timeout == 30

    def test_config_with_mcp_servers(self) -> None:
        cfg = Config(
            mcp_servers={
                "my_server": MCPServerConfig(command="node", args=["server.js"])
            }
        )
        assert "my_server" in cfg.mcp_servers
        assert cfg.mcp_servers["my_server"].command == "node"

    def test_config_with_hooks(self) -> None:
        cfg = Config(
            hooks={
                "pre_tool_call": [HookConfig(command="validate.sh")]
            }
        )
        assert len(cfg.hooks["pre_tool_call"]) == 1


# ---------------------------------------------------------------------------
# config_dir property
# ---------------------------------------------------------------------------


class TestConfigDir:
    """config_dir should respect PYGEMINI_HOME."""

    def test_default_config_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PYGEMINI_HOME", raising=False)
        cfg = Config()
        assert cfg.config_dir == Path("~/.pygemini").expanduser()

    def test_custom_config_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path / "custom"))
        cfg = Config()
        assert cfg.config_dir == tmp_path / "custom"


# ---------------------------------------------------------------------------
# ensure_config_dir
# ---------------------------------------------------------------------------


class TestEnsureConfigDir:
    """ensure_config_dir should create the directory."""

    def test_creates_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "new_config"
        monkeypatch.setenv("PYGEMINI_HOME", str(target))
        result = ensure_config_dir()
        assert result == target
        assert target.is_dir()

    def test_with_config_object(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "from_config"
        monkeypatch.setenv("PYGEMINI_HOME", str(target))
        cfg = Config()
        result = ensure_config_dir(cfg)
        assert result == target
        assert target.is_dir()


# ---------------------------------------------------------------------------
# load_config — layered loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """load_config should merge layers correctly."""

    def test_defaults_only(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("PYGEMINI_MODEL", raising=False)
        monkeypatch.delenv("PYGEMINI_SANDBOX", raising=False)
        monkeypatch.delenv("PYGEMINI_APPROVAL_MODE", raising=False)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.api_key is None

    def test_env_api_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.api_key == "env-key"

    def test_env_model(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("PYGEMINI_MODEL", "gemini-2.0-pro")
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "gemini-2.0-pro"

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config(model="cli-model")
        assert cfg.model == "cli-model"

    def test_cli_none_does_not_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config(model=None)
        assert cfg.model == "env-model"

    def test_user_toml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("PYGEMINI_MODEL", raising=False)
        monkeypatch.delenv("PYGEMINI_SANDBOX", raising=False)
        monkeypatch.delenv("PYGEMINI_APPROVAL_MODE", raising=False)
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_text('model = "toml-model"\ntheme = "dark"\n')
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "toml-model"
        assert cfg.theme == "dark"

    def test_project_toml_overrides_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("PYGEMINI_MODEL", raising=False)
        monkeypatch.delenv("PYGEMINI_SANDBOX", raising=False)
        monkeypatch.delenv("PYGEMINI_APPROVAL_MODE", raising=False)
        # User-level
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_text('model = "user-model"\ntheme = "user-theme"\n')
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        # Project-level
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".pygemini").mkdir()
        (project_dir / ".pygemini" / "settings.toml").write_text('model = "project-model"\n')
        monkeypatch.chdir(project_dir)
        cfg = load_config()
        assert cfg.model == "project-model"
        # User-level theme should still be there
        assert cfg.theme == "user-theme"

    def test_env_overrides_toml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_text('model = "toml-model"\n')
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("PYGEMINI_SANDBOX", raising=False)
        monkeypatch.delenv("PYGEMINI_APPROVAL_MODE", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "env-model"

    def test_full_layer_precedence(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """CLI > env > project TOML > user TOML > defaults."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("PYGEMINI_SANDBOX", raising=False)
        monkeypatch.delenv("PYGEMINI_APPROVAL_MODE", raising=False)
        # User TOML
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_text('model = "user"\ntheme = "user-theme"\n')
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        # Project TOML
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".pygemini").mkdir()
        (project_dir / ".pygemini" / "settings.toml").write_text('model = "project"\n')
        monkeypatch.chdir(project_dir)
        # Env
        monkeypatch.setenv("PYGEMINI_MODEL", "env")
        # CLI wins
        cfg = load_config(model="cli")
        assert cfg.model == "cli"
        assert cfg.theme == "user-theme"  # survived all layers
