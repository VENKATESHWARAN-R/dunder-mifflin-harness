"""Unit tests for pygemini.core.config.

Covers:
- Config field defaults
- load_config with env var overrides (monkeypatch)
- load_config with CLI overrides
- config_dir property
- TOML loading (user-level and project-level)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pygemini.core.config import (
    Config,
    DEFAULT_CORE_TOOLS,
    HookConfig,
    MCPServerConfig,
    ensure_config_dir,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all env vars that influence load_config."""
    for var in ("GEMINI_API_KEY", "PYGEMINI_MODEL", "PYGEMINI_SANDBOX", "PYGEMINI_APPROVAL_MODE"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_model_default(self) -> None:
        assert Config().model == "gemini-2.5-flash"

    def test_fallback_model_default(self) -> None:
        assert Config().fallback_model == "gemini-2.5-flash"

    def test_api_key_none(self) -> None:
        assert Config().api_key is None

    def test_approval_mode_default(self) -> None:
        assert Config().approval_mode == "interactive"

    def test_sandbox_default(self) -> None:
        assert Config().sandbox == "none"

    def test_theme_default(self) -> None:
        assert Config().theme == "default"

    def test_hooks_enabled_default(self) -> None:
        assert Config().hooks_enabled is True

    def test_checkpointing_disabled_default(self) -> None:
        assert Config().checkpointing_enabled is False

    def test_core_tools_contains_read_file(self) -> None:
        assert "read_file" in Config().core_tools

    def test_core_tools_matches_constant(self) -> None:
        assert Config().core_tools == DEFAULT_CORE_TOOLS

    def test_allowed_tools_empty(self) -> None:
        assert Config().allowed_tools == []

    def test_excluded_tools_empty(self) -> None:
        assert Config().excluded_tools == []

    def test_mcp_servers_empty(self) -> None:
        assert Config().mcp_servers == {}

    def test_hooks_empty(self) -> None:
        assert Config().hooks == {}

    def test_gemini_md_paths_empty(self) -> None:
        assert Config().gemini_md_paths == []

    def test_include_directories_empty(self) -> None:
        assert Config().include_directories == []


# ---------------------------------------------------------------------------
# Config explicit construction
# ---------------------------------------------------------------------------


class TestConfigExplicit:
    def test_set_api_key(self) -> None:
        assert Config(api_key="my-key").api_key == "my-key"

    def test_set_model(self) -> None:
        assert Config(model="gemini-2.0-pro").model == "gemini-2.0-pro"

    def test_approval_mode_yolo(self) -> None:
        assert Config(approval_mode="yolo").approval_mode == "yolo"

    def test_approval_mode_auto_edit(self) -> None:
        assert Config(approval_mode="auto_edit").approval_mode == "auto_edit"

    def test_sandbox_docker(self) -> None:
        assert Config(sandbox="docker").sandbox == "docker"

    def test_invalid_approval_mode_raises(self) -> None:
        with pytest.raises(Exception):
            Config(approval_mode="bad")  # type: ignore[arg-type]

    def test_invalid_sandbox_raises(self) -> None:
        with pytest.raises(Exception):
            Config(sandbox="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        mcp = MCPServerConfig(command="node")
        assert mcp.command == "node"
        assert mcp.args == []
        assert mcp.env == {}

    def test_full(self) -> None:
        mcp = MCPServerConfig(command="npx", args=["-y", "@mcp/s"], env={"K": "V"})
        assert mcp.args == ["-y", "@mcp/s"]
        assert mcp.env == {"K": "V"}


class TestHookConfig:
    def test_defaults(self) -> None:
        h = HookConfig(command="run.sh")
        assert h.command == "run.sh"
        assert h.timeout == 10

    def test_custom_timeout(self) -> None:
        assert HookConfig(command="run.sh", timeout=30).timeout == 30


# ---------------------------------------------------------------------------
# config_dir property
# ---------------------------------------------------------------------------


class TestConfigDir:
    def test_default_when_no_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PYGEMINI_HOME", raising=False)
        assert Config().config_dir == Path("~/.pygemini").expanduser()

    def test_respects_pygemini_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        custom = tmp_path / "custom_home"
        monkeypatch.setenv("PYGEMINI_HOME", str(custom))
        assert Config().config_dir == custom

    def test_returns_path_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PYGEMINI_HOME", raising=False)
        assert isinstance(Config().config_dir, Path)


# ---------------------------------------------------------------------------
# ensure_config_dir
# ---------------------------------------------------------------------------


class TestEnsureConfigDir:
    def test_creates_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "new_dir"
        monkeypatch.setenv("PYGEMINI_HOME", str(target))
        result = ensure_config_dir()
        assert result == target
        assert target.is_dir()

    def test_returns_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path / "x"))
        assert isinstance(ensure_config_dir(), Path)

    def test_with_config_object(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "from_config"
        monkeypatch.setenv("PYGEMINI_HOME", str(target))
        cfg = Config()
        result = ensure_config_dir(cfg)
        assert result == target
        assert target.is_dir()

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "idempotent"
        monkeypatch.setenv("PYGEMINI_HOME", str(target))
        ensure_config_dir()
        ensure_config_dir()  # second call must not raise
        assert target.is_dir()


# ---------------------------------------------------------------------------
# load_config — env var overrides
# ---------------------------------------------------------------------------


class TestLoadConfigEnvOverrides:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("GEMINI_API_KEY", "env-api-key")
        cfg = load_config()
        assert cfg.api_key == "env-api-key"

    def test_model_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("PYGEMINI_MODEL", "gemini-env-model")
        cfg = load_config()
        assert cfg.model == "gemini-env-model"

    def test_sandbox_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("PYGEMINI_SANDBOX", "docker")
        cfg = load_config()
        assert cfg.sandbox == "docker"

    def test_approval_mode_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("PYGEMINI_APPROVAL_MODE", "yolo")
        cfg = load_config()
        assert cfg.approval_mode == "yolo"

    def test_no_env_vars_gives_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        cfg = load_config()
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.api_key is None


# ---------------------------------------------------------------------------
# load_config — CLI overrides
# ---------------------------------------------------------------------------


class TestLoadConfigCLIOverrides:
    def test_cli_overrides_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        cfg = load_config(model="cli-model")
        assert cfg.model == "cli-model"

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        cfg = load_config(model="cli-model")
        assert cfg.model == "cli-model"

    def test_cli_none_does_not_clobber_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        cfg = load_config(model=None)
        assert cfg.model == "env-model"

    def test_cli_multiple_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        cfg = load_config(model="my-model", approval_mode="yolo", sandbox="docker")
        assert cfg.model == "my-model"
        assert cfg.approval_mode == "yolo"
        assert cfg.sandbox == "docker"

    def test_cli_api_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        cfg = load_config(api_key="cli-key")
        assert cfg.api_key == "cli-key"


# ---------------------------------------------------------------------------
# load_config — TOML loading
# ---------------------------------------------------------------------------


class TestLoadConfigTOML:
    def test_user_toml_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_bytes(
            b'model = "toml-model"\ntheme = "dark"\n'
        )
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "toml-model"
        assert cfg.theme == "dark"

    def test_project_toml_overrides_user_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clean_env(monkeypatch)
        # user-level
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_bytes(
            b'model = "user-model"\ntheme = "user-theme"\n'
        )
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        # project-level
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".pygemini").mkdir()
        (project_dir / ".pygemini" / "settings.toml").write_bytes(
            b'model = "project-model"\n'
        )
        monkeypatch.chdir(project_dir)
        cfg = load_config()
        assert cfg.model == "project-model"
        assert cfg.theme == "user-theme"  # user setting survives

    def test_env_overrides_toml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch)
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_bytes(b'model = "toml-model"\n')
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        monkeypatch.setenv("PYGEMINI_MODEL", "env-model")
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "env-model"

    def test_missing_toml_is_silently_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clean_env(monkeypatch)
        # Point PYGEMINI_HOME at an empty directory (no settings.toml)
        monkeypatch.setenv("PYGEMINI_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "gemini-2.5-flash"

    def test_full_layer_precedence(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """CLI > env > project TOML > user TOML > defaults."""
        _clean_env(monkeypatch)
        config_home = tmp_path / "home"
        config_home.mkdir()
        (config_home / "settings.toml").write_bytes(
            b'model = "user"\ntheme = "user-theme"\n'
        )
        monkeypatch.setenv("PYGEMINI_HOME", str(config_home))
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".pygemini").mkdir()
        (project_dir / ".pygemini" / "settings.toml").write_bytes(b'model = "project"\n')
        monkeypatch.chdir(project_dir)
        monkeypatch.setenv("PYGEMINI_MODEL", "env")
        cfg = load_config(model="cli")
        assert cfg.model == "cli"
        assert cfg.theme == "user-theme"
