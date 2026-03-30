"""Tests for pygemini.context.file_discovery (FileDiscovery)."""

from __future__ import annotations

from pathlib import Path


from pygemini.context.file_discovery import FileDiscovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(root: Path, files: list[str]) -> None:
    """Create a set of files (relative paths) under root."""
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("content", encoding="utf-8")


# ---------------------------------------------------------------------------
# TestDiscover
# ---------------------------------------------------------------------------


class TestDiscover:
    """discover() should walk the tree and return non-ignored relative paths."""

    def test_finds_files_in_root(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["main.py", "README.md"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        names = {p.name for p in found}
        assert "main.py" in names
        assert "README.md" in names

    def test_finds_files_in_subdirectories(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["src/app.py", "src/utils.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        names = {p.name for p in found}
        assert "app.py" in names
        assert "utils.py" in names

    def test_returns_relative_paths(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["src/app.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        # All paths should be relative (not absolute)
        assert all(not p.is_absolute() for p in found)

    def test_ignores_dot_git(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, [".git/config", "main.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert not any(".git" in str(p) for p in found)
        assert any(p.name == "main.py" for p in found)

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["__pycache__/module.pyc", "src/app.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert not any("__pycache__" in str(p) for p in found)

    def test_ignores_custom_patterns(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["secrets.txt", "config.py"])
        fd = FileDiscovery(tmp_path, ignore_patterns=["secrets.txt"])
        found = fd.discover()
        names = {p.name for p in found}
        assert "secrets.txt" not in names
        assert "config.py" in names

    def test_ignores_binary_extensions(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["image.png", "archive.zip", "script.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        names = {p.name for p in found}
        assert "image.png" not in names
        assert "archive.zip" not in names
        assert "script.py" in names

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert found == []

    def test_node_modules_ignored(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["node_modules/package/index.js", "app.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert not any("node_modules" in str(p) for p in found)

    def test_venv_ignored(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, [".venv/lib/python.py", "main.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert not any(".venv" in str(p) for p in found)
        assert any(p.name == "main.py" for p in found)

    def test_pyc_files_ignored(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, ["app.pyc", "app.py"])
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        names = {p.name for p in found}
        assert "app.pyc" not in names
        assert "app.py" in names


# ---------------------------------------------------------------------------
# TestResolveReference
# ---------------------------------------------------------------------------


class TestResolveReference:
    """resolve_reference() should convert @file references to absolute paths."""

    def test_absolute_path_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "absolute.py"
        f.write_text("x = 1", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference(str(f))
        assert result == f

    def test_absolute_path_missing_returns_none(self, tmp_path: Path) -> None:
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference(str(tmp_path / "nonexistent.py"))
        assert result is None

    def test_relative_path_resolved_against_root(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "app.py"
        f.parent.mkdir()
        f.write_text("x = 1", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference("src/app.py")
        assert result is not None
        assert result.name == "app.py"

    def test_relative_path_missing_returns_none(self, tmp_path: Path) -> None:
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference("does/not/exist.py")
        assert result is None

    def test_glob_pattern_matches_first(self, tmp_path: Path) -> None:
        (tmp_path / "module_a.py").write_text("a", encoding="utf-8")
        (tmp_path / "module_b.py").write_text("b", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference("module_*.py")
        assert result is not None
        assert result.suffix == ".py"

    def test_glob_pattern_no_match_returns_none(self, tmp_path: Path) -> None:
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference("*.xyz")
        assert result is None

    def test_plain_filename_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "config.toml"
        f.write_text("[tool]", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        result = fd.resolve_reference("config.toml")
        assert result is not None
        assert result.name == "config.toml"


# ---------------------------------------------------------------------------
# TestLoadIgnoreFile
# ---------------------------------------------------------------------------


class TestLoadIgnoreFile:
    """load_ignore_file() should parse patterns from a .pygeminiignore-style file."""

    def test_parses_patterns(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".pygeminiignore"
        ignore_file.write_text("*.log\nbuild/\n", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(ignore_file)
        assert "*.log" in patterns
        assert "build/" in patterns

    def test_ignores_comment_lines(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".pygeminiignore"
        ignore_file.write_text("# this is a comment\n*.log\n", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(ignore_file)
        assert "# this is a comment" not in patterns
        assert "*.log" in patterns

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".pygeminiignore"
        ignore_file.write_text("\n\n*.log\n\n", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(ignore_file)
        assert "" not in patterns
        assert "*.log" in patterns

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(tmp_path / "nonexistent")
        assert patterns == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".pygeminiignore"
        ignore_file.write_text("", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(ignore_file)
        assert patterns == []

    def test_strips_whitespace_from_patterns(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".pygeminiignore"
        ignore_file.write_text("  *.log  \n", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        patterns = fd.load_ignore_file(ignore_file)
        assert "*.log" in patterns


# ---------------------------------------------------------------------------
# TestShouldIgnore
# ---------------------------------------------------------------------------


class TestShouldIgnore:
    """_should_ignore() should detect binary extensions and ignored paths."""

    def test_binary_png_not_in_discover_results(self, tmp_path: Path) -> None:
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert all(p.suffix != ".png" for p in found)

    def test_binary_pdf_not_in_discover_results(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert all(p.suffix != ".pdf" for p in found)

    def test_dotfile_in_discovery(self, tmp_path: Path) -> None:
        """Dotfiles that aren't in the default ignore list should be discovered."""
        (tmp_path / ".env-example").write_text("KEY=value", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        # .env-example should appear (it's not in the default ignore list)
        # .env itself is ignored by default
        names = {p.name for p in found}
        assert ".env-example" in names

    def test_dot_env_ignored_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=yes", encoding="utf-8")
        fd = FileDiscovery(tmp_path)
        found = fd.discover()
        assert all(p.name != ".env" for p in found)
