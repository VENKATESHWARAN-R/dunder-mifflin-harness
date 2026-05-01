"""Tests for pygemini.context.gemini_md (GeminiMDDiscovery)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygemini.context.gemini_md import ContextFile, GeminiMDDiscovery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def discovery() -> GeminiMDDiscovery:
    return GeminiMDDiscovery()


# ---------------------------------------------------------------------------
# TestContextFile
# ---------------------------------------------------------------------------


class TestContextFile:
    """ContextFile dataclass should be constructable and frozen."""

    def test_construction(self, tmp_path: Path) -> None:
        p = tmp_path / "GEMINI.md"
        ctx = ContextFile(path=p, content="hello", origin="test")
        assert ctx.path == p
        assert ctx.content == "hello"
        assert ctx.origin == "test"

    def test_frozen(self, tmp_path: Path) -> None:
        p = tmp_path / "GEMINI.md"
        ctx = ContextFile(path=p, content="hello", origin="test")
        with pytest.raises((AttributeError, TypeError)):
            ctx.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDiscover
# ---------------------------------------------------------------------------


class TestDiscover:
    """discover() should find GEMINI.md files at the right hierarchy levels."""

    def test_global_file_found(self, discovery: GeminiMDDiscovery, tmp_path: Path) -> None:
        """Config-dir GEMINI.md should appear as the first result."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "GEMINI.md").write_text("global context", encoding="utf-8")

        # cwd without any project root markers
        cwd = tmp_path / "project"
        cwd.mkdir()

        files = discovery.discover(cwd, config_dir=config_dir)

        origins = [f.origin for f in files]
        assert any("global" in o for o in origins)
        assert files[0].content == "global context"

    def test_project_root_file_found(self, discovery: GeminiMDDiscovery, tmp_path: Path) -> None:
        """GEMINI.md at the project root (marked by pyproject.toml) should be discovered."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (project / "GEMINI.md").write_text("project context", encoding="utf-8")

        config_dir = tmp_path / "config"
        config_dir.mkdir()  # Empty — no global GEMINI.md

        files = discovery.discover(project, config_dir=config_dir)

        assert any("project context" in f.content for f in files)

    def test_cwd_file_found(self, discovery: GeminiMDDiscovery, tmp_path: Path) -> None:
        """A GEMINI.md in the current working directory should be discovered."""
        project = tmp_path / "project"
        subdir = project / "sub"
        subdir.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (subdir / "GEMINI.md").write_text("subdir context", encoding="utf-8")

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        files = discovery.discover(subdir, config_dir=config_dir)

        assert any("subdir context" in f.content for f in files)

    def test_hierarchical_order(self, discovery: GeminiMDDiscovery, tmp_path: Path) -> None:
        """Global file appears first; project-root file before cwd file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "GEMINI.md").write_text("global", encoding="utf-8")

        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (project / "GEMINI.md").write_text("root level", encoding="utf-8")

        subdir = project / "sub"
        subdir.mkdir()
        (subdir / "GEMINI.md").write_text("sub level", encoding="utf-8")

        files = discovery.discover(subdir, config_dir=config_dir)

        contents = [f.content for f in files]
        assert contents.index("global") < contents.index("root level")
        assert contents.index("root level") < contents.index("sub level")

    def test_no_files_returns_empty_list(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """When no GEMINI.md files exist, discover() returns an empty list."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()

        files = discovery.discover(cwd, config_dir=config_dir)
        assert files == []

    def test_child_directories_scanned(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """GEMINI.md inside immediate subdirectories of cwd should be found."""
        project = tmp_path / "project"
        child = project / "child"
        child.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (child / "GEMINI.md").write_text("child context", encoding="utf-8")

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        files = discovery.discover(project, config_dir=config_dir)
        assert any("child context" in f.content for f in files)


# ---------------------------------------------------------------------------
# TestProcessImports
# ---------------------------------------------------------------------------


class TestProcessImports:
    """process_imports() should resolve @import directives."""

    def test_import_resolved(self, discovery: GeminiMDDiscovery, tmp_path: Path) -> None:
        """An @import directive should be replaced by the file's contents."""
        imported = tmp_path / "extra.md"
        imported.write_text("imported content", encoding="utf-8")

        content = "@extra.md\n"
        result = discovery.process_imports(content, base_path=tmp_path)

        assert "imported content" in result

    def test_circular_import_detected(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """Circular imports should produce a comment and not loop forever."""
        # File A imports itself
        self_ref = tmp_path / "self.md"
        self_ref.write_text("@self.md\n", encoding="utf-8")

        content = "@self.md\n"
        result = discovery.process_imports(content, base_path=tmp_path)

        # The recursive call should detect the cycle
        assert "circular" in result.lower() or "self.md" in result

    def test_missing_import_handled(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """A missing import file should produce an error comment, not an exception."""
        content = "@does_not_exist.md\n"
        result = discovery.process_imports(content, base_path=tmp_path)

        # Should contain an HTML-style comment about the missing file
        assert "<!--" in result
        assert "does_not_exist.md" in result

    def test_non_import_line_unchanged(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """Regular Markdown text should not be modified."""
        content = "# Heading\n\nSome text.\n"
        result = discovery.process_imports(content, base_path=tmp_path)
        assert result == content

    def test_nested_imports_resolved(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        """Nested @import directives should be resolved recursively."""
        deep = tmp_path / "deep.md"
        deep.write_text("deep content", encoding="utf-8")

        middle = tmp_path / "middle.md"
        middle.write_text("@deep.md\n", encoding="utf-8")

        content = "@middle.md\n"
        result = discovery.process_imports(content, base_path=tmp_path)

        assert "deep content" in result


# ---------------------------------------------------------------------------
# TestConcatenate
# ---------------------------------------------------------------------------


class TestConcatenate:
    """concatenate() should join files with section headers."""

    def test_empty_list_returns_empty_string(self, discovery: GeminiMDDiscovery) -> None:
        assert discovery.concatenate([]) == ""

    def test_single_file_contains_content(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        ctx = ContextFile(path=tmp_path / "GEMINI.md", content="hello world", origin="test")
        result = discovery.concatenate([ctx])
        assert "hello world" in result

    def test_single_file_has_header_and_footer(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        ctx = ContextFile(path=tmp_path / "GEMINI.md", content="content", origin="my-origin")
        result = discovery.concatenate([ctx])
        assert "my-origin" in result
        assert "---" in result

    def test_multiple_files_all_present(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        files = [
            ContextFile(path=tmp_path / "a.md", content="alpha", origin="origin-a"),
            ContextFile(path=tmp_path / "b.md", content="beta", origin="origin-b"),
        ]
        result = discovery.concatenate(files)
        assert "alpha" in result
        assert "beta" in result

    def test_multiple_files_separator_present(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        files = [
            ContextFile(path=tmp_path / "a.md", content="alpha", origin="origin-a"),
            ContextFile(path=tmp_path / "b.md", content="beta", origin="origin-b"),
        ]
        result = discovery.concatenate(files)
        # Files should be separated by a blank line
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# TestLoadContext
# ---------------------------------------------------------------------------


class TestLoadContext:
    """load_context() is the end-to-end convenience method."""

    def test_no_files_returns_empty_string(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()

        result = discovery.load_context(cwd, config_dir=config_dir)
        assert result == ""

    def test_single_file_content_present(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "GEMINI.md").write_text("system instructions", encoding="utf-8")

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        result = discovery.load_context(cwd, config_dir=config_dir)
        assert "system instructions" in result

    def test_imports_resolved_in_result(
        self, discovery: GeminiMDDiscovery, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        extra = config_dir / "extra.md"
        extra.write_text("extra content", encoding="utf-8")
        (config_dir / "GEMINI.md").write_text("@extra.md\n", encoding="utf-8")

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        result = discovery.load_context(cwd, config_dir=config_dir)
        assert "extra content" in result
