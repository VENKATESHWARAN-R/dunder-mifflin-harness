"""GEMINI.md hierarchical discovery and loading.

Discovers GEMINI.md context files from multiple locations, processes @import
directives, and concatenates them into a single string suitable for inclusion
in the system prompt.

Discovery order:
1. Global: ``~/.pygemini/GEMINI.md``
2. Ancestry: walk from project root down to CWD, collecting GEMINI.md at each level
3. Children: immediate subdirectories of CWD (1 level deep)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Markers used to detect a project root directory.
_PROJECT_ROOT_MARKERS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "Cargo.toml",
)

# Regex matching an @import directive on its own line.
# Captures the file path (group 1).
_IMPORT_RE = re.compile(r"^@(\S+)\s*$", re.MULTILINE)

_CONTEXT_FILENAME = "GEMINI.md"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextFile:
    """A discovered context file with its origin."""

    path: Path
    content: str
    origin: str  # e.g., "global (~/.pygemini/GEMINI.md)"


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------


class GeminiMDDiscovery:
    """Discovers and loads GEMINI.md context files hierarchically."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        cwd: Path,
        config_dir: Path | None = None,
    ) -> list[ContextFile]:
        """Discover GEMINI.md files from global, ancestry, and child locations.

        Search hierarchy:
        1. ``config_dir/GEMINI.md`` (global user-level, defaults to
           ``~/.pygemini/``).
        2. Walk from the detected project root down toward *cwd*, collecting
           any ``GEMINI.md`` files found at each directory level.  The project
           root is the nearest ancestor of *cwd* that contains a project root
           marker (``.git``, ``pyproject.toml``, etc.).
        3. Immediate subdirectories of *cwd* (1 level deep).

        Returns files ordered: global first, then from root toward *cwd*,
        then children.
        """
        cwd = cwd.resolve()
        if config_dir is None:
            config_dir = Path("~/.pygemini").expanduser()
        else:
            config_dir = config_dir.resolve()

        files: list[ContextFile] = []

        # 1. Global -----------------------------------------------------------
        global_path = config_dir / _CONTEXT_FILENAME
        global_file = self._try_read(global_path)
        if global_file is not None:
            origin = f"global ({global_path})"
            logger.debug("Discovered global context: %s", global_path)
            files.append(ContextFile(path=global_path, content=global_file, origin=origin))

        # 2. Ancestry (project root -> cwd) -----------------------------------
        project_root = self._find_project_root(cwd)
        if project_root is not None:
            logger.debug("Detected project root: %s", project_root)
            # Build the list of directories from root down to cwd (inclusive).
            ancestry = self._ancestry_from_root(cwd, project_root)
            for directory in ancestry:
                md_path = directory / _CONTEXT_FILENAME
                content = self._try_read(md_path)
                if content is not None:
                    origin = self._origin_label(md_path, cwd, project_root, config_dir)
                    logger.debug("Discovered ancestry context: %s", md_path)
                    files.append(ContextFile(path=md_path, content=content, origin=origin))
        else:
            # No project root found; just check cwd itself.
            md_path = cwd / _CONTEXT_FILENAME
            content = self._try_read(md_path)
            if content is not None:
                origin = f"directory ({md_path})"
                logger.debug("Discovered cwd context (no project root): %s", md_path)
                files.append(ContextFile(path=md_path, content=content, origin=origin))

        # 3. Child directories (1 level deep) ---------------------------------
        try:
            children = sorted(p for p in cwd.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            children = []

        for child_dir in children:
            md_path = child_dir / _CONTEXT_FILENAME
            content = self._try_read(md_path)
            if content is not None:
                origin = f"directory ({md_path})"
                logger.debug("Discovered child context: %s", md_path)
                files.append(ContextFile(path=md_path, content=content, origin=origin))

        return files

    def process_imports(self, content: str, base_path: Path) -> str:
        """Process ``@import`` directives in GEMINI.md content.

        Syntax: ``@path/to/file.md`` on its own line.  The path is resolved
        relative to *base_path* (the directory containing the GEMINI.md file).

        Circular imports are detected and skipped with a warning comment.
        Missing files are replaced with an error comment.
        """
        seen: set[Path] = set()
        return self._resolve_imports(content, base_path.resolve(), seen)

    def concatenate(self, files: list[ContextFile]) -> str:
        """Combine all discovered context files into a single string.

        Format::

            --- Context from <origin> ---
            <content>
            --- End of <origin> ---

        Returns an empty string if *files* is empty.
        """
        if not files:
            return ""

        sections: list[str] = []
        for ctx in files:
            sections.append(
                f"--- Context from {ctx.origin} ---\n"
                f"{ctx.content}\n"
                f"--- End of {ctx.origin} ---"
            )
        return "\n\n".join(sections)

    def load_context(self, cwd: Path, config_dir: Path | None = None) -> str:
        """Convenience: discover, process imports, concatenate.

        Returns the fully resolved context string ready for the system prompt,
        or an empty string when no GEMINI.md files are found.
        """
        files = self.discover(cwd, config_dir)
        processed: list[ContextFile] = []
        for ctx in files:
            resolved_content = self.process_imports(ctx.content, ctx.path.parent)
            processed.append(
                ContextFile(path=ctx.path, content=resolved_content, origin=ctx.origin)
            )
        return self.concatenate(processed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_read(path: Path) -> str | None:
        """Read a file's text content, returning ``None`` on any error."""
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
        return None

    @staticmethod
    def _find_project_root(cwd: Path) -> Path | None:
        """Walk up from *cwd* to find the nearest project root marker."""
        current = cwd
        while True:
            for marker in _PROJECT_ROOT_MARKERS:
                if (current / marker).exists():
                    return current
            parent = current.parent
            if parent == current:
                # Reached filesystem root without finding a marker.
                return None
            current = parent

    @staticmethod
    def _ancestry_from_root(cwd: Path, root: Path) -> list[Path]:
        """Return directories from *root* down to *cwd* inclusive, in order."""
        parts: list[Path] = []
        current = cwd
        while True:
            parts.append(current)
            if current == root:
                break
            parent = current.parent
            if parent == current:
                # Safety: should not happen if root is an ancestor of cwd.
                break
            current = parent
        parts.reverse()
        return parts

    @staticmethod
    def _origin_label(
        md_path: Path,
        cwd: Path,
        project_root: Path,
        config_dir: Path,
    ) -> str:
        """Produce a human-readable origin label for a GEMINI.md path."""
        parent = md_path.parent
        if parent == project_root:
            return f"project ({md_path})"
        if parent == cwd:
            return f"directory ({md_path})"
        return f"directory ({md_path})"

    def _resolve_imports(
        self,
        content: str,
        base_dir: Path,
        seen: set[Path],
    ) -> str:
        """Recursively resolve ``@import`` directives."""

        def _replacer(match: re.Match[str]) -> str:
            rel_path = match.group(1)
            target = (base_dir / rel_path).resolve()

            if target in seen:
                logger.warning("Circular import detected: %s", target)
                return f"<!-- circular import: {rel_path} -->"

            seen.add(target)

            try:
                if not target.is_file():
                    logger.warning("Imported file not found: %s", target)
                    return f"<!-- import not found: {rel_path} -->"

                imported = target.read_text(encoding="utf-8")
                # Recurse to handle nested imports.
                return self._resolve_imports(imported, target.parent, seen)
            except OSError as exc:
                logger.warning("Failed to read import %s: %s", target, exc)
                return f"<!-- import error: {rel_path} ({exc}) -->"

        return _IMPORT_RE.sub(_replacer, content)
