"""FileDiscovery — index project files for @file reference completion and validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePath

logger = logging.getLogger(__name__)

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".zip",
        ".tar",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pdf",
    }
)


class FileDiscovery:
    """Index project files for @file reference completion and validation."""

    def __init__(self, root: Path, ignore_patterns: list[str] | None = None) -> None:
        """
        Args:
            root: Project root directory to index from.
            ignore_patterns: Glob patterns for files to exclude (e.g., from .pygeminiignore).
        """
        self._root = root
        self._ignore_patterns = ignore_patterns or []
        self._default_ignores: list[str] = [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*.egg-info",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
        ]

        # Auto-load .pygeminiignore if present at the project root.
        ignore_file = root / ".pygeminiignore"
        if ignore_file.exists():
            loaded = self.load_ignore_file(ignore_file)
            self._ignore_patterns.extend(loaded)

    @classmethod
    def from_project(cls, project_root: Path) -> "FileDiscovery":
        """Create a FileDiscovery configured for a project, auto-loading .pygeminiignore."""
        return cls(root=project_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[Path]:
        """Walk the project tree and return all non-ignored file paths.

        Returns paths relative to root.  Uses ``os.walk()`` for efficiency.
        Applies default ignores plus any custom ``ignore_patterns`` supplied at
        construction time.  Binary files (identified by extension) are skipped.
        """
        results: list[Path] = []
        all_patterns = self._default_ignores + self._ignore_patterns

        for dirpath, dirs, files in os.walk(self._root, topdown=True):
            current_dir = Path(dirpath)

            # Prune ignored directories in-place so os.walk skips their subtrees.
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore_name(d, all_patterns)
            ]

            for filename in files:
                file_path = current_dir / filename
                relative = file_path.relative_to(self._root)

                if self._should_ignore(relative, all_patterns):
                    continue

                if file_path.suffix.lower() in _BINARY_EXTENSIONS:
                    continue

                results.append(relative)

        logger.debug(
            "FileDiscovery: indexed %d files under %s", len(results), self._root
        )
        return results

    def resolve_reference(self, ref: str) -> Path | None:
        """Resolve a @file reference to an absolute path.

        - If *ref* is absolute, return it directly (if it exists).
        - If relative, resolve against the project root.
        - Glob patterns are supported; the first match is returned.

        Returns ``None`` if nothing is found.
        """
        ref_path = Path(ref)

        # Absolute path — validate existence and return.
        if ref_path.is_absolute():
            return ref_path if ref_path.exists() else None

        # Try direct resolution first (handles both plain names and sub-paths).
        candidate = self._root / ref_path
        if candidate.exists():
            return candidate.resolve()

        # Fall back to glob expansion relative to root.
        matches = list(self._root.glob(ref))
        if matches:
            logger.debug(
                "FileDiscovery: glob '%s' resolved to %s (first of %d match(es))",
                ref,
                matches[0],
                len(matches),
            )
            return matches[0].resolve()

        logger.debug("FileDiscovery: could not resolve reference '%s'", ref)
        return None

    def load_ignore_file(self, path: Path) -> list[str]:
        """Load patterns from a .pygeminiignore file.

        Lines starting with ``#`` are treated as comments and are skipped.
        Blank lines are also skipped.  Returns an empty list if the file does
        not exist or cannot be read.
        """
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.debug("FileDiscovery: could not read ignore file %s", path)
            return []

        patterns: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)

        logger.debug(
            "FileDiscovery: loaded %d pattern(s) from %s", len(patterns), path
        )
        return patterns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_ignore(self, path: Path, patterns: list[str]) -> bool:
        """Check if *path* (relative to root) matches any ignore pattern."""
        for pattern in patterns:
            # Match against the full relative path and against each individual part
            # so that patterns like "__pycache__" prune nested directories too.
            if PurePath(path).match(pattern):
                return True
            for part in path.parts:
                if PurePath(part).match(pattern):
                    return True
        return False

    def _should_ignore_name(self, name: str, patterns: list[str]) -> bool:
        """Check if a bare directory or file *name* matches any ignore pattern."""
        for pattern in patterns:
            if PurePath(name).match(pattern):
                return True
        return False
