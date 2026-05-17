"""Per-run in-memory cache for large tool outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class ToolResultCache:
    """Run-scoped store for verbatim tool outputs referenced by handle."""

    _store: dict[str, str] = field(default_factory=dict)
    max_entries: int = 64

    def store(self, content: str) -> str:
        handle = uuid4().hex[:12]
        if len(self._store) >= self.max_entries:
            oldest = next(iter(self._store))
            self._store.pop(oldest, None)
        self._store[handle] = content
        return handle

    def fetch(self, handle: str) -> str | None:
        return self._store.get(handle)

    def clear(self) -> None:
        self._store.clear()
