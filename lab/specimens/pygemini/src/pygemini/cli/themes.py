"""Terminal color themes for PyGeminiCLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Color theme for terminal output."""

    prompt_color: str = "green"
    model_color: str = "cyan"
    tool_color: str = "yellow"
    error_color: str = "red"
    info_color: str = "blue"
    success_color: str = "green"
    dim_color: str = "dim"


DEFAULT_THEME = Theme()

_THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
}


def get_theme(name: str = "default") -> Theme:
    """Get a theme by name. Falls back to default if not found."""
    return _THEMES.get(name, DEFAULT_THEME)
