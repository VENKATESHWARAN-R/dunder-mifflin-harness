"""JAC — Just Another CLI."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__", "build_agent"]

try:
    __version__ = version("jac")
except PackageNotFoundError:
    __version__ = "0.4.1"


def __getattr__(name: str) -> object:
    if name == "build_agent":
        from jac.agents import config_loader

        return config_loader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
