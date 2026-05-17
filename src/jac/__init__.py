"""JAC — Just Another CLI. A multi-agent dev harness with tiered model routing."""

try:
    from importlib.metadata import version

    __version__ = version("jac")
except Exception:
    __version__ = "0.5.0-dev"
