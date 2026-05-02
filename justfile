set positional-arguments := true

# Set default recipe to list all available commands
default:
    @just --list

# Initialize or sync the virtual environment (core + dev + lab)
sync:
    uv sync --all-groups --all-extras

# Install JAC as a global uv tool from this checkout
install:
    uv tool install .

# Reinstall the global JAC tool from this checkout after local changes
upgrade:
    uv tool install --reinstall .

# Lint the code using ruff
lint:
    uvx ruff check src tests

# Format the code using ruff
format:
    uvx ruff format src tests

# Lints and formats the code with fixes
fix:
    uvx ruff check --fix src tests
    uvx ruff format src tests

# Type check the code using ty
typecheck:
    uvx ty check src tests

# Run the application with optional arguments
run *args:
    uv run --env-file .env {{args}}

# Run tests using pytest
test:
    uv run pytest tests

# Clean up build artifacts and tool caches (ruff, pytest, mypy, coverage, etc.)
clean:
    rm -rf dist/ build/ \
        .ruff_cache/ .pytest_cache/ .mypy_cache/ .pytype/ .hypothesis/ \
        htmlcov/ .cache/
    rm -f .coverage .coverage.* .dmypy.json dmypy.json
    find . -depth -type d -name "__pycache__" -exec rm -rf {} +
    find . -depth -type d -name "*.egg-info" -exec rm -rf {} +
