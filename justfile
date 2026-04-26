set positional-arguments := true

# Set default recipe to list all available commands
default:
    @just --list

# Initialize or sync the virtual environment
sync:
    uv sync --all-groups --all-extras

# Lint the code using ruff
lint:
    uvx ruff check main.py src tests

# Format the code using ruff
format:
    uvx ruff format main.py src tests

# Lints and formats the code with fixes
fix:
    uvx ruff check --fix main.py src tests
    uvx ruff format main.py src tests

# Type check the code using ty
typecheck:
    uvx ty check main.py src tests

# Run the application with optional arguments
run *args:
    uv run --group harness --env-file .env {{args}}

# Run tests using pytest
test:
    uv run pytest tests

# Clean up build artifacts
clean:
    rm -rf dist/
    find . -type d -name "__pycache__" -exec rm -rf {} +
