# Set default recipe to list all available commands
default:
    @just --list

# Initialize or sync the virtual environment
sync:
    uv sync --all-groups --all-extras

# Lint the code using ruff
lint:
    uvx ruff check .

# Format the code using ruff
format:
    uvx ruff format .

# Lints and formats the code with fixes
fix:
    uvx ruff check --fix .
    uvx ruff format .

# Type check the code using ty
typecheck:
    uvx ty check

# Run the main application
run:
    uv run python main.py

# Run tests using pytest
test:
    uv run pytest

# Clean up build artifacts
clean:
    rm -rf dist/
    find . -type d -name "__pycache__" -exec rm -rf {} +
