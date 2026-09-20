# Contributing

Contributions are welcome! Please follow these guidelines for local development, testing, and submitting pull requests.

## Environment Setup

Initialize the virtual environment and development dependencies:

```bash
# Bootstrap dependencies
script/bootstrap

# Setup development environment
script/setup
```

## Running Tests

Run the test suite:

```bash
script/test
```

To run a specific test file:

```bash
uv run pytest tests/test_conversation.py
```

## Code Quality & Linting

All changes must pass formatting and static analysis checks:

```bash
script/lint
```

This runs:

- `ruff` (formatting and linting)
- `ty` (static type checking)
- `codespell` (spell check)
- `yamllint` (YAML syntax)
- `prettier` (markdown and JSON formatting)
