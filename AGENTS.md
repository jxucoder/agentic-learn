# AGENTS.md

Instructions for AI coding agents working in `agentic-learn`.

## Tooling Baseline

- Use `uv` as the default Python toolchain for this repository.
- Do not use `pip`, `python -m venv`, or `poetry` unless explicitly requested.
- Prefer `uv run ...` for any Python command so execution is always in the project environment.

## Environment Setup

```bash
# One-time setup
uv sync --dev
```

## Common Commands

```bash
# Run tests
uv run pytest tests/ -v --tb=short

# Lint and formatting checks
uv run ruff check src/ examples/ tests/
uv run ruff format --check src/ examples/ tests/

# Run a benchmark/example
uv run python examples/synth_classification.py --seed 42 --steps 10
```

## Dependency Management

```bash
# Add runtime dependency
uv add <package>

# Add dev dependency
uv add --dev <package>

# Refresh lockfile/environment
uv sync --dev
```

## Project Notes

- Main package code lives in `src/aglearn/`.
- Example runs live in `examples/`.
- Tests live in `tests/`.
- Generated experiment artifacts are written under `output/`.

## Change Guidelines

- Keep changes focused and minimal.
- Follow existing code style and keep public APIs stable unless asked to change them.
- Run relevant tests and lint checks with `uv run` before finalizing.
