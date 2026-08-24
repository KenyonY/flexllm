# Repository Guidelines

## Project Structure & Module Organization

`flexllm/` contains the library and CLI. Provider implementations live in `flexllm/clients/`,
concurrency primitives in `async_api/`, caching in `cache/`, and multimodal preprocessing in
`msg_processors/`. Keep reusable pricing and bundled web data under `flexllm/pricing/` and
`flexllm/data/`. Tests are grouped by scope in `tests/unit/`, `tests/integration/`, and `tests/e2e/`;
`tests/test_mock_e2e.py` exercises complete flows without external credentials. Put runnable usage
samples in `examples/` and user-facing documentation in `docs/`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[test]"` installs an editable checkout with the same test extras used
  by CI. Use `.[all]` when developing every optional provider or media feature.
- `pytest tests/unit/ -q --tb=short` runs the fast unit suite used by the pre-push hook.
- `pytest tests/test_mock_e2e.py -v --tb=short` validates routing, caching, and resume behavior
  against the local mock server.
- `pytest -m "not slow"` runs all discovered tests except those marked slow.
- `ruff check flexllm tests` and `ruff format --check flexllm tests` reproduce CI lint checks.
  Use `ruff check --fix ...` and `ruff format ...` before committing.
- `pre-commit install --hook-type pre-commit --hook-type pre-push` enables repository hooks.
- `python -m build` creates wheel and source distributions in `dist/` when the `build` package is
  installed.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, and keep lines within Ruff's 100-character limit.
Ruff 0.9.7 handles formatting, import ordering, and Pyflakes checks. Use `snake_case` for modules,
functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep provider
differences inside client subclasses rather than branching throughout shared request logic.

## Testing Guidelines

Pytest discovers `test_*.py` files and `test_*` functions; async tests run automatically through
`pytest-asyncio`. Add regression tests at the narrowest suitable tier. External E2E tests may need
`GEMINI_API_KEY` or `SILICONFLOW_API_KEY`; never make them a prerequisite for credential-free unit
coverage. Declare new test dependencies in the `test` extra in `pyproject.toml`.

## Commit & Pull Request Guidelines

Follow the established Conventional Commit form: `feat(clients): ...`, `fix(cache): ...`,
`docs(readme): ...`, or `chore: ...`. Keep each commit focused. Pull requests should explain the
behavioral change, link relevant issues, list exact verification commands, and update docs or
examples when public behavior changes. Include terminal output or screenshots for observable CLI
or web UI changes.

## Security & Configuration

Start from `flexllm_config.example.yaml`; keep API keys in environment variables or an ignored local
configuration file. Never commit credentials, generated result JSONL files, or private endpoint
URLs.
