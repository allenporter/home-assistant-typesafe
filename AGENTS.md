# Project Style & Testing Guidelines

## Testing Principles

- **Fakes Over Mocks**: Prefer real Home Assistant registries (`ar.async_get(hass)`, `er.async_get(hass)`, `dr.async_get(hass)`), real state entities, and dedicated in-memory fakes (e.g., `FakeDecisionEngine`) over `unittest.mock.MagicMock`.
- **No Redundant Type Checking**: Do not write `assert isinstance(...)` type assertions in tests. Trust static typing and focus assertions on behavior and outcomes.
- **Parallel Test Hierarchy**: Test directories and test filenames must match the source tree 1-to-1 (e.g., `retrieval/lexical.py` -> `tests/.../retrieval/test_lexical.py`).
- **Test-Driven Development**: Write stage and unit tests first, verifying expectations before implementing.

## Coding Style & Patterns

- **No Numbered Comments**: Do not use numbered lists in code comments (e.g. `# 1. ...`, `# 2. ...`). Write concise, descriptive comments or self-documenting code.
- **No Broad Exception Handlers**: Do not use `except Exception:` unless specifically justified at an external boundary. Handle specific exception types or use defensive guards.
- **Package `__init__.py` Conventions**: Do not add `__all__` forwarding everywhere or re-import submodules into `__init__.py`. Instead, write comprehensive module docstrings explaining the package role and principles; import directly from the relevant module.
