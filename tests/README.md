# Tests

This directory contains the pytest test suite for RP_tools.

## Running Tests

From the repository root:

```bash
pytest tests/
```

For verbose output:

```bash
pytest -v tests/
```

For a coverage report (requires `pytest-cov`):

```bash
pytest --cov=utilities tests/
```

## Test Files

| File | Description |
|------|-------------|
| `conftest.py` | Shared pytest fixtures |
| `test_nuclide.py` | Tests for the `Nuclide` class and `load_nuclides()` loader |
| `test_radioactive_decay.py` | Tests for the radioactive decay calculation functions |

## Conventions

- All test files are named `test_*.py`.
- Tests use `pytest` fixtures defined in `conftest.py` where data is shared
  between multiple test modules.
- Physical results are validated against known analytical values; tolerances
  are documented inline.
