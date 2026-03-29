# RP_tools

A collection of radiation protection calculation tools built around shared
utility and data-handling functions.

## Project Structure

Each distinct tool lives in its own top-level folder. The `utilities` package
provides common data classes and calculation functions (radioactive decay,
nuclide data, etc.) that are shared across all tools.

```
RP_tools/
├── docs/             # Project-level documentation
├── data/             # Shared reference data (nuclides, dose coefficients, …)
├── utilities/        # Common data-handling classes and shared functions
├── gaussian_plume/   # Gaussian plume dispersion model (NRPB-R91)
├── skin_dose/        # (planned) Skin dose model
├── ingestion_dose/   # (planned) Ingestion dose model
└── tests/            # Pytest test suite covering utilities and all tools
```

## Getting Started

Install the package in editable mode with development dependencies, then run
the test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

Alternatively, if you prefer the legacy `requirements.txt` workflow:

```bash
pip install -r requirements.txt
pytest tests/
```

See `docs/README.md` and the `README.md` in each subdirectory for further
details.