# Documentation

This directory contains documentation for the RP_tools project.

## Overview

RP_tools is a collection of radiation protection calculation tools. Each
distinct tool lives in its own top-level folder and is built on top of shared
data-handling and common utility functions provided by the `utilities` package.

## Planned Tools

Each tool below will occupy its own folder at the root of the repository:

| Folder | Description | Status |
|---|---|---|
| `gaussian_plume/` | Atmospheric dispersion and air-dose modelling (NRPB-R91) | Implemented |
| `skin_dose/` | Dose to skin from beta and photon sources | Planned |
| `ingestion_dose/` | Internal dose via the ingestion pathway | Planned |
| `biokinetic/` | Systemic distribution of radionuclides | Planned |

Radioactive decay calculations are provided as a shared utility in
`utilities/radioactive_decay.py` and consumed by multiple tools.

## Structure

```
RP_tools/
├── docs/             # Project documentation (this directory)
├── data/             # Shared reference data (nuclides, dose coefficients, …)
├── utilities/        # Common data-handling classes and shared calculation functions
├── gaussian_plume/   # Gaussian plume dispersion tool (NRPB-R91)
├── skin_dose/        # (planned) Skin dose tool
├── ingestion_dose/   # (planned) Ingestion dose tool
└── tests/            # Pytest test suite covering utilities and all tools
```

## Getting Started

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/
```

See the individual `README.md` files in each subdirectory for details on the
data formats, module APIs, and usage examples.  The test suite uses
[pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/)
for coverage reporting and [flake8](https://flake8.pycqa.org/) for style checking.
