# Documentation

This directory contains documentation for the RP_tools project.

## Overview

RP_tools is a collection of radiation protection calculation tools built around
common utility and data handling functions.

## Planned Tools

- **Radioactive Decay** – Activity calculations, decay chain support
- **Gaussian Plume Model** – Atmospheric dispersion and dose modelling
- **Skin Dose Models** – Dose to skin from beta and photon sources
- **Ingestion Dose Modelling** – Internal dose via ingestion pathway
- **Biokinetic Models** – Systemic distribution of radionuclides (future)

## Structure

```
RP_tools/
├── docs/        # Project documentation (this directory)
├── data/        # Reference data files (nuclides, dose coefficients, etc.)
├── utilities/   # Core calculation modules and data classes
└── tests/       # Pytest test suite
```

## Getting Started

See the individual `README.md` files in each subdirectory for details on the
data formats, module APIs, and usage examples.

Dependencies are listed in `requirements.txt` (to be added). The test suite
uses [pytest](https://docs.pytest.org/).
