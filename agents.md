# Agent Guide: RP_tools

This document helps AI agents and contributors quickly understand the repository structure, development workflow, and code conventions for the **RP_tools** project.

---

## Project Overview

**RP_tools** is a Python library of modular **radiation protection calculation tools**. All tools share a common utilities layer (nuclide data handling, radioactive decay) and are designed to be independently usable.

| Status | Tool |
|--------|------|
| ✅ Implemented | Gaussian Plume atmospheric dispersion model (NRPB-R91) |
| ✅ Implemented | Utilities: nuclide data loader, radioactive decay functions |
| ⏳ Planned | Skin dose model (beta/photon) |
| ⏳ Planned | Ingestion dose model (internal exposure) |
| ⏳ Planned | Biokinetic models (radionuclide distribution) |

---

## Repository Structure

```
RP_tools/
├── agents.md                       # This file
├── README.md                       # Project overview and quick start
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration (sets pythonpath = .)
├── .flake8                         # Flake8 style configuration (max-line-length = 127)
├── .gitignore
├── LICENSE
├── .github/
│   └── workflows/
│       └── tests.yml               # CI: lint with flake8 + test with pytest
├── data/
│   ├── README.md                   # nuclides.json schema and data sources
│   └── nuclides.json               # Nuclear data for 7 nuclides (Co60, Fe55, Cs137, H3, Sr90, Fe56, Co59)
├── docs/
│   └── README.md                   # High-level architecture and planned tools
├── utilities/
│   ├── __init__.py
│   ├── README.md                   # Utilities API reference and examples
│   ├── nuclide.py                  # Nuclide dataclass + load_nuclides() JSON loader
│   └── radioactive_decay.py        # Pure-function decay calculations
├── gaussian_plume/
│   ├── __init__.py
│   ├── README.md                   # Background, usage, equations, limitations
│   ├── dispersion.py               # Clarke (1979) σy and σz dispersion coefficients
│   └── plume.py                    # GaussianPlume class (main model)
└── tests/
    ├── README.md                   # Testing conventions and how to run tests
    ├── conftest.py                 # Shared pytest fixtures (nuclides, data_dir)
    ├── test_nuclide.py             # Tests for Nuclide class and load_nuclides()
    ├── test_radioactive_decay.py   # Tests for decay calculation functions
    └── test_gaussian_plume.py      # Tests for dispersion coefficients and plume model
```

---

## Setup and Installation

**Requirements:** Python 3.11+

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥9.0 | Test runner |
| `pytest-cov` | ≥7.0 | Test coverage reporting |
| `flake8` | ≥7.0 | PEP 8 style linting |
| `numpy` | ≥1.24 | Numerical arrays and vectorised calculations |
| `matplotlib` | ≥3.7 | Plotting (optional, only for visualisation) |

There is **no build step** — the project is pure Python. Imports work relative to the repository root (configured in `pytest.ini` via `pythonpath = .`).

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run with coverage report
pytest --cov=utilities --cov=gaussian_plume --cov-report=term-missing tests/

# Run a single test file
pytest tests/test_gaussian_plume.py

# Run a single test by name
pytest tests/test_radioactive_decay.py::test_activity_at_time
```

**Test layout:**

| Test file | What it covers |
|-----------|---------------|
| `tests/test_nuclide.py` | `Nuclide` dataclass validation, `load_nuclides()` JSON parsing |
| `tests/test_radioactive_decay.py` | `decay_constant()`, `activity_at_time()`, `decays_in_period()`, `time_to_activity()` |
| `tests/test_gaussian_plume.py` | Clarke dispersion coefficients, `GaussianPlume` concentration calculations, grid output, plotting |

**Shared fixtures** (`tests/conftest.py`):

```python
@pytest.fixture
def nuclides():
    """Returns the full nuclide database as a dict keyed by nuclide name."""
    return load_nuclides()

@pytest.fixture
def data_dir():
    """Returns the Path to the data/ directory."""
    return Path(__file__).parent.parent / "data"
```

---

## Running the Linter

```bash
# Lint the whole project (max line length: 127)
flake8 .

# Lint a specific file
flake8 utilities/radioactive_decay.py
```

Flake8 is configured in `.flake8`:

```ini
[flake8]
max-line-length = 127
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/tests.yml`) runs on every push and pull request:

1. Install dependencies (`pip install -r requirements.txt`)
2. Lint with `flake8` (max line length 127)
3. Test with `pytest --cov=utilities --cov=gaussian_plume --cov-report=term-missing`

Tests run on **Ubuntu latest** with **Python 3.11 and 3.12**.

---

## Key Modules and APIs

### `utilities/nuclide.py`

```python
from utilities.nuclide import Nuclide, load_nuclides

nuclides = load_nuclides()          # dict[str, Nuclide] keyed by name e.g. "Co60"
co60 = nuclides["Co60"]

co60.name                           # "Co60"
co60.long_name                      # "Cobalt-60"
co60.A                              # 60  (mass number)
co60.Z                              # 27  (atomic number)
co60.N                              # 33  (neutron number, derived property)
co60.stable                         # False
co60.half_life_seconds              # float
co60.half_life_years                # float
co60.gamma_lines                    # list[dict] with "energy_MeV" and "intensity_percent"
```

`load_nuclides()` reads `data/nuclides.json` relative to the `data/` directory.

### `utilities/radioactive_decay.py`

Pure, stateless functions — no side effects:

```python
from utilities.radioactive_decay import (
    decay_constant,
    activity_at_time,
    decays_in_period,
    time_to_activity,
)

lam = decay_constant(half_life_s)                     # ln(2) / T½  [s⁻¹]
A_t = activity_at_time(A0, half_life_s, t_s)          # A0 · exp(−λt)  [Bq]
N   = decays_in_period(A0, half_life_s, t_s)          # total decays in interval
t   = time_to_activity(A0, A_target, half_life_s)     # time to reach A_target [s]
```

### `gaussian_plume/dispersion.py`

Implements Clarke (1979) dispersion coefficient equations for six Pasquill-Gifford stability categories (`"A"` very unstable → `"F"` moderately stable):

```python
from gaussian_plume.dispersion import sigma_y, sigma_z

sy = sigma_y(x=1000.0, stability_category="D")   # crosswind spread [m]
sz = sigma_z(x=1000.0, stability_category="D")   # vertical spread [m]
```

### `gaussian_plume/plume.py`

```python
from gaussian_plume.plume import GaussianPlume

plume = GaussianPlume(
    release={"Co60": 3.7e10, "Cs137": 1.0e9},  # source term {nuclide: Bq/s}
    wind_speed=2.0,                              # m/s (must be > 0)
    stability_category="D",                      # one of A–F
    release_height=50.0,                         # metres (must be >= 0)
)

# Concentration at a single point (x downwind, y crosswind, z height, all metres)
chi = plume.air_concentration(x=1000.0, y=0.0, z=0.0)     # dict[str, float] Bq/m³

# Centreline ground-level concentration (y=0, z=0)
chi = plume.centreline_concentration(x=1000.0)             # dict[str, float] Bq/m³

# Concentration on an (x, y) grid at ground level
grid = plume.concentration_on_grid(
    x_edges=[100, 500, 1000, 5000],
    y_edges=[-500, 0, 500],
)   # dict[str, np.ndarray] shape (nx, ny)

# Visualisation
fig, ax = plume.plot_xy_slice(x_edges, y_edges, nuclide="Co60")
fig, ax = plume.plot_centreline(x_values=[100, 500, 1000, 5000], nuclide="Co60")
```

---

## Code Conventions

### Style

- **Python 3.11+**; `from __future__ import annotations` at the top of every module for forward-reference type hints.
- **PEP 8** enforced by `flake8` with a max line length of **127 characters**.
- **Full type hints** on all function signatures and class attributes.
- **Docstrings** on every public class, method, and function (Google/NumPy style).

### Design Patterns

**Data classes with validation in `__init__`:**

```python
class Nuclide:
    def __init__(self, data: dict[str, Any]) -> None:
        self.name: str = data["name"]
        self.A: int = int(data["A"])
        if self.A <= 0:
            raise ValueError(f"A must be positive, got {self.A}")
    
    @property
    def N(self) -> int:
        """Neutron number (derived)."""
        return self.A - self.Z
```

**Pure functions for physics calculations:**

```python
def activity_at_time(A0: float, half_life: float, t: float) -> float:
    """Return activity at time t.

    Args:
        A0: Initial activity in Bq.
        half_life: Half-life in seconds.
        t: Elapsed time in seconds.

    Returns:
        Activity at time t in Bq.
    """
    lam = decay_constant(half_life)
    return A0 * math.exp(-lam * t)
```

**NumPy vectorisation for grid calculations:**

```python
y_arr = np.asarray(y_centres)               # shape (ny,)
factor = crosswind[:, np.newaxis]            # shape (ny, 1) for broadcasting
result = factor * vertical[np.newaxis, :]   # shape (ny, nz)
```

**Look-up tables with named tuples/dataclasses:**

```python
STABILITY_CATEGORIES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

_COEFFS: dict[str, _DispersionCoeffs] = {
    "A": _DispersionCoeffs(a_y=0.22, b_y=0.0001, ...),
    "D": _DispersionCoeffs(a_y=0.08, b_y=0.0001, ...),
}
```

**Optional `matplotlib` import (lazy, inside plotting methods):**

```python
def plot_xy_slice(self, ...):
    import matplotlib.pyplot as plt          # imported only when plotting is called
    from matplotlib.colors import LogNorm
    ...
```

### Testing Conventions

- One test file per module (e.g. `test_nuclide.py` ↔ `utilities/nuclide.py`).
- Use `pytest.approx` for floating-point comparisons.
- Use `pytest.raises` for expected exceptions.
- Shared test data via fixtures in `conftest.py`.
- Tests are self-contained and do not depend on external network access or files outside the repo.

```python
def test_activity_at_time_zero(nuclides):
    co60 = nuclides["Co60"]
    A0 = 1.0e9
    result = activity_at_time(A0, co60.half_life_seconds, 0.0)
    assert result == pytest.approx(A0, rel=1e-9)

def test_invalid_wind_speed():
    with pytest.raises(ValueError, match="wind_speed"):
        GaussianPlume({"Co60": 1e6}, wind_speed=-1.0, stability_category="D", release_height=50.0)
```

### Adding a New Tool

1. Create a new package directory (e.g. `skin_dose/`) with an `__init__.py` and a `README.md`.
2. Import shared utilities from `utilities/` — do not duplicate decay or nuclide logic.
3. Add a corresponding test file in `tests/` (e.g. `tests/test_skin_dose.py`).
4. Update the top-level `README.md` and `docs/README.md` to document the new tool.
5. If the new package needs coverage, add `--cov=skin_dose` to the CI test command in `.github/workflows/tests.yml`.

---

## Data Format (`data/nuclides.json`)

The top-level key is `"nuclides"`, which maps nuclide names to their data objects:

```json
{
  "nuclides": {
    "Co60": {
      "name": "Co60",
      "long_name": "Cobalt-60",
      "symbol": "Co",
      "A": 60,
      "Z": 27,
      "stable": false,
      "half_life_seconds": 166348000.0,
      "half_life_years": 5.2713,
      "decay_modes": [
        {"mode": "beta-", "branching_fraction": 0.9988, "q_value_MeV": 2.824,
         "endpoint_energy_MeV": 0.318, "daughter": "Ni60"}
      ],
      "gamma_lines": [
        {"energy_MeV": 1.1732, "intensity_percent": 99.85},
        {"energy_MeV": 1.3325, "intensity_percent": 99.98}
      ],
      "beta_lines": [
        {"endpoint_energy_MeV": 0.318, "intensity_percent": 99.88}
      ],
      "x_ray_lines": []
    }
  }
}
```

**Available nuclides:** Co60, Fe55, Cs137, H3 (Tritium), Sr90, Fe56, Co59.
