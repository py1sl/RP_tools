# Utilities

This directory contains **shared** data-handling classes and common calculation
functions used across all RP_tools tool packages.

Individual tools (e.g. Gaussian plume model, skin dose model) each live in
their own top-level folder and import from this package. New tool-specific
logic should go in the relevant tool folder, while genuinely reusable
functions belong here.

## Modules

### `nuclide.py`

Provides the `Nuclide` data class and the `load_nuclides()` helper that reads
`data/nuclides.json` and returns a dictionary of `Nuclide` objects keyed by
nuclide name.  Used by any tool that needs radionuclide properties.

**Quick example**

```python
from utilities.nuclide import load_nuclides

nuclides = load_nuclides()
co60 = nuclides["Co60"]
print(co60.half_life_seconds)   # 1.66348e8
print(co60.gamma_lines)         # [{'energy_MeV': 1.1732, ...}, ...]
```

### `radioactive_decay.py`

Pure-function module for radioactive decay calculations.  These functions are
used by multiple tools (e.g. inventory calculations in a plume model, source
term decay for skin dose).

| Function | Description |
|---|---|
| `activity_at_time(A0, half_life, t)` | Activity after time *t* |
| `decays_in_period(A0, half_life, t_start, duration)` | Total decays in a time window |
| `time_to_activity(A0, A_target, half_life)` | Time to reach a target activity |
| `decay_constant(half_life)` | Decay constant λ from half-life |

All times and half-lives must be supplied in **consistent units** (seconds are
recommended). Activities are in Bq (decays per second) unless the caller uses
different but consistent units.

**Quick example**

```python
from utilities.radioactive_decay import activity_at_time, decays_in_period

# Co-60 half-life in seconds
T_HALF = 1.66348e8

# Activity after one year (3.15576e7 s)
A = activity_at_time(3.7e10, T_HALF, 3.15576e7)

# Decays during a 1-hour measurement
N = decays_in_period(3.7e10, T_HALF, t_start=0, duration=3600)
```

### `icrp_data.py`

Class-based loader for external dose-coefficient tables from:

* `data/icrp74/*.txt`
* `data/icrp116/*.txt`

Use this module when tools need geometry-specific conversion coefficients
(`AP`, `PA`, `ISO`, etc.) as tabulated by ICRP.

**Quick example**

```python
from utilities.icrp_data import ICRPDataLibrary

lib = ICRPDataLibrary()
table = lib.get_table("116", "photons")

energies = table.energies_MeV
ap_coeff = table.column("AP")
```

### `immersion_dose.py`

Model-agnostic post-processing utilities that convert concentration data
(`Bq/m^3`) into external immersion effective dose rate (`Sv/s`) using nuclide
gamma emission lines and ICRP photon dose-per-fluence coefficients.

This module is independent of how concentrations are produced (Gaussian plume,
CFD, measurements, etc.).

**Quick example**

```python
from utilities.immersion_dose import ImmersionDoseCalculator

calc = ImmersionDoseCalculator(geometry="ISO", publication="116")

# Point concentration (per nuclide)
dose_point = calc.dose_rate_from_concentration({"Cs137": 2.5e3})

# Grid concentration (per nuclide)
dose_grid = calc.dose_rate_on_grid({"Cs137": concentration_grid})
```

### `ground_plane_dose.py`

Model-agnostic post-processing utilities for external gamma dose from
deposited activity on the ground (semi-infinite plane approximation).

Input is deposition (`Bq/m^2`) per nuclide at points or on grids; output is
dose rate (`Sv/s`) per nuclide using nuclide gamma lines and ICRP photon
dose-per-fluence coefficients.

**Quick example**

```python
from utilities.ground_plane_dose import SemiInfinitePlaneDoseCalculator

calc = SemiInfinitePlaneDoseCalculator(geometry="ISO", publication="116")

# Point deposition (per nuclide)
dose_point = calc.dose_rate_from_deposition({"Cs137": 2.5e5})

# Grid deposition (per nuclide)
dose_grid = calc.dose_rate_on_grid({"Cs137": deposition_grid})
```

## Adding New Shared Modules

Place new shared utility modules directly in this directory and export them
from the package-level `__init__.py`.  Each module should be accompanied by
tests in `tests/`.  If a function is only used by one specific tool, put it
in that tool's own folder instead.
