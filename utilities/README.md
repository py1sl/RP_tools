# Utilities

This directory contains the core calculation modules and data-handling classes
for RP_tools.

## Modules

### `nuclide.py`

Provides the `Nuclide` data class and the `load_nuclides()` helper that reads
`data/nuclides.json` and returns a dictionary of `Nuclide` objects keyed by
nuclide name.

**Quick example**

```python
from utilities.nuclide import load_nuclides

nuclides = load_nuclides()
co60 = nuclides["Co60"]
print(co60.half_life_seconds)   # 1.66348e8
print(co60.gamma_lines)         # [{'energy_MeV': 1.1732, ...}, ...]
```

### `radioactive_decay.py`

Pure-function module for radioactive decay calculations.

| Function | Description |
|----------|-------------|
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

## Adding New Modules

Place new utility modules directly in this directory and import them from the
package-level `__init__.py` if appropriate. Each module should be accompanied
by tests in `tests/`.
