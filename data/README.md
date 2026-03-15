# Data

This directory contains reference data files used by RP_tools.

## Files

### `nuclides.json`

A JSON database of nuclide properties. Each entry is keyed by the canonical
nuclide name (e.g. `"Co60"`, `"Cs137"`) and contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Short identifier, e.g. `"Co60"` |
| `long_name` | string | Human-readable name, e.g. `"Cobalt-60"` |
| `symbol` | string | Element symbol, e.g. `"Co"` |
| `A` | int | Mass number |
| `Z` | int | Atomic number |
| `stable` | bool | `true` if the nuclide is stable |
| `half_life_seconds` | float | Half-life in seconds (unstable only) |
| `half_life_years` | float | Half-life in years for convenience (unstable only) |
| `decay_modes` | array | List of decay mode objects (unstable only) |
| `gamma_lines` | array | Gamma/X-ray emission lines (unstable only) |
| `x_ray_lines` | array | Characteristic X-ray lines (unstable only, where applicable) |
| `beta_lines` | array | Beta endpoint energies and intensities (unstable only, where applicable) |

#### Decay mode object

```json
{
  "mode": "beta-",
  "branching_fraction": 1.0,
  "q_value_MeV": 2.824,
  "endpoint_energy_MeV": 0.318,
  "daughter": "Ni60"
}
```

#### Gamma line object

```json
{
  "energy_MeV": 1.1732,
  "intensity_percent": 99.85
}
```

## Adding New Nuclides

Add a new key to `nuclides.json` following the schema above. Nuclear data
should be sourced from evaluated nuclear data libraries such as ENSDF, NUBASE,
or IAEA's Live Chart of Nuclides.
