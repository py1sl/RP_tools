# gaussian_plume

Gaussian plume atmospheric dispersion model for RP_tools, based on the
methodology described in NRPB-R91 (Simmonds et al., 1993).

## Background

The model calculates the steady-state air concentration of radionuclides
downwind of a continuous elevated point source.  It uses the Clarke (1979)
dispersion parametrisation for open-country (rural) conditions as reproduced
in NRPB-R91.

### Supported weather categories

Six Pasquill-Gifford atmospheric stability categories are supported:

| Category | Description       | Typical conditions                         |
|----------|-------------------|--------------------------------------------|
| A        | Very unstable     | Strong insolation, light winds             |
| B        | Unstable          | Moderate insolation                        |
| C        | Slightly unstable | Slight insolation                          |
| D        | Neutral           | Overcast, or moderate-to-strong winds      |
| E        | Slightly stable   | Slight incoming solar radiation at night   |
| F        | Moderately stable | Low insolation or clear sky at night       |

## Modules

### `dispersion.py`

Provides the Clarke (1979) crosswind (σy) and vertical (σz) dispersion
coefficients as functions of downwind distance and stability category:

```python
from gaussian_plume.dispersion import sigma_y, sigma_z

sy = sigma_y(x=1000.0, category="D")   # metres
sz = sigma_z(x=1000.0, category="D")   # metres
```

The formula used is:

```
σ(x) = a · x · (1 + b · x)^c
```

where the constants `a`, `b`, `c` are taken from Table A1 of NRPB-R91.

### `plume.py`

Provides the `GaussianPlume` class.  The source term is supplied as a
dictionary mapping nuclide names to continuous release rates in Bq/s.

```python
from gaussian_plume.plume import GaussianPlume

plume = GaussianPlume(
    release={"Co60": 3.7e10, "Cs137": 1.0e9},  # Bq/s per nuclide
    wind_speed=2.0,                              # m/s
    stability_category="D",
    release_height=50.0,                         # metres above ground
)

# Ground-level centreline air concentration at 1 km downwind (Bq/m³)
chi = plume.centreline_concentration(x=1000.0)

# Air concentration at an arbitrary point (x, y, z) in metres
chi_off = plume.air_concentration(x=1000.0, y=100.0, z=0.0)
```

## Gaussian plume equation

The model implements the standard ground-reflection Gaussian plume formula:

```
χ(x, y, z) = Q / (2π · ū · σy · σz)
             · exp[−y² / (2σy²)]
             · { exp[−(z−H)² / (2σz²)] + exp[−(z+H)² / (2σz²)] }
```

where H is the effective release height, ū is the mean wind speed, and
x, y, z are the downwind, crosswind, and vertical coordinates.

## Limitations

The current implementation covers the core dispersion calculation only.
The following features from the full NRPB-R91 methodology are **not yet
implemented**:

* Dry and wet deposition / ground-shine
* Radioactive decay during atmospheric transport
* Building-wake and urban-roughness corrections
* Mixing-layer height capping
