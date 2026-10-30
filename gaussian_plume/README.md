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
    half_lives={"Co60": 1.663e8, "Cs137": 9.496e8},  # seconds (optional)
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

### `dry_deposition.py`

Provides the `DryDepositionModel` class, which calculates the ground-surface
concentration (Bq/m²) from dry deposition of airborne radionuclides, following
the ADMC / NRPB-R91 methodology.

#### Background

Dry deposition is modelled as a *deposition velocity* process.  The
instantaneous deposition flux onto the ground surface is:

```
F_d(x, y)  =  V_d  ×  χ(x, y, 0)          [Bq/m²/s]
```

Integrating over an exposure time *T* gives the ground-surface concentration:

```
σ(x, y)  =  V_d  ×  χ(x, y, 0)  ×  T      [Bq/m²]
```

The **no-depletion** approximation is used (NRPB-R91, §3): the plume
concentration is not reduced to account for material removed by deposition.
This is appropriate for routine assessment calculations where the deposition
velocity is small relative to *ū × σz*.

**Typical deposition velocities (NRPB-R91 / ADMC):**

| Material type                     | V_d (m/s) |
|-----------------------------------|-----------|
| Particles / aerosols              | 0.001     |
| Reactive gases (e.g. elemental I) | 0.01      |
| Inert / organic vapours           | 0.001     |

#### Usage

```python
from gaussian_plume.plume import GaussianPlume
from gaussian_plume.dry_deposition import DryDepositionModel

plume = GaussianPlume(
    release={"Cs137": 1.0e6, "Co60": 3.7e10},  # Bq/s per nuclide
    wind_speed=2.0,                              # m/s
    stability_category="D",
    release_height=50.0,                         # metres above ground
)

# One hour of continuous release with default 1 mm/s deposition velocity
model = DryDepositionModel(plume, integration_time_s=3600.0)

# Ground concentration on the plume centreline at 1 km (Bq/m²)
sigma = model.centreline_ground_concentration(x=1000.0)
# {"Cs137": <float> Bq/m², "Co60": <float> Bq/m²}

# Ground concentration at an arbitrary (x, y) point
sigma_off = model.ground_concentration(x=1000.0, y=200.0)

# Instantaneous deposition rate (Bq/m²/s) at a point
fd = model.deposition_rate(x=1000.0, y=0.0)

# Per-nuclide deposition velocities
model_mixed = DryDepositionModel(
    plume,
    deposition_velocities={"Cs137": 0.001, "Co60": 0.001},
    integration_time_s=3600.0,
)

# Ground-surface concentration map on a spatial grid (Bq/m²)
grid = model.ground_concentration_on_grid(
    x_edges=[0, 500, 1000, 2000, 5000, 10000],
    y_edges=[-1000, -500, -250, 0, 250, 500, 1000],
)
# {"Cs137": (5, 6) ndarray Bq/m², "Co60": (5, 6) ndarray Bq/m²}
```

## Radioactive decay during transport

When the optional *half_lives* parameter is supplied to `GaussianPlume`, the
effective release rate at each downwind distance is reduced to account for
radioactive decay during atmospheric transport:

```
Q_eff(x) = Q · exp(−λ · x / ū)
```

where `λ = ln(2) / T½` is the decay constant and `x / ū` is the estimated
travel time from source to receptor.  Nuclides not listed in *half_lives* are
treated as stable (no decay correction applied).

```python
plume = GaussianPlume(
    release={"Co60": 3.7e10, "Cs137": 1.0e9},
    wind_speed=2.0,
    stability_category="D",
    release_height=50.0,
    half_lives={"Co60": 1.663e8, "Cs137": 9.496e8},  # seconds
)
chi = plume.centreline_concentration(x=1000.0)
```

## Limitations

The implementation provides core dispersion, dry deposition, and wet
deposition calculations. Deposition currently uses a no-depletion
approximation; wet deposition estimates ground deposition and does not calculate
ground-shine dose.

### NRPB-R91 corrections not yet implemented

* Building-wake and urban-roughness corrections
* Mixing-layer height capping
* Ground-shine dose calculation

These reference-backed corrections are the first candidates for extending the
current steady-state model. Other possible, broader extensions include
time-varying releases and meteorology, finite-duration or puff releases, terrain
effects, and uncertainty or sensitivity analysis. Each requires explicit
requirements, model-specific scientific references, and independent validation
before use in radiological assessment.

## Optional geographic map overlay

The optional `gaussian_plume.basemap` module overlays a ground-level plume on
a georeferenced raster without changing the model's local-metre calculations.
It includes a generic XYZ tile provider and a cache-first wrapper. Configure a
tile URL and attribution for a provider whose access and caching terms allow
your use; satellite providers may require an API key in the URL template.

```python
from gaussian_plume.basemap import CachedBasemap, XYZTileProvider, plot_plume_on_map

tiles = XYZTileProvider(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution="© OpenStreetMap contributors",
    source="OpenStreetMap",
    user_agent="YourApplication/1.0 (contact@example.org)",
)
basemap = CachedBasemap(tiles, cache_dir="~/.cache/rp-tools/maps")

ax = plot_plume_on_map(
    plume,
    source_longitude=-1.5,
    source_latitude=52.0,
    downwind_extent_m=5000,
    crosswind_extent_m=1500,
    wind_direction_deg=225,  # direction the wind blows toward; clockwise from north
    basemap=basemap,
    resolution_m=100,
)
```

`CachedBasemap` checks for an image covering the requested bounds at adequate
pixel density before downloading; set `offline=True` to raise a clear
`FileNotFoundError` on a cache miss. Cache metadata records provider identity,
geographic bounds, dimensions, attribution, and imagery date when supplied by
the provider. `BasemapImage` can also be passed directly for caller-managed
images. XYZ tiles use Web Mercator and are limited to approximately ±85.05°
latitude; requests crossing the antimeridian are not supported.

OpenStreetMap tiles provide a general-purpose map, not satellite imagery.
Other XYZ-compatible sources can provide aerial or satellite imagery, but
check each service's attribution, API, display, and caching requirements before
use; the cache does not override provider terms. Imagery may be dated or
incomplete. The overlay uses a local tangent-plane approximation and the
underlying plume model does not account for terrain or buildings, so this is a
visualisation aid rather than a site-level prediction.
