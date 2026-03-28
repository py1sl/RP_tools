"""gaussian_plume – NRPB-R91 Gaussian plume atmospheric dispersion model.

This package implements the atmospheric dispersion model described in NRPB-R91
(Simmonds et al., 1993), providing:

* Six Pasquill-Gifford stability categories (A–F).
* Clarke (1979) σy and σz dispersion coefficients for open-country conditions.
* Continuous elevated point-source air concentration calculations.

Typical usage::

    from gaussian_plume.plume import GaussianPlume

    plume = GaussianPlume(
        release={"Cs137": 1.0e6, "Co60": 3.7e10},
        wind_speed=2.0,
        stability_category="D",
        release_height=50.0,
    )
    chi = plume.centreline_concentration(x=1000.0)
    # {"Cs137": 2.31e-05, "Co60": 8.55e+05}  (Bq/m³)

Reference:
    Simmonds, J. R. et al. (1993). *The Methodology for Assessing the
    Radiological Consequences of Routine Releases of Radionuclides to the
    Atmosphere*.  NRPB-R91, National Radiological Protection Board, Chilton, UK.
"""

from gaussian_plume.dispersion import STABILITY_CATEGORIES, sigma_y, sigma_z
from gaussian_plume.plume import GaussianPlume
from gaussian_plume.dry_deposition import DEFAULT_DEPOSITION_VELOCITY_M_S, DryDepositionModel

__all__ = [
    "GaussianPlume",
    "STABILITY_CATEGORIES",
    "sigma_y",
    "sigma_z",
    "DryDepositionModel",
    "DEFAULT_DEPOSITION_VELOCITY_M_S",
]
