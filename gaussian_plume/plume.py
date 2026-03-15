"""Gaussian plume model implementation based on NRPB-R91.

Provides the :class:`GaussianPlume` class, which models the steady-state air
concentration of radionuclides downwind of a continuous elevated point source.

The Gaussian plume equation with ground reflection (NRPB-R91, Section 2)::

    χ(x, y, z) = Q / (2π · ū · σy · σz)
                 · exp[−y² / (2σy²)]
                 · { exp[−(z − H)² / (2σz²)]  +  exp[−(z + H)² / (2σz²)] }

where:

* χ  – air concentration (Bq/m³)
* Q  – continuous release rate for a single nuclide (Bq/s)
* ū  – mean wind speed at the release height (m/s)
* σy – crosswind dispersion coefficient (m), from Clarke (1979) via NRPB-R91
* σz – vertical dispersion coefficient (m), from Clarke (1979) via NRPB-R91
* H  – effective release height above ground (m)
* x  – downwind distance from source (m)
* y  – crosswind distance from the plume axis (m)
* z  – height above ground (m)

The ground reflection term (image source at −H) assumes perfect reflection
with no deposition at ground level.

.. note::
    This implementation does not currently account for dry or wet deposition,
    radioactive decay during transport, or building-wake effects.

Reference:
    Simmonds, J. R. et al. (1993). *The Methodology for Assessing the
    Radiological Consequences of Routine Releases of Radionuclides to the
    Atmosphere*.  NRPB-R91, National Radiological Protection Board, Chilton, UK.
"""

from __future__ import annotations

import math

from gaussian_plume.dispersion import STABILITY_CATEGORIES, sigma_y, sigma_z


class GaussianPlume:
    """Continuous-release elevated Gaussian plume model (NRPB-R91).

    Models the steady-state air concentration of radionuclides downwind of a
    continuous point source using the Pasquill-Gifford / Clarke (1979)
    dispersion parametrisation from NRPB-R91.  A perfect ground-reflection
    boundary condition is applied (no deposition).

    Args:
        release: Source term — a mapping of nuclide name to continuous release
            rate in Bq/s.  Example: ``{"Co60": 3.7e10, "Cs137": 1.0e9}``.
            All release rates must be non-negative; at least one entry is
            required.
        wind_speed: Mean wind speed at the release height (m/s).  Must be > 0.
        stability_category: Pasquill-Gifford atmospheric stability category,
            one of ``'A'``–``'F'`` (A = very unstable, F = moderately stable).
        release_height: Effective release height above ground level (m).
            Use ``0`` for a ground-level release.  Must be ≥ 0.

    Raises:
        ValueError: If *wind_speed* ≤ 0, *release_height* < 0, *release* is
            empty, any release rate is negative, or *stability_category* is
            not one of ``'A'``–``'F'``.

    Example::

        from gaussian_plume.plume import GaussianPlume

        plume = GaussianPlume(
            release={"Cs137": 1.0e6, "Co60": 3.7e10},
            wind_speed=2.0,
            stability_category="D",
            release_height=50.0,
        )
        chi = plume.centreline_concentration(x=1000.0)
        # {"Cs137": <float> Bq/m³, "Co60": <float> Bq/m³}
    """

    def __init__(
        self,
        release: dict[str, float],
        wind_speed: float,
        stability_category: str,
        release_height: float,
    ) -> None:
        if wind_speed <= 0:
            raise ValueError(f"wind_speed must be positive, got {wind_speed}")
        if release_height < 0:
            raise ValueError(f"release_height must be non-negative, got {release_height}")
        if stability_category not in STABILITY_CATEGORIES:
            raise ValueError(
                f"Unknown stability category {stability_category!r}. "
                f"Valid options are: {', '.join(STABILITY_CATEGORIES)}"
            )
        if not release:
            raise ValueError("release must contain at least one nuclide.")
        for name, rate in release.items():
            if rate < 0:
                raise ValueError(
                    f"Release rate for '{name}' must be non-negative, got {rate}"
                )

        self.release: dict[str, float] = dict(release)
        self.wind_speed: float = wind_speed
        self.stability_category: str = stability_category
        self.release_height: float = release_height

    # ------------------------------------------------------------------
    # Concentration calculations
    # ------------------------------------------------------------------

    def air_concentration(self, x: float, y: float, z: float) -> dict[str, float]:
        """Return air concentration (Bq/m³) at point (x, y, z) for each nuclide.

        Applies the Gaussian plume equation with ground reflection::

            χ = Q / (2π · ū · σy · σz)
                · exp[−y² / (2σy²)]
                · { exp[−(z−H)² / (2σz²)]  +  exp[−(z+H)² / (2σz²)] }

        The concentration for each nuclide is proportional to its release rate
        in :attr:`release`.

        Args:
            x: Downwind distance from the source (m).  Must be positive.
            y: Crosswind distance from the plume axis (m).
            z: Height above ground (m).  Must be non-negative.

        Returns:
            Dictionary mapping each nuclide name to its air concentration
            in Bq/m³.

        Raises:
            ValueError: If *x* ≤ 0 or *z* < 0.
        """
        if x <= 0:
            raise ValueError(f"Downwind distance x must be positive, got {x}")
        if z < 0:
            raise ValueError(f"Height z must be non-negative, got {z}")

        sy = sigma_y(x, self.stability_category)
        sz = sigma_z(x, self.stability_category)
        H = self.release_height
        u = self.wind_speed

        crosswind = math.exp(-y ** 2 / (2.0 * sy ** 2))
        vertical = (
            math.exp(-((z - H) ** 2) / (2.0 * sz ** 2))
            + math.exp(-((z + H) ** 2) / (2.0 * sz ** 2))
        )
        # Common dispersion factor (m⁻³); multiply by Q (Bq/s) to get Bq/m³.
        factor = crosswind * vertical / (2.0 * math.pi * u * sy * sz)

        return {name: Q * factor for name, Q in self.release.items()}

    def centreline_concentration(self, x: float) -> dict[str, float]:
        """Return ground-level centreline air concentration (Bq/m³) at distance x.

        Equivalent to :meth:`air_concentration` with ``y=0, z=0``.  For a
        release at height *H* this simplifies to::

            χ(x, 0, 0) = Q / (π · ū · σy · σz) · exp[−H² / (2σz²)]

        Args:
            x: Downwind distance from the source (m).  Must be positive.

        Returns:
            Dictionary mapping each nuclide name to its centreline ground-level
            air concentration in Bq/m³.

        Raises:
            ValueError: If *x* ≤ 0.
        """
        return self.air_concentration(x, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nuclides = ", ".join(f"{k}: {v:.3g} Bq/s" for k, v in self.release.items())
        return (
            f"GaussianPlume(release={{{nuclides}}}, "
            f"wind_speed={self.wind_speed} m/s, "
            f"stability={self.stability_category!r}, "
            f"H={self.release_height} m)"
        )
