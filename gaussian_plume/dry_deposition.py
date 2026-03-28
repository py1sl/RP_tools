"""Dry deposition model based on ADMC and NRPB-R91 methodology.

This module computes the ground-surface concentration (Bq/m²) resulting from
dry deposition of airborne radionuclides released from a continuous point source,
using air concentrations from the :class:`~gaussian_plume.plume.GaussianPlume`
model.

Background
----------
Dry deposition describes the gravitational settling and surface impaction of
airborne material onto the ground surface.  The NRPB-R91 / ADMC methodology
models this as a *deposition velocity* process:

.. math::

    F_d(x, y) = V_d \\cdot \\chi(x, y, 0)  \\qquad  [\\text{Bq m}^{-2}\\text{ s}^{-1}]

where:

* :math:`V_d` – dry deposition velocity (m/s); characterises the rate at which
  material is transferred from the air column to the ground surface.
* :math:`\\chi(x, y, 0)` – ground-level air concentration (Bq/m³) from the
  Gaussian plume model.

Integrating over an exposure time *T* gives the ground-surface concentration:

.. math::

    \\sigma(x, y) = V_d \\cdot \\chi(x, y, 0) \\cdot T  \\qquad  [\\text{Bq m}^{-2}]

This implementation uses the **no-depletion** approximation, which is appropriate
when the deposition velocity is small relative to :math:`\\bar{u} \\cdot \\sigma_z`
(the usual case for routine assessments per NRPB-R91, §3).

Typical deposition velocities (NRPB-R91 / ADMC):

* Particles / aerosols: 0.001 m/s (1 mm/s)
* Reactive gases (e.g. elemental iodine): 0.01 m/s (10 mm/s)
* Inert / organic vapours: 0.001 m/s (1 mm/s)

References:
    Simmonds, J. R. et al. (1993). *The Methodology for Assessing the
    Radiological Consequences of Routine Releases of Radionuclides to the
    Atmosphere*.  NRPB-R91, National Radiological Protection Board, Chilton, UK.

    Clarke, R. H. (1979). *A Model for Short and Medium Range Dispersion of
    Radionuclides Released to the Atmosphere* (the 1st Report of a Working
    Group on Atmospheric Dispersion). NRPB-R91 Appendix, NRPB, Chilton, UK.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

from gaussian_plume.plume import GaussianPlume

#: Default dry deposition velocity (m/s) used when no per-nuclide value is given.
#: This value (1 mm/s) is appropriate for particles/aerosols (NRPB-R91).
DEFAULT_DEPOSITION_VELOCITY_M_S: float = 1.0e-3


class DryDepositionModel:
    """Dry deposition ground-concentration model (ADMC / NRPB-R91).

    Computes the ground-surface concentration of deposited activity (Bq/m²) by
    combining a :class:`~gaussian_plume.plume.GaussianPlume` air-concentration
    field with a dry deposition velocity.

    The ground-surface concentration at a point (x, y) after a continuous
    release of duration *T* is::

        σ(x, y)  =  V_d  ×  χ(x, y, 0)  ×  T          [Bq/m²]

    where χ(x, y, 0) is the ground-level air concentration from the Gaussian
    plume model, V_d is the dry deposition velocity (m/s), and T is the
    integration time (s).

    The no-depletion approximation is applied (NRPB-R91, §3): the plume
    concentration is not reduced to account for material removed by deposition.
    This is conservative and appropriate for routine assessment calculations.

    Args:
        plume: A configured :class:`~gaussian_plume.plume.GaussianPlume`
            instance providing the ground-level air-concentration field.
        deposition_velocities: Dry deposition velocity (m/s).  May be:

            * A single :class:`float` applied uniformly to every nuclide in the
              release (default: 0.001 m/s).
            * A :class:`dict` mapping nuclide names to individual velocities.
              All nuclides in *plume.release* must be present.  All values must
              be non-negative.

        integration_time_s: Duration of the continuous release over which
            deposition is integrated (s).  Must be positive.  Defaults to
            ``1.0`` s, giving a *deposition rate* (Bq/m²/s) when the result is
            treated as Bq/m² per second of release.

    Raises:
        TypeError: If *plume* is not a :class:`~gaussian_plume.plume.GaussianPlume`
            instance.
        ValueError: If any deposition velocity is negative, if a nuclide in the
            release is missing from *deposition_velocities*, or if
            *integration_time_s* ≤ 0.

    Example::

        from gaussian_plume.plume import GaussianPlume
        from gaussian_plume.dry_deposition import DryDepositionModel

        plume = GaussianPlume(
            release={"Cs137": 1.0e6, "Co60": 3.7e10},
            wind_speed=2.0,
            stability_category="D",
            release_height=50.0,
        )

        # One hour of continuous release, uniform 1 mm/s deposition velocity
        model = DryDepositionModel(plume, integration_time_s=3600.0)

        # Ground concentration on the centreline at 1 km
        sigma = model.ground_concentration(x=1000.0, y=0.0)
        # {"Cs137": <float> Bq/m², "Co60": <float> Bq/m²}

        # Or on a spatial grid
        grid = model.ground_concentration_on_grid(
            x_edges=[0, 500, 1000, 5000, 10000],
            y_edges=[-1000, -500, 0, 500, 1000],
        )
        # {"Cs137": (4, 4) array Bq/m², "Co60": (4, 4) array Bq/m²}
    """

    def __init__(
        self,
        plume: GaussianPlume,
        deposition_velocities: Union[float, Mapping[str, float]] = DEFAULT_DEPOSITION_VELOCITY_M_S,
        integration_time_s: float = 1.0,
    ) -> None:
        if not isinstance(plume, GaussianPlume):
            raise TypeError(
                f"plume must be a GaussianPlume instance, got {type(plume).__name__!r}"
            )
        if integration_time_s <= 0:
            raise ValueError(
                f"integration_time_s must be positive, got {integration_time_s}"
            )

        self.plume: GaussianPlume = plume
        self.integration_time_s: float = float(integration_time_s)

        # Resolve deposition velocities to a per-nuclide dict
        if isinstance(deposition_velocities, (int, float)):
            vd = float(deposition_velocities)
            if vd < 0:
                raise ValueError(
                    f"deposition_velocity must be non-negative, got {vd}"
                )
            self.deposition_velocities: dict[str, float] = {
                name: vd for name in plume.release
            }
        else:
            # Mapping supplied — validate completeness and sign
            vd_map = dict(deposition_velocities)
            missing = set(plume.release) - set(vd_map)
            if missing:
                raise ValueError(
                    f"deposition_velocities is missing entries for nuclides: "
                    f"{', '.join(sorted(missing))}"
                )
            for name, vd in vd_map.items():
                if vd < 0:
                    raise ValueError(
                        f"deposition_velocity for '{name}' must be non-negative, got {vd}"
                    )
            self.deposition_velocities = {
                name: float(vd_map[name]) for name in plume.release
            }

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def deposition_rate(self, x: float, y: float) -> dict[str, float]:
        """Return instantaneous dry deposition rate at (x, y) for each nuclide.

        The deposition rate is the flux of activity depositing onto the ground
        surface per unit area per unit time::

            F_d(x, y)  =  V_d  ×  χ(x, y, 0)          [Bq/m²/s]

        Args:
            x: Downwind distance from the source (m).  Must be positive.
            y: Crosswind distance from the plume axis (m).

        Returns:
            Dictionary mapping each nuclide name to its deposition rate in
            Bq/m²/s.

        Raises:
            ValueError: If *x* ≤ 0.
        """
        chi = self.plume.air_concentration(x, y, 0.0)
        return {name: self.deposition_velocities[name] * chi[name] for name in chi}

    def ground_concentration(self, x: float, y: float) -> dict[str, float]:
        """Return ground-surface concentration at (x, y) for each nuclide.

        Integrates the deposition rate over :attr:`integration_time_s` to give
        the accumulated activity per unit area on the ground surface::

            σ(x, y)  =  V_d  ×  χ(x, y, 0)  ×  T      [Bq/m²]

        Args:
            x: Downwind distance from the source (m).  Must be positive.
            y: Crosswind distance from the plume axis (m).

        Returns:
            Dictionary mapping each nuclide name to its ground-surface
            concentration in Bq/m².

        Raises:
            ValueError: If *x* ≤ 0.
        """
        fd = self.deposition_rate(x, y)
        return {name: fd[name] * self.integration_time_s for name in fd}

    def centreline_ground_concentration(self, x: float) -> dict[str, float]:
        """Return ground-surface concentration on the centreline at distance x.

        Equivalent to :meth:`ground_concentration` with ``y = 0``.

        Args:
            x: Downwind distance from the source (m).  Must be positive.

        Returns:
            Dictionary mapping each nuclide name to its centreline
            ground-surface concentration in Bq/m².

        Raises:
            ValueError: If *x* ≤ 0.
        """
        return self.ground_concentration(x, 0.0)

    def ground_concentration_on_grid(
        self,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """Return ground-surface concentration (Bq/m²) on an xy grid.

        Evaluates :meth:`ground_concentration` at the bin-centre midpoints of
        an xy grid.  Bins whose centre falls at x ≤ 0 are set to ``NaN``.

        Args:
            x_edges: Monotonically increasing sequence of downwind bin-edge
                positions (m).  Must have at least two values.
            y_edges: Monotonically increasing sequence of crosswind bin-edge
                positions (m).  Must have at least two values.

        Returns:
            Dictionary mapping each nuclide name to a NumPy array of
            ground-surface concentrations in Bq/m².  Array shape is
            ``(nx, ny)``, where *nx* and *ny* are the number of bins in each
            direction.  Bins at x ≤ 0 are ``NaN``.

        Raises:
            ValueError: If any edge sequence has fewer than two elements.
        """
        x_edges = list(x_edges)
        y_edges = list(y_edges)

        if len(x_edges) < 2:
            raise ValueError("x_edges must have at least two values.")
        if len(y_edges) < 2:
            raise ValueError("y_edges must have at least two values.")

        # Delegate air-concentration grid evaluation to the plume model
        chi_grid = self.plume.concentration_on_grid(x_edges, y_edges)

        result: dict[str, np.ndarray] = {}
        for name in self.plume.release:
            vd = self.deposition_velocities[name]
            sigma = chi_grid[name] * vd * self.integration_time_s
            result[name] = sigma
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        vd_str = ", ".join(
            f"{k}: {v:.3g} m/s" for k, v in self.deposition_velocities.items()
        )
        return (
            f"DryDepositionModel(plume={self.plume!r}, "
            f"deposition_velocities={{{vd_str}}}, "
            f"integration_time_s={self.integration_time_s})"
        )
