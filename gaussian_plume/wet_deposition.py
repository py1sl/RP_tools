"""Wet deposition model based on ADMC and NRPB-R91 methodology.

This module computes the ground-surface concentration (Bq/m²) resulting from
wet deposition of airborne radionuclides released from a continuous point source,
using air concentrations from the :class:`~gaussian_plume.plume.GaussianPlume`
model.

Background
----------
Wet deposition describes the removal of airborne material by precipitation
(rain, snow, etc.).  The NRPB-R91 / ADMC methodology models this as a
*washout coefficient* process:

.. math::

    F_w(x, y) = \\Lambda \\cdot \\chi(x, y, 0)  \\qquad  [\\text{Bq m}^{-2}\\text{ s}^{-1}]

where:

* :math:`\\Lambda` – washout coefficient (m/s); characterises the rate at which
  precipitation scavenges material from the air column to the ground surface.
* :math:`\\chi(x, y, 0)` – ground-level air concentration (Bq/m³) from the
  Gaussian plume model.

Integrating over an exposure time *T* gives the ground-surface concentration:

.. math::

    \\sigma_w(x, y) = \\Lambda \\cdot \\chi(x, y, 0) \\cdot T  \\qquad  [\\text{Bq m}^{-2}]

This implementation uses the **no-depletion** approximation, which is appropriate
when the washout coefficient is small relative to :math:`\\bar{u} \\cdot \\sigma_z`
(the usual case for routine assessments per NRPB-R91, §3).

Typical washout coefficients (NRPB-R91 / ADMC):

* Particles / aerosols: 5×10⁻⁴ m/s (below-cloud scavenging)
* Reactive gases (e.g. elemental iodine): 1×10⁻³ m/s
* Inert / organic vapours: 5×10⁻⁴ m/s

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

#: Default washout coefficient (m/s) used when no per-nuclide value is given.
#: This value (0.5 mm/s) is appropriate for particles/aerosols (NRPB-R91).
DEFAULT_WASHOUT_COEFFICIENT_M_S: float = 5.0e-4


class WetDepositionModel:
    """Wet deposition ground-concentration model (ADMC / NRPB-R91).

    Computes the ground-surface concentration of deposited activity (Bq/m²) by
    combining a :class:`~gaussian_plume.plume.GaussianPlume` air-concentration
    field with a washout coefficient.

    The ground-surface concentration at a point (x, y) after a continuous
    release of duration *T* is::

        σ_w(x, y)  =  Λ  ×  χ(x, y, 0)  ×  T          [Bq/m²]

    where χ(x, y, 0) is the ground-level air concentration from the Gaussian
    plume model, Λ is the washout coefficient (m/s), and T is the integration
    time (s).

    The no-depletion approximation is applied (NRPB-R91, §3): the plume
    concentration is not reduced to account for material removed by wet
    scavenging.  This is conservative and appropriate for routine assessment
    calculations.

    Args:
        plume: A configured :class:`~gaussian_plume.plume.GaussianPlume`
            instance providing the ground-level air-concentration field.
        washout_coefficients: Washout coefficient (m/s).  May be:

            * A single :class:`float` applied uniformly to every nuclide in the
              release (default: 5×10⁻⁴ m/s).
            * A :class:`dict` mapping nuclide names to individual coefficients.
              All nuclides in *plume.release* must be present.  All values must
              be non-negative.

        integration_time_s: Duration of the continuous release over which
            deposition is integrated (s).  Must be positive.  Defaults to
            ``1.0`` s, giving a *deposition rate* (Bq/m²/s) when the result is
            treated as Bq/m² per second of release.

    Raises:
        TypeError: If *plume* is not a :class:`~gaussian_plume.plume.GaussianPlume`
            instance.
        ValueError: If any washout coefficient is negative, if a nuclide in the
            release is missing from *washout_coefficients*, or if
            *integration_time_s* ≤ 0.

    Example::

        from gaussian_plume.plume import GaussianPlume
        from gaussian_plume.wet_deposition import WetDepositionModel

        plume = GaussianPlume(
            release={"Cs137": 1.0e6, "Co60": 3.7e10},
            wind_speed=2.0,
            stability_category="D",
            release_height=50.0,
        )

        # One hour of continuous release, uniform 0.5 mm/s washout coefficient
        model = WetDepositionModel(plume, integration_time_s=3600.0)

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
        washout_coefficients: Union[float, Mapping[str, float]] = DEFAULT_WASHOUT_COEFFICIENT_M_S,
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

        # Resolve washout coefficients to a per-nuclide dict
        if isinstance(washout_coefficients, (int, float)):
            lam = float(washout_coefficients)
            if lam < 0:
                raise ValueError(
                    f"washout_coefficient must be non-negative, got {lam}"
                )
            self.washout_coefficients: dict[str, float] = {
                name: lam for name in plume.release
            }
        else:
            # Mapping supplied — validate completeness and sign
            lam_map = dict(washout_coefficients)
            missing = set(plume.release) - set(lam_map)
            if missing:
                raise ValueError(
                    f"washout_coefficients is missing entries for nuclides: "
                    f"{', '.join(sorted(missing))}"
                )
            for name, lam in lam_map.items():
                if lam < 0:
                    raise ValueError(
                        f"washout_coefficient for '{name}' must be non-negative, got {lam}"
                    )
            self.washout_coefficients = {
                name: float(lam_map[name]) for name in plume.release
            }

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def deposition_rate(self, x: float, y: float) -> dict[str, float]:
        """Return instantaneous wet deposition rate at (x, y) for each nuclide.

        The deposition rate is the flux of activity deposited onto the ground
        surface per unit area per unit time via precipitation::

            F_w(x, y)  =  Λ  ×  χ(x, y, 0)          [Bq/m²/s]

        Args:
            x: Downwind distance from the source (m).  Must be positive.
            y: Crosswind distance from the plume axis (m).

        Returns:
            Dictionary mapping each nuclide name to its wet deposition rate in
            Bq/m²/s.

        Raises:
            ValueError: If *x* ≤ 0.
        """
        chi = self.plume.air_concentration(x, y, 0.0)
        return {name: self.washout_coefficients[name] * chi[name] for name in chi}

    def ground_concentration(self, x: float, y: float) -> dict[str, float]:
        """Return ground-surface concentration at (x, y) for each nuclide.

        Integrates the wet deposition rate over :attr:`integration_time_s` to
        give the accumulated activity per unit area on the ground surface::

            σ_w(x, y)  =  Λ  ×  χ(x, y, 0)  ×  T      [Bq/m²]

        Args:
            x: Downwind distance from the source (m).  Must be positive.
            y: Crosswind distance from the plume axis (m).

        Returns:
            Dictionary mapping each nuclide name to its ground-surface
            concentration in Bq/m².

        Raises:
            ValueError: If *x* ≤ 0.
        """
        fw = self.deposition_rate(x, y)
        return {name: fw[name] * self.integration_time_s for name in fw}

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
            lam = self.washout_coefficients[name]
            sigma = chi_grid[name] * lam * self.integration_time_s
            result[name] = sigma
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lam_str = ", ".join(
            f"{k}: {v:.3g} m/s" for k, v in self.washout_coefficients.items()
        )
        return (
            f"WetDepositionModel(plume={self.plume!r}, "
            f"washout_coefficients={{{lam_str}}}, "
            f"integration_time_s={self.integration_time_s})"
        )
