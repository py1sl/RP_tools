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

When radioactive decay is enabled (via the *half_lives* parameter), the
effective release rate is reduced by the radioactive decay that occurs during
atmospheric transport::

    Q_eff(x) = Q · exp(−λ · x / ū)

where λ = ln(2) / T½ is the decay constant and x / ū is the travel time
from source to receptor.

.. note::
    This implementation does not currently account for dry or wet deposition
    or building-wake effects.

Reference:
    Simmonds, J. R. et al. (1993). *The Methodology for Assessing the
    Radiological Consequences of Routine Releases of Radionuclides to the
    Atmosphere*.  NRPB-R91, National Radiological Protection Board, Chilton, UK.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from gaussian_plume.dispersion import STABILITY_CATEGORIES, sigma_y, sigma_z
import gaussian_plume.grid


class GaussianPlume:
    """Continuous-release elevated Gaussian plume model (NRPB-R91).

    Models the steady-state air concentration of radionuclides downwind of a
    continuous point source using the Pasquill-Gifford / Clarke (1979)
    dispersion parametrisation from NRPB-R91.  A perfect ground-reflection
    boundary condition is applied (no deposition).

    Radioactive decay during atmospheric transport can optionally be accounted
    for by supplying *half_lives*.  The travel time from the source to a
    receptor at downwind distance *x* is estimated as ``x / wind_speed``, and
    the release rate for each nuclide is reduced by the factor
    ``exp(−λ · x / ū)`` before computing the concentration.

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
        half_lives: Optional mapping of nuclide name to half-life in seconds.
            When provided, radioactive decay during atmospheric transport is
            included.  Only nuclides listed here have decay applied; any
            nuclide in *release* that is absent from *half_lives* is treated
            as stable.  All half-lives must be positive.  Any nuclide name in
            *half_lives* that is absent from *release* raises a
            :exc:`ValueError`.  If ``None`` (default), no decay correction is
            applied.

    Raises:
        ValueError: If *wind_speed* ≤ 0, *release_height* < 0, *release* is
            empty, any release rate is negative, *stability_category* is not
            one of ``'A'``–``'F'``, any half-life is not positive, or
            *half_lives* contains a nuclide not present in *release*.

    Example::

        from gaussian_plume.plume import GaussianPlume

        plume = GaussianPlume(
            release={"Cs137": 1.0e6, "Co60": 3.7e10},
            wind_speed=2.0,
            stability_category="D",
            release_height=50.0,
            half_lives={"Cs137": 9.496e8, "Co60": 1.663e8},  # seconds
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
        half_lives: dict[str, float] | None = None,
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
        if half_lives is not None:
            for name, hl in half_lives.items():
                if name not in release:
                    raise ValueError(
                        f"half_lives contains nuclide '{name}' which is not in the release."
                    )
                if hl <= 0:
                    raise ValueError(
                        f"half_life for '{name}' must be positive, got {hl}"
                    )

        self.release: dict[str, float] = dict(release)
        self.wind_speed: float = wind_speed
        self.stability_category: str = stability_category
        self.release_height: float = release_height
        self.half_lives: dict[str, float] | None = dict(half_lives) if half_lives is not None else None

    # ------------------------------------------------------------------
    # Concentration calculations
    # ------------------------------------------------------------------

    def air_concentration(self, x: float, y: float, z: float) -> dict[str, float]:
        """Return air concentration (Bq/m³) at point (x, y, z) for each nuclide.

        Applies the Gaussian plume equation with ground reflection::

            χ = Q_eff / (2π · ū · σy · σz)
                · exp[−y² / (2σy²)]
                · { exp[−(z−H)² / (2σz²)]  +  exp[−(z+H)² / (2σz²)] }

        where ``Q_eff = Q · exp(−λ · x / ū)`` when radioactive decay is
        enabled (see :attr:`half_lives`), or ``Q_eff = Q`` otherwise.

        The concentration for each nuclide is proportional to its effective
        (decay-corrected) release rate.

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

        return {name: Q * factor for name, Q in self._decayed_release(x).items()}

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

    def concentration_on_grid(
        self,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        z_edges: Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return air concentration (Bq/m³) on an xy or xyz grid.

        The grid is defined by bin edges in metres.  Concentrations are
        evaluated at the bin-centre midpoints.  If *z_edges* is omitted the
        grid is treated as a horizontal (xy) slice at ground level (z = 0).

        Only positive downwind bin centres (x > 0) are evaluated; any bin
        whose centre falls at x ≤ 0 is set to ``NaN``.

        Args:
            x_edges: Monotonically increasing sequence of downwind bin-edge
                positions (m).  Must have at least two values.
            y_edges: Monotonically increasing sequence of crosswind bin-edge
                positions (m).  Must have at least two values.
            z_edges: Optional monotonically increasing sequence of vertical
                bin-edge heights (m).  If provided, all values must be ≥ 0.
                When omitted a ground-level (z = 0) xy slice is returned.

        Returns:
            Dictionary mapping each nuclide name to a NumPy array of air
            concentrations in Bq/m³.  The array shape is ``(nx, ny)`` for an
            xy grid or ``(nx, ny, nz)`` for an xyz grid, where *nx*, *ny*,
            and *nz* are the number of bins in each direction.

        Raises:
            ValueError: If any edge sequence has fewer than two elements, or
                if any z edge value is negative.
        """
        x_edges = list(x_edges)
        y_edges = list(y_edges)

        if len(x_edges) < 2:
            raise ValueError("x_edges must have at least two values.")
        if len(y_edges) < 2:
            raise ValueError("y_edges must have at least two values.")

        x_centres = gaussian_plume.grid.bin_centres(x_edges)
        y_centres = gaussian_plume.grid.bin_centres(y_edges)

        if z_edges is None:
            # xy slice at ground level
            z_centres = np.array([0.0])
            squeeze_z = True
        else:
            z_edges = list(z_edges)
            if len(z_edges) < 2:
                raise ValueError("z_edges must have at least two values.")
            if any(z < 0 for z in z_edges):
                raise ValueError("All z_edges values must be non-negative.")
            z_centres = gaussian_plume.grid.bin_centres(z_edges)
            squeeze_z = False

        nx = len(x_centres)
        ny = len(y_centres)
        nz = len(z_centres)

        # Initialise result arrays to NaN (x ≤ 0 bins will remain NaN)
        result: dict[str, np.ndarray] = {
            name: np.full((nx, ny, nz), np.nan) for name in self.release
        }

        for ix, x in enumerate(x_centres):
            if x <= 0:
                continue
            sy = sigma_y(x, self.stability_category)
            sz = sigma_z(x, self.stability_category)
            H = self.release_height
            u = self.wind_speed
            two_sy_sq = 2.0 * sy ** 2
            two_sz_sq = 2.0 * sz ** 2
            # Vectorise over y and z
            y_arr = np.asarray(y_centres)  # shape (ny,)
            z_arr = np.asarray(z_centres)  # shape (nz,)
            crosswind = np.exp(-np.square(y_arr) / two_sy_sq)              # (ny,)
            vertical = (
                np.exp(-np.square(z_arr - H) / two_sz_sq)
                + np.exp(-np.square(z_arr + H) / two_sz_sq)
            )                                                               # (nz,)
            factor = (
                crosswind[:, np.newaxis] * vertical[np.newaxis, :]
                / (2.0 * math.pi * u * sy * sz)
            )                                                               # (ny, nz)
            for name, Q in self._decayed_release(x).items():
                result[name][ix] = Q * factor

        if squeeze_z:
            return {name: arr[:, :, 0] for name, arr in result.items()}
        return result

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------

    def plot_xy_slice(
        self,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        nuclide: str | None = None,
        ax=None,
        log_scale: bool = True,
        **kwargs,
    ):
        """Plot a ground-level xy concentration slice using matplotlib.

        Produces a filled colour map (``pcolormesh``) of air concentration
        (Bq/m³) in the horizontal plane at z = 0 for a single nuclide.

        Args:
            x_edges: Downwind bin-edge positions (m).  At least two values.
            y_edges: Crosswind bin-edge positions (m).  At least two values.
            nuclide: Nuclide to plot.  If *None* and only one nuclide is in the
                release, that nuclide is used automatically; otherwise a
                ``ValueError`` is raised.
            ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new
                figure and axes are created if not provided.
            log_scale: If *True* (default) the colour scale is logarithmic.
            **kwargs: Additional keyword arguments forwarded to
                ``ax.pcolormesh``.

        Returns:
            The :class:`matplotlib.axes.Axes` used for the plot.

        Raises:
            ValueError: If *nuclide* is not specified and more than one nuclide
                is present in the release.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        nuclide = self._resolve_nuclide(nuclide)
        grid = self.concentration_on_grid(x_edges, y_edges)
        data = grid[nuclide]  # shape (nx, ny)

        if ax is None:
            _, ax = plt.subplots()

        x_edges_arr = np.asarray(x_edges)
        y_edges_arr = np.asarray(y_edges)

        plot_kwargs: dict = dict(kwargs)
        if log_scale:
            # Replace any non-positive values with NaN for log scale
            data = np.where(data > 0, data, np.nan)
            if "norm" not in plot_kwargs:
                finite = data[np.isfinite(data)]
                if finite.size > 0:
                    plot_kwargs["norm"] = LogNorm(vmin=finite.min(), vmax=finite.max())

        # pcolormesh expects (ny, nx) — transpose data accordingly
        mesh = ax.pcolormesh(x_edges_arr, y_edges_arr, data.T, **plot_kwargs)
        plt.colorbar(mesh, ax=ax, label=f"{nuclide} concentration (Bq/m³)")
        ax.set_xlabel("Downwind distance x (m)")
        ax.set_ylabel("Crosswind distance y (m)")
        ax.set_title(f"{nuclide} ground-level concentration — stability {self.stability_category!r}")
        return ax

    def plot_centreline(
        self,
        x_edges: Sequence[float],
        nuclide: str | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot ground-level centreline concentration vs downwind distance.

        Produces a line plot of air concentration (Bq/m³) along the plume
        centreline (y = 0, z = 0) for a single nuclide.

        Args:
            x_edges: Downwind bin-edge positions (m).  At least two values.
                Bin centres are used as the x-axis values.
            nuclide: Nuclide to plot.  If *None* and only one nuclide is in the
                release, that nuclide is used automatically; otherwise a
                ``ValueError`` is raised.
            ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new
                figure and axes are created if not provided.
            **kwargs: Additional keyword arguments forwarded to ``ax.plot``.

        Returns:
            The :class:`matplotlib.axes.Axes` used for the plot.

        Raises:
            ValueError: If *nuclide* is not specified and more than one nuclide
                is present in the release.
        """
        import matplotlib.pyplot as plt

        nuclide = self._resolve_nuclide(nuclide)
        x_centres = gaussian_plume.grid.bin_centres(list(x_edges))
        concentrations = []
        for x in x_centres:
            if x > 0:
                concentrations.append(self.centreline_concentration(x)[nuclide])
            else:
                concentrations.append(float("nan"))

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(x_centres, concentrations, **kwargs)
        ax.set_xlabel("Downwind distance x (m)")
        ax.set_ylabel(f"{nuclide} concentration (Bq/m³)")
        ax.set_title(
            f"{nuclide} centreline ground-level concentration — stability {self.stability_category!r}"
        )
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decayed_release(self, x: float) -> dict[str, float]:
        """Return per-nuclide release rates corrected for radioactive decay.

        The travel time from the source to downwind distance *x* is estimated
        as ``x / wind_speed``.  For each nuclide listed in :attr:`half_lives`,
        the release rate is multiplied by ``exp(−λ · x / ū)`` where
        ``λ = ln(2) / T½``.  Nuclides absent from :attr:`half_lives` are
        returned unchanged.

        If :attr:`half_lives` is ``None``, :attr:`release` is returned
        directly without copying.

        Args:
            x: Downwind distance from the source (m).  Must be positive.

        Returns:
            Dictionary mapping each nuclide name to its effective release rate
            in Bq/s after decay correction.
        """
        if self.half_lives is None:
            return self.release
        t = x / self.wind_speed
        result: dict[str, float] = {}
        for name, Q in self.release.items():
            if name in self.half_lives:
                lam = math.log(2) / self.half_lives[name]
                result[name] = Q * math.exp(-lam * t)
            else:
                result[name] = Q
        return result

    def _resolve_nuclide(self, nuclide: str | None) -> str:
        """Return the nuclide name to use, inferring it when only one exists.

        Raises:
            ValueError: If *nuclide* is None and more than one nuclide is in
                the release, or if the named nuclide is not in the release.
        """
        if nuclide is None:
            if len(self.release) == 1:
                return next(iter(self.release))
            raise ValueError(
                "nuclide must be specified when the release contains more than one nuclide. "
                f"Available: {', '.join(self.release)}"
            )
        if nuclide not in self.release:
            raise ValueError(
                f"Nuclide {nuclide!r} not found in release. "
                f"Available: {', '.join(self.release)}"
            )
        return nuclide

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nuclides = ", ".join(f"{k}: {v:.3g} Bq/s" for k, v in self.release.items())
        decay_str = ""
        if self.half_lives is not None:
            hl_parts = ", ".join(f"{k}: {v:.3g} s" for k, v in self.half_lives.items())
            decay_str = f", half_lives={{{hl_parts}}}"
        return (
            f"GaussianPlume(release={{{nuclides}}}, "
            f"wind_speed={self.wind_speed} m/s, "
            f"stability={self.stability_category!r}, "
            f"H={self.release_height} m{decay_str})"
        )
