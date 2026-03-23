"""Semi-infinite ground-plane external gamma dose post-processing utilities.

This module converts deposited activity concentration maps (Bq/m^2) into
external effective dose-rate estimates (Sv/s) for a receptor above ground.

The implementation is model-agnostic: any source that provides deposition
values per nuclide can use this post-processor.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from utilities.icrp_data import ICRPDataLibrary, load_icrp_data
from utilities.nuclide import Nuclide, load_nuclides


class SemiInfinitePlaneDoseCalculator:
    """Estimate external gamma dose rate from deposited activity on ground.

    Approximation used per gamma line:

    1. Upward fluence-rate estimate at receptor height ``h``:

       ``Phi = A_dep * I_gamma * f_up * exp(-mu_air * h)``

       where:
       - ``A_dep`` is deposition (Bq/m^2)
       - ``I_gamma`` is photons/decay
       - ``f_up`` is the upward emission fraction (default 0.5)
       - ``mu_air`` is effective air attenuation coefficient (1/m)

    2. Convert fluence rate to photons/cm^2/s and multiply by ICRP photon
       effective-dose-per-fluence coefficient ``k(E, geometry)`` (pSv*cm^2):

       ``D_dot = (Phi / 1e4) * k * 1e-12``

    Total nuclide dose rate is the sum over gamma lines.

    Notes:
        This is a screening-level semi-infinite-plane approximation. It does
        not include soil self-shielding depth profiles, roughness effects,
        terrain shielding, or buildup corrections.
    """

    def __init__(
        self,
        geometry: str = "ISO",
        publication: str = "116",
        receptor_height_m: float = 1.0,
        air_attenuation_coeff_m_inv: float = 0.006,
        upward_emission_fraction: float = 0.5,
        icrp_library: ICRPDataLibrary | None = None,
        nuclides: Mapping[str, Nuclide] | None = None,
    ) -> None:
        if receptor_height_m < 0.0:
            raise ValueError(
                f"receptor_height_m must be non-negative, got {receptor_height_m}"
            )
        if air_attenuation_coeff_m_inv <= 0.0:
            raise ValueError(
                "air_attenuation_coeff_m_inv must be positive, "
                f"got {air_attenuation_coeff_m_inv}"
            )
        if not (0.0 <= upward_emission_fraction <= 1.0):
            raise ValueError(
                "upward_emission_fraction must be in [0, 1], "
                f"got {upward_emission_fraction}"
            )

        self.geometry = geometry
        self.publication = publication
        self.receptor_height_m = float(receptor_height_m)
        self.air_attenuation_coeff_m_inv = float(air_attenuation_coeff_m_inv)
        self.upward_emission_fraction = float(upward_emission_fraction)

        self._icrp = icrp_library or load_icrp_data()
        self._nuclides = dict(nuclides) if nuclides is not None else load_nuclides()

        table = self._icrp.get_table(publication, "photons")
        if geometry not in table.columns:
            raise ValueError(
                f"Unknown geometry {geometry!r} for ICRP publication {publication!r}. "
                f"Available: {', '.join(table.columns[1:])}"
            )

        self._energies = table.energies_MeV
        self._coeff_psv_cm2 = table.column(geometry)
        self._air_transmission = np.exp(
            -self.air_attenuation_coeff_m_inv * self.receptor_height_m
        )

    def dose_factor_sv_s_per_bq_m2(self, nuclide: str) -> float:
        """Return dose-rate factor so that ``D_dot = A_dep * factor``."""
        if nuclide not in self._nuclides:
            raise ValueError(
                f"Nuclide {nuclide!r} not found in nuclide database. "
                "Add it to data/nuclides.json or supply a custom nuclide map."
            )

        gamma_lines = self._nuclides[nuclide].gamma_lines
        if not gamma_lines:
            return 0.0

        factor = 0.0
        for line in gamma_lines:
            intensity_fraction = float(line.get("intensity_percent", 0.0)) / 100.0
            if intensity_fraction <= 0.0:
                continue

            energy_mev = float(line["energy_MeV"])
            k_psv_cm2 = float(
                np.interp(
                    energy_mev,
                    self._energies,
                    self._coeff_psv_cm2,
                    left=self._coeff_psv_cm2[0],
                    right=self._coeff_psv_cm2[-1],
                )
            )

            factor += (
                intensity_fraction
                * self.upward_emission_fraction
                * self._air_transmission
                * (1.0 / 1.0e4)
                * k_psv_cm2
                * 1.0e-12
            )
        return factor

    def dose_rate_from_deposition(
        self,
        deposition_bq_m2: Mapping[str, float],
    ) -> dict[str, float]:
        """Convert per-nuclide deposition at one point to dose rates (Sv/s)."""
        result: dict[str, float] = {}
        for nuclide, deposition in deposition_bq_m2.items():
            a = float(deposition)
            if a <= 0.0:
                result[nuclide] = 0.0
                continue
            result[nuclide] = a * self.dose_factor_sv_s_per_bq_m2(nuclide)
        return result

    def dose_rate_on_grid(
        self,
        deposition_grid_bq_m2: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Convert per-nuclide deposition grids to dose-rate grids (Sv/s)."""
        result: dict[str, np.ndarray] = {}
        for nuclide, deposition in deposition_grid_bq_m2.items():
            factor = self.dose_factor_sv_s_per_bq_m2(nuclide)
            arr = np.asarray(deposition, dtype=float)
            out = arr * factor
            out = np.where(np.isfinite(arr), out, np.nan)
            result[nuclide] = out
        return result


def semi_infinite_plane_dose_rate_from_deposition(
    deposition_bq_m2: Mapping[str, float],
    geometry: str = "ISO",
    publication: str = "116",
    receptor_height_m: float = 1.0,
    air_attenuation_coeff_m_inv: float = 0.006,
    upward_emission_fraction: float = 0.5,
) -> dict[str, float]:
    """Convenience wrapper for one-point deposition to dose rate."""
    calc = SemiInfinitePlaneDoseCalculator(
        geometry=geometry,
        publication=publication,
        receptor_height_m=receptor_height_m,
        air_attenuation_coeff_m_inv=air_attenuation_coeff_m_inv,
        upward_emission_fraction=upward_emission_fraction,
    )
    return calc.dose_rate_from_deposition(deposition_bq_m2)


def semi_infinite_plane_dose_rate_on_grid(
    deposition_grid_bq_m2: Mapping[str, np.ndarray],
    geometry: str = "ISO",
    publication: str = "116",
    receptor_height_m: float = 1.0,
    air_attenuation_coeff_m_inv: float = 0.006,
    upward_emission_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for deposition grids to dose-rate grids."""
    calc = SemiInfinitePlaneDoseCalculator(
        geometry=geometry,
        publication=publication,
        receptor_height_m=receptor_height_m,
        air_attenuation_coeff_m_inv=air_attenuation_coeff_m_inv,
        upward_emission_fraction=upward_emission_fraction,
    )
    return calc.dose_rate_on_grid(deposition_grid_bq_m2)
