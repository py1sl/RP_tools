"""Model-agnostic immersion gamma dose-rate post-processing utilities.

This module converts airborne activity concentrations (Bq/m^3) into external
immersive effective dose rates (Sv/s) using:

- nuclide gamma emission lines from ``data/nuclides.json``
- photon dose-per-fluence coefficients from ICRP-74/116 tables

The API is intentionally independent of any concentration model
(Gaussian plume or otherwise).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from utilities.icrp_data import ICRPDataLibrary, load_icrp_data
from utilities.nuclide import Nuclide, load_nuclides


class ImmersionDoseCalculator:
    """Convert concentration fields to immersion effective dose-rate fields.

    The method uses an infinite-cloud, exponential-attenuation approximation:

    1. For each gamma line, source rate density is ``S = C * I_gamma``
       where ``C`` is concentration (Bq/m^3) and ``I_gamma`` is photons/decay.
    2. Fluence rate is approximated as ``Phi = S / mu`` (photons/m^2/s),
       where ``mu`` is an effective attenuation coefficient in air (1/m).
    3. Convert to photons/cm^2/s and multiply by ICRP photon coefficient
       ``k(E, geometry)`` in pSv*cm^2.

    Per-gamma-line contribution:

    ``D_dot = (S / mu / 1e4) * k * 1e-12``  (Sv/s)

    The nuclide dose-rate is the sum over its gamma lines.
    """

    def __init__(
        self,
        geometry: str = "ISO",
        publication: str = "116",
        attenuation_coeff_m_inv: float | Mapping[str, float] = 0.006,
        icrp_library: ICRPDataLibrary | None = None,
        nuclides: Mapping[str, Nuclide] | None = None,
    ) -> None:
        self.geometry = geometry
        self.publication = publication
        self.attenuation_coeff_m_inv = attenuation_coeff_m_inv

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

    def dose_factor_sv_s_per_bq_m3(self, nuclide: str) -> float:
        """Return immersion dose-rate factor for one nuclide.

        Returns the multiplicative factor ``f`` such that:
        ``dose_rate_Sv_s = concentration_Bq_m3 * f``.
        """
        if nuclide not in self._nuclides:
            raise ValueError(
                f"Nuclide {nuclide!r} not found in nuclide database. "
                "Add it to data/nuclides.json or supply a custom nuclide map."
            )

        mu = self._attenuation_for(nuclide)
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
            # concentration (Bq/m^3) multiplied later by caller
            factor += intensity_fraction * (1.0 / mu) * (1.0 / 1.0e4) * k_psv_cm2 * 1.0e-12

        return factor

    def dose_rate_from_concentration(
        self,
        concentration_bq_m3: Mapping[str, float],
    ) -> dict[str, float]:
        """Convert per-nuclide concentration at one point to dose rates (Sv/s)."""
        result: dict[str, float] = {}
        for nuclide, concentration in concentration_bq_m3.items():
            c = float(concentration)
            if c <= 0.0:
                result[nuclide] = 0.0
                continue
            result[nuclide] = c * self.dose_factor_sv_s_per_bq_m3(nuclide)
        return result

    def dose_rate_on_grid(
        self,
        concentration_grid_bq_m3: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Convert per-nuclide concentration grids to dose-rate grids (Sv/s)."""
        result: dict[str, np.ndarray] = {}
        for nuclide, concentration in concentration_grid_bq_m3.items():
            factor = self.dose_factor_sv_s_per_bq_m3(nuclide)
            arr = np.asarray(concentration, dtype=float)
            out = arr * factor
            out = np.where(np.isfinite(arr), out, np.nan)
            result[nuclide] = out
        return result

    def _attenuation_for(self, nuclide: str) -> float:
        if isinstance(self.attenuation_coeff_m_inv, Mapping):
            if nuclide not in self.attenuation_coeff_m_inv:
                raise ValueError(f"Missing attenuation coefficient for nuclide {nuclide!r}.")
            mu = float(self.attenuation_coeff_m_inv[nuclide])
        else:
            mu = float(self.attenuation_coeff_m_inv)

        if mu <= 0.0:
            raise ValueError(
                f"attenuation_coeff_m_inv must be positive, got {mu} for {nuclide!r}"
            )
        return mu


def immersion_dose_rate_from_concentration(
    concentration_bq_m3: Mapping[str, float],
    geometry: str = "ISO",
    publication: str = "116",
    attenuation_coeff_m_inv: float | Mapping[str, float] = 0.006,
) -> dict[str, float]:
    """Convenience wrapper for one-point concentration to immersion dose rate."""
    calculator = ImmersionDoseCalculator(
        geometry=geometry,
        publication=publication,
        attenuation_coeff_m_inv=attenuation_coeff_m_inv,
    )
    return calculator.dose_rate_from_concentration(concentration_bq_m3)


def immersion_dose_rate_on_grid(
    concentration_grid_bq_m3: Mapping[str, np.ndarray],
    geometry: str = "ISO",
    publication: str = "116",
    attenuation_coeff_m_inv: float | Mapping[str, float] = 0.006,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for concentration grids to immersion dose-rate grids."""
    calculator = ImmersionDoseCalculator(
        geometry=geometry,
        publication=publication,
        attenuation_coeff_m_inv=attenuation_coeff_m_inv,
    )
    return calculator.dose_rate_on_grid(concentration_grid_bq_m3)
