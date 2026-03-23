"""Tests for utilities.immersion_dose model-agnostic post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from utilities.icrp_data import load_icrp_data
from utilities.immersion_dose import (
    ImmersionDoseCalculator,
    immersion_dose_rate_from_concentration,
    immersion_dose_rate_on_grid,
)
from utilities.nuclide import load_nuclides


class TestImmersionDoseCalculator:
    def test_returns_positive_for_gamma_emitter(self):
        calc = ImmersionDoseCalculator()
        dose = calc.dose_rate_from_concentration({"Cs137": 1.0})
        assert dose["Cs137"] > 0.0

    def test_returns_zero_for_non_gamma_emitter(self):
        calc = ImmersionDoseCalculator()
        dose = calc.dose_rate_from_concentration({"H3": 1.0})
        assert dose["H3"] == 0.0

    def test_linear_with_concentration(self):
        calc = ImmersionDoseCalculator()
        d1 = calc.dose_rate_from_concentration({"Cs137": 1.0e3})["Cs137"]
        d2 = calc.dose_rate_from_concentration({"Cs137": 2.0e3})["Cs137"]
        assert d2 == pytest.approx(2.0 * d1, rel=1e-12)

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            ImmersionDoseCalculator(geometry="BAD")

    def test_missing_attenuation_mapping_entry_raises(self):
        calc = ImmersionDoseCalculator(attenuation_coeff_m_inv={"Co60": 0.006})
        with pytest.raises(ValueError, match="Missing attenuation coefficient"):
            calc.dose_rate_from_concentration({"Cs137": 1.0})

    def test_non_positive_attenuation_raises(self):
        calc = ImmersionDoseCalculator(attenuation_coeff_m_inv=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            calc.dose_rate_from_concentration({"Cs137": 1.0})

    def test_manual_match_for_co60(self):
        calc = ImmersionDoseCalculator(geometry="ISO", publication="116", attenuation_coeff_m_inv=0.006)
        c = 1.0e5

        icrp = load_icrp_data()
        table = icrp.get_table("116", "photons")
        energies = table.energies_MeV
        k_iso = table.column("ISO")

        nuclides = load_nuclides()
        expected = 0.0
        for line in nuclides["Co60"].gamma_lines:
            intensity_fraction = line["intensity_percent"] / 100.0
            k = np.interp(line["energy_MeV"], energies, k_iso, left=k_iso[0], right=k_iso[-1])
            expected += c * intensity_fraction * (1.0 / 0.006) * (1.0 / 1.0e4) * k * 1.0e-12

        got = calc.dose_rate_from_concentration({"Co60": c})["Co60"]
        assert got == pytest.approx(expected, rel=1e-12)


class TestImmersionDoseOnGrid:
    def test_grid_shape_is_preserved(self):
        calc = ImmersionDoseCalculator()
        c_grid = {
            "Cs137": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "H3": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        }
        d_grid = calc.dose_rate_on_grid(c_grid)
        assert d_grid["Cs137"].shape == (2, 2)
        assert d_grid["H3"].shape == (2, 2)

    def test_grid_linearity(self):
        calc = ImmersionDoseCalculator()
        a = np.array([[1.0, 2.0], [5.0, 10.0]], dtype=float)
        d1 = calc.dose_rate_on_grid({"Cs137": a})["Cs137"]
        d2 = calc.dose_rate_on_grid({"Cs137": 2.0 * a})["Cs137"]
        np.testing.assert_allclose(d2, 2.0 * d1)

    def test_nan_is_preserved(self):
        calc = ImmersionDoseCalculator()
        a = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=float)
        d = calc.dose_rate_on_grid({"Cs137": a})["Cs137"]
        assert np.isnan(d[0, 1])


class TestConvenienceFunctions:
    def test_point_wrapper(self):
        out = immersion_dose_rate_from_concentration({"Cs137": 1.0})
        assert out["Cs137"] > 0.0

    def test_grid_wrapper(self):
        out = immersion_dose_rate_on_grid({"Cs137": np.array([1.0, 2.0])})
        assert out["Cs137"].shape == (2,)
