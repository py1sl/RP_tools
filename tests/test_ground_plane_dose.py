"""Tests for utilities.ground_plane_dose semi-infinite plane post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from utilities.ground_plane_dose import (
    SemiInfinitePlaneDoseCalculator,
    semi_infinite_plane_dose_rate_from_deposition,
    semi_infinite_plane_dose_rate_on_grid,
)
from utilities.icrp_data import load_icrp_data
from utilities.nuclide import load_nuclides


class TestSemiInfinitePlaneDoseCalculator:
    def test_positive_for_gamma_emitter(self):
        calc = SemiInfinitePlaneDoseCalculator()
        dose = calc.dose_rate_from_deposition({"Cs137": 1.0})
        assert dose["Cs137"] > 0.0

    def test_zero_for_non_gamma_emitter(self):
        calc = SemiInfinitePlaneDoseCalculator()
        dose = calc.dose_rate_from_deposition({"H3": 1.0})
        assert dose["H3"] == 0.0

    def test_linear_with_deposition(self):
        calc = SemiInfinitePlaneDoseCalculator()
        d1 = calc.dose_rate_from_deposition({"Cs137": 1.0e4})["Cs137"]
        d2 = calc.dose_rate_from_deposition({"Cs137": 2.0e4})["Cs137"]
        assert d2 == pytest.approx(2.0 * d1, rel=1e-12)

    def test_manual_match_for_co60(self):
        calc = SemiInfinitePlaneDoseCalculator(
            geometry="ISO",
            publication="116",
            receptor_height_m=1.0,
            air_attenuation_coeff_m_inv=0.006,
            upward_emission_fraction=0.5,
        )
        a = 1.0e6

        icrp = load_icrp_data()
        table = icrp.get_table("116", "photons")
        energies = table.energies_MeV
        k_iso = table.column("ISO")

        nuclides = load_nuclides()
        transmission = np.exp(-0.006 * 1.0)
        expected = 0.0
        for line in nuclides["Co60"].gamma_lines:
            intensity_fraction = line["intensity_percent"] / 100.0
            k = np.interp(line["energy_MeV"], energies, k_iso, left=k_iso[0], right=k_iso[-1])
            expected += a * intensity_fraction * 0.5 * transmission * (1.0 / 1.0e4) * k * 1.0e-12

        got = calc.dose_rate_from_deposition({"Co60": a})["Co60"]
        assert got == pytest.approx(expected, rel=1e-12)

    def test_higher_detector_height_reduces_dose(self):
        low = SemiInfinitePlaneDoseCalculator(receptor_height_m=1.0)
        high = SemiInfinitePlaneDoseCalculator(receptor_height_m=10.0)
        d_low = low.dose_rate_from_deposition({"Cs137": 1.0e5})["Cs137"]
        d_high = high.dose_rate_from_deposition({"Cs137": 1.0e5})["Cs137"]
        assert d_low > d_high

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            SemiInfinitePlaneDoseCalculator(receptor_height_m=-1.0)
        with pytest.raises(ValueError, match="must be positive"):
            SemiInfinitePlaneDoseCalculator(air_attenuation_coeff_m_inv=0.0)
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            SemiInfinitePlaneDoseCalculator(upward_emission_fraction=1.5)

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            SemiInfinitePlaneDoseCalculator(geometry="BAD")


class TestSemiInfinitePlaneDoseOnGrid:
    def test_shape_preserved(self):
        calc = SemiInfinitePlaneDoseCalculator()
        a_grid = {
            "Cs137": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "H3": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        }
        d_grid = calc.dose_rate_on_grid(a_grid)
        assert d_grid["Cs137"].shape == (2, 2)
        assert d_grid["H3"].shape == (2, 2)

    def test_nan_preserved(self):
        calc = SemiInfinitePlaneDoseCalculator()
        arr = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=float)
        out = calc.dose_rate_on_grid({"Cs137": arr})["Cs137"]
        assert np.isnan(out[0, 1])


class TestConvenienceWrappers:
    def test_point_wrapper(self):
        out = semi_infinite_plane_dose_rate_from_deposition({"Cs137": 1.0})
        assert out["Cs137"] > 0.0

    def test_grid_wrapper(self):
        out = semi_infinite_plane_dose_rate_on_grid({"Cs137": np.array([1.0, 2.0])})
        assert out["Cs137"].shape == (2,)
