"""Tests for gaussian_plume.wet_deposition.WetDepositionModel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaussian_plume.plume import GaussianPlume
from gaussian_plume.wet_deposition import (
    DEFAULT_WASHOUT_COEFFICIENT_M_S,
    WetDepositionModel,
)
from gaussian_plume.dispersion import sigma_y, sigma_z


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_plume():
    """Single-nuclide ground-level release for straightforward verification."""
    return GaussianPlume(
        release={"Cs137": 1.0e6},
        wind_speed=2.0,
        stability_category="D",
        release_height=0.0,
    )


@pytest.fixture
def elevated_plume():
    """Single-nuclide elevated release (50 m stack)."""
    return GaussianPlume(
        release={"Cs137": 1.0e6},
        wind_speed=2.0,
        stability_category="D",
        release_height=50.0,
    )


@pytest.fixture
def multi_nuclide_plume():
    """Two-nuclide plume for multi-nuclide tests."""
    return GaussianPlume(
        release={"Cs137": 1.0e6, "Co60": 3.7e10},
        wind_speed=3.0,
        stability_category="C",
        release_height=30.0,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestWetDepositionModelConstruction:

    def test_default_construction(self, simple_plume):
        model = WetDepositionModel(simple_plume)
        assert model.integration_time_s == pytest.approx(1.0)
        assert model.washout_coefficients == {"Cs137": DEFAULT_WASHOUT_COEFFICIENT_M_S}

    def test_scalar_washout_coefficient(self, multi_nuclide_plume):
        model = WetDepositionModel(multi_nuclide_plume, washout_coefficients=0.001)
        assert model.washout_coefficients["Cs137"] == pytest.approx(0.001)
        assert model.washout_coefficients["Co60"] == pytest.approx(0.001)

    def test_per_nuclide_washout_coefficients(self, multi_nuclide_plume):
        lam = {"Cs137": 5.0e-4, "Co60": 1.0e-3}
        model = WetDepositionModel(multi_nuclide_plume, washout_coefficients=lam)
        assert model.washout_coefficients["Cs137"] == pytest.approx(5.0e-4)
        assert model.washout_coefficients["Co60"] == pytest.approx(1.0e-3)

    def test_integration_time_stored(self, simple_plume):
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        assert model.integration_time_s == pytest.approx(3600.0)

    def test_invalid_plume_type_raises(self):
        with pytest.raises(TypeError, match="GaussianPlume"):
            WetDepositionModel("not a plume")

    def test_zero_integration_time_raises(self, simple_plume):
        with pytest.raises(ValueError, match="positive"):
            WetDepositionModel(simple_plume, integration_time_s=0.0)

    def test_negative_integration_time_raises(self, simple_plume):
        with pytest.raises(ValueError, match="positive"):
            WetDepositionModel(simple_plume, integration_time_s=-1.0)

    def test_negative_scalar_coefficient_raises(self, simple_plume):
        with pytest.raises(ValueError, match="non-negative"):
            WetDepositionModel(simple_plume, washout_coefficients=-5.0e-4)

    def test_negative_dict_coefficient_raises(self, simple_plume):
        with pytest.raises(ValueError, match="non-negative"):
            WetDepositionModel(simple_plume, washout_coefficients={"Cs137": -5.0e-4})

    def test_missing_nuclide_in_dict_raises(self, multi_nuclide_plume):
        # Only supply one of the two nuclides
        with pytest.raises(ValueError, match="missing entries"):
            WetDepositionModel(multi_nuclide_plume, washout_coefficients={"Cs137": 5.0e-4})

    def test_zero_washout_coefficient_allowed(self, simple_plume):
        """Zero Λ means no wet deposition — should be accepted, result is zero."""
        model = WetDepositionModel(simple_plume, washout_coefficients=0.0)
        assert model.washout_coefficients["Cs137"] == 0.0
        sigma = model.ground_concentration(x=1000.0, y=0.0)
        assert sigma["Cs137"] == 0.0


# ---------------------------------------------------------------------------
# deposition_rate
# ---------------------------------------------------------------------------

class TestDepositionRate:

    def test_units_bq_m2_s(self, simple_plume):
        """Wet deposition rate should be positive for a positive release."""
        model = WetDepositionModel(simple_plume)
        fw = model.deposition_rate(1000.0, 0.0)
        assert fw["Cs137"] > 0.0

    def test_matches_manual_formula(self, simple_plume):
        """F_w = Λ * chi(x, y=0, z=0)."""
        lam = 1.0e-3
        model = WetDepositionModel(simple_plume, washout_coefficients=lam)
        x = 500.0
        chi = simple_plume.air_concentration(x, 0.0, 0.0)
        expected = lam * chi["Cs137"]
        got = model.deposition_rate(x, 0.0)["Cs137"]
        assert got == pytest.approx(expected, rel=1e-12)

    def test_decreases_with_downwind_distance(self, simple_plume):
        """Wet deposition rate generally decreases far downwind for a ground-level release."""
        model = WetDepositionModel(simple_plume)
        fw_near = model.deposition_rate(200.0, 0.0)["Cs137"]
        fw_far = model.deposition_rate(10000.0, 0.0)["Cs137"]
        assert fw_near > fw_far

    def test_crosswind_symmetry(self, simple_plume):
        """F_w is symmetric about the plume centreline (y ↔ -y)."""
        model = WetDepositionModel(simple_plume)
        fw_pos = model.deposition_rate(1000.0, +200.0)["Cs137"]
        fw_neg = model.deposition_rate(1000.0, -200.0)["Cs137"]
        assert fw_pos == pytest.approx(fw_neg, rel=1e-12)

    def test_x_must_be_positive(self, simple_plume):
        model = WetDepositionModel(simple_plume)
        with pytest.raises(ValueError):
            model.deposition_rate(0.0, 0.0)

    def test_proportional_to_release_rate(self):
        """Doubling the source term doubles the wet deposition rate."""
        plume1 = GaussianPlume(
            release={"Cs137": 1.0e6}, wind_speed=2.0,
            stability_category="D", release_height=0.0
        )
        plume2 = GaussianPlume(
            release={"Cs137": 2.0e6}, wind_speed=2.0,
            stability_category="D", release_height=0.0
        )
        m1 = WetDepositionModel(plume1)
        m2 = WetDepositionModel(plume2)
        fw1 = m1.deposition_rate(1000.0, 0.0)["Cs137"]
        fw2 = m2.deposition_rate(1000.0, 0.0)["Cs137"]
        assert fw2 == pytest.approx(2.0 * fw1, rel=1e-12)

    def test_proportional_to_washout_coefficient(self, simple_plume):
        """Doubling Λ doubles the wet deposition rate."""
        m1 = WetDepositionModel(simple_plume, washout_coefficients=5.0e-4)
        m2 = WetDepositionModel(simple_plume, washout_coefficients=1.0e-3)
        fw1 = m1.deposition_rate(1000.0, 0.0)["Cs137"]
        fw2 = m2.deposition_rate(1000.0, 0.0)["Cs137"]
        assert fw2 == pytest.approx(2.0 * fw1, rel=1e-12)


# ---------------------------------------------------------------------------
# ground_concentration
# ---------------------------------------------------------------------------

class TestGroundConcentration:

    def test_result_in_bq_m2(self, simple_plume):
        """Ground concentration should be positive for a positive release."""
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        sigma = model.ground_concentration(1000.0, 0.0)
        assert sigma["Cs137"] > 0.0

    def test_equals_deposition_rate_times_time(self, simple_plume):
        """σ_w(x,y) = F_w(x,y) × T."""
        T = 7200.0
        model = WetDepositionModel(simple_plume, integration_time_s=T)
        fw = model.deposition_rate(1000.0, 0.0)["Cs137"]
        sigma = model.ground_concentration(1000.0, 0.0)["Cs137"]
        assert sigma == pytest.approx(fw * T, rel=1e-12)

    def test_unit_time_equals_deposition_rate(self, simple_plume):
        """With T=1 s, ground_concentration equals deposition_rate."""
        model = WetDepositionModel(simple_plume, integration_time_s=1.0)
        fw = model.deposition_rate(1000.0, 50.0)["Cs137"]
        sigma = model.ground_concentration(1000.0, 50.0)["Cs137"]
        assert sigma == pytest.approx(fw, rel=1e-12)

    def test_proportional_to_integration_time(self, simple_plume):
        """Doubling integration time doubles the ground concentration."""
        m1 = WetDepositionModel(simple_plume, integration_time_s=1000.0)
        m2 = WetDepositionModel(simple_plume, integration_time_s=2000.0)
        s1 = m1.ground_concentration(500.0, 0.0)["Cs137"]
        s2 = m2.ground_concentration(500.0, 0.0)["Cs137"]
        assert s2 == pytest.approx(2.0 * s1, rel=1e-12)

    def test_manual_formula_ground_level_release(self):
        """Verify σ_w = Λ * Q/(π*u*σy*σz) * T for a ground-level release (H=0).

        For H=0 the Gaussian plume ground-level centreline simplifies to::

            χ(x, 0, 0) = Q / (π * u * σy * σz)

        so::

            σ_w(x, 0) = Λ * Q / (π * u * σy * σz) * T
        """
        Q = 1.0e6   # Bq/s
        u = 2.0     # m/s
        cat = "D"
        lam = 5.0e-4  # m/s
        T = 3600.0  # s
        x = 1000.0  # m

        plume = GaussianPlume(
            release={"Cs137": Q},
            wind_speed=u,
            stability_category=cat,
            release_height=0.0,
        )
        model = WetDepositionModel(plume, washout_coefficients=lam, integration_time_s=T)

        sy = sigma_y(x, cat)
        sz = sigma_z(x, cat)
        expected = lam * Q / (math.pi * u * sy * sz) * T

        got = model.ground_concentration(x, 0.0)["Cs137"]
        assert got == pytest.approx(expected, rel=1e-9)

    def test_multi_nuclide_per_nuclide_coefficients(self, multi_nuclide_plume):
        """Per-nuclide coefficients produce independent results."""
        lam = {"Cs137": 5.0e-4, "Co60": 1.0e-3}
        T = 60.0
        model = WetDepositionModel(multi_nuclide_plume, washout_coefficients=lam, integration_time_s=T)
        sigma = model.ground_concentration(2000.0, 0.0)
        assert "Cs137" in sigma
        assert "Co60" in sigma
        # Co60 has 2× higher Λ → its σ should be 2× what it would be at 5e-4
        model_ref = WetDepositionModel(multi_nuclide_plume, washout_coefficients=5.0e-4, integration_time_s=T)
        sigma_ref = model_ref.ground_concentration(2000.0, 0.0)
        assert sigma["Co60"] == pytest.approx(2.0 * sigma_ref["Co60"], rel=1e-12)


# ---------------------------------------------------------------------------
# centreline_ground_concentration
# ---------------------------------------------------------------------------

class TestCentrelineGroundConcentration:

    def test_matches_ground_concentration_at_y0(self, simple_plume):
        """centreline result matches ground_concentration with y=0."""
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        x = 1500.0
        assert model.centreline_ground_concentration(x)["Cs137"] == pytest.approx(
            model.ground_concentration(x, 0.0)["Cs137"], rel=1e-12
        )

    def test_positive_for_positive_release(self, simple_plume):
        model = WetDepositionModel(simple_plume, integration_time_s=100.0)
        assert model.centreline_ground_concentration(500.0)["Cs137"] > 0.0

    def test_invalid_x_raises(self, simple_plume):
        model = WetDepositionModel(simple_plume)
        with pytest.raises(ValueError):
            model.centreline_ground_concentration(-1.0)


# ---------------------------------------------------------------------------
# ground_concentration_on_grid
# ---------------------------------------------------------------------------

class TestGroundConcentrationOnGrid:

    def test_shape(self, simple_plume):
        """Output arrays have shape (nx, ny)."""
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        x_edges = [0, 500, 1000, 2000, 5000]   # 4 bins
        y_edges = [-500, -250, 0, 250, 500]      # 4 bins
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        assert grid["Cs137"].shape == (4, 4)

    def test_nan_for_non_positive_x(self, simple_plume):
        """Bins with x ≤ 0 centre must be NaN."""
        model = WetDepositionModel(simple_plume, integration_time_s=1.0)
        x_edges = [-1000, 0, 500, 1000]   # first bin centre = -500
        y_edges = [-500, 0, 500]
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        assert np.all(np.isnan(grid["Cs137"][0, :]))

    def test_positive_downwind_values(self, simple_plume):
        """All positive-x bins should have non-negative, finite values."""
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        x_edges = [100, 500, 1000, 5000]
        y_edges = [-500, 0, 500]
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        assert np.all(np.isfinite(grid["Cs137"]))
        assert np.all(grid["Cs137"] >= 0.0)

    def test_matches_pointwise_calculation(self, simple_plume):
        """Grid value at a bin centre matches the point calculation."""
        T = 3600.0
        model = WetDepositionModel(simple_plume, integration_time_s=T)
        # x_edges=[500, 1000] → one bin, centre at 750 m
        # y_edges=[-250, 250] → one bin, centre at 0 m
        x_edges = [500, 1000]
        y_edges = [-250, 250]
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        point = model.ground_concentration(750.0, 0.0)["Cs137"]
        assert grid["Cs137"][0, 0] == pytest.approx(point, rel=1e-12)

    def test_multi_nuclide_grid(self, multi_nuclide_plume):
        """Both nuclides present with correct shapes."""
        model = WetDepositionModel(
            multi_nuclide_plume,
            washout_coefficients={"Cs137": 5.0e-4, "Co60": 1.0e-3},
            integration_time_s=600.0,
        )
        x_edges = [0, 1000, 5000]
        y_edges = [-1000, 0, 1000]
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        assert set(grid.keys()) == {"Cs137", "Co60"}
        assert grid["Cs137"].shape == (2, 2)
        assert grid["Co60"].shape == (2, 2)

    def test_crosswind_symmetry_on_grid(self, simple_plume):
        """Grid values are symmetric about y=0."""
        model = WetDepositionModel(simple_plume, integration_time_s=3600.0)
        x_edges = [500, 1000, 5000]
        y_edges = [-1000, -500, 0, 500, 1000]
        grid = model.ground_concentration_on_grid(x_edges, y_edges)
        # y bin 0 (centre -750) and y bin 3 (centre 750) should be equal
        arr = grid["Cs137"]
        assert arr[:, 0] == pytest.approx(arr[:, 3], rel=1e-10)
        # y bin 1 (centre -250) and y bin 2 (centre 250) should be equal
        assert arr[:, 1] == pytest.approx(arr[:, 2], rel=1e-10)

    def test_short_x_edges_raises(self, simple_plume):
        model = WetDepositionModel(simple_plume)
        with pytest.raises(ValueError, match="x_edges"):
            model.ground_concentration_on_grid([500], [-500, 0, 500])

    def test_short_y_edges_raises(self, simple_plume):
        model = WetDepositionModel(simple_plume)
        with pytest.raises(ValueError, match="y_edges"):
            model.ground_concentration_on_grid([0, 500, 1000], [0])

    def test_grid_proportional_to_integration_time(self, simple_plume):
        """Doubling integration time doubles all grid values."""
        m1 = WetDepositionModel(simple_plume, integration_time_s=1000.0)
        m2 = WetDepositionModel(simple_plume, integration_time_s=2000.0)
        x_edges = [500, 1000, 3000]
        y_edges = [-500, 0, 500]
        g1 = m1.ground_concentration_on_grid(x_edges, y_edges)["Cs137"]
        g2 = m2.ground_concentration_on_grid(x_edges, y_edges)["Cs137"]
        assert g2 == pytest.approx(2.0 * g1, rel=1e-12)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_key_info(self, simple_plume):
        model = WetDepositionModel(simple_plume, washout_coefficients=1.0e-3, integration_time_s=60.0)
        r = repr(model)
        assert "WetDepositionModel" in r
        assert "Cs137" in r
        assert "60.0" in r
