"""Tests for gaussian_plume – dispersion coefficients and GaussianPlume class.

Reference values are derived analytically from the Clarke (1979) formula::

    σ(x) = a · x · (1 + b · x)^c

and the Gaussian plume equation::

    χ(x, y, z) = Q / (2π · ū · σy · σz)
                 · exp[−y² / (2σy²)]
                 · { exp[−(z−H)² / (2σz²)]  +  exp[−(z+H)² / (2σz²)] }
"""

from __future__ import annotations

import math

import pytest

from gaussian_plume.dispersion import (
    CATEGORY_DESCRIPTIONS,
    STABILITY_CATEGORIES,
    sigma_y,
    sigma_z,
)
from gaussian_plume.plume import GaussianPlume


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

# Representative downwind distances (m)
X_100 = 100.0
X_500 = 500.0
X_1000 = 1000.0
X_5000 = 5000.0

# Standard single-nuclide release (1 Bq/s of Cs-137)
RELEASE_1BQ = {"Cs137": 1.0}

# Co-60 reference release rate
RELEASE_CO60 = {"Co60": 3.7e10}

# Multi-nuclide release
RELEASE_MULTI = {"Co60": 3.7e10, "Cs137": 1.0e9}


# ---------------------------------------------------------------------------
# sigma_y() – Clarke (1979) crosswind dispersion coefficient
# ---------------------------------------------------------------------------


class TestSigmaY:
    def test_returns_positive_for_all_categories(self):
        for cat in STABILITY_CATEGORIES:
            assert sigma_y(X_1000, cat) > 0

    def test_increases_with_distance(self):
        """σy must grow as downwind distance increases."""
        for cat in STABILITY_CATEGORIES:
            assert sigma_y(X_1000, cat) > sigma_y(X_500, cat)

    def test_decreases_with_stability_at_same_distance(self):
        """More unstable (A) → larger σy than more stable (F) at same x."""
        cats = list(STABILITY_CATEGORIES)
        values = [sigma_y(X_1000, c) for c in cats]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], (
                f"σy({cats[i]}) should exceed σy({cats[i+1]})"
            )

    def test_known_value_category_D_x1000(self):
        # σy = 0.08 * 1000 * (1 + 0.0001*1000)^(-0.5) = 80 / sqrt(1.1)
        expected = 0.08 * 1000.0 * (1.0 + 0.0001 * 1000.0) ** (-0.5)
        assert sigma_y(X_1000, "D") == pytest.approx(expected, rel=1e-9)

    def test_known_value_category_A_x1000(self):
        expected = 0.22 * 1000.0 * (1.0 + 0.0001 * 1000.0) ** (-0.5)
        assert sigma_y(X_1000, "A") == pytest.approx(expected, rel=1e-9)

    def test_known_value_category_F_x1000(self):
        expected = 0.04 * 1000.0 * (1.0 + 0.0001 * 1000.0) ** (-0.5)
        assert sigma_y(X_1000, "F") == pytest.approx(expected, rel=1e-9)

    def test_zero_x_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sigma_y(0.0, "D")

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sigma_y(-100.0, "D")

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown stability category"):
            sigma_y(X_1000, "G")

    def test_lowercase_category_raises(self):
        with pytest.raises(ValueError, match="Unknown stability category"):
            sigma_y(X_1000, "d")


# ---------------------------------------------------------------------------
# sigma_z() – Clarke (1979) vertical dispersion coefficient
# ---------------------------------------------------------------------------


class TestSigmaZ:
    def test_returns_positive_for_all_categories(self):
        for cat in STABILITY_CATEGORIES:
            assert sigma_z(X_1000, cat) > 0

    def test_increases_with_distance(self):
        for cat in STABILITY_CATEGORIES:
            assert sigma_z(X_1000, cat) > sigma_z(X_500, cat)

    def test_decreases_with_stability_at_same_distance(self):
        """σz(A) > σz(B) > … > σz(F) at the same downwind distance."""
        cats = list(STABILITY_CATEGORIES)
        values = [sigma_z(X_1000, c) for c in cats]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], (
                f"σz({cats[i]}) should exceed σz({cats[i+1]})"
            )

    def test_known_value_category_A_x1000(self):
        # σz = 0.20 * 1000 * (1 + 0*1000)^0 = 200 m
        assert sigma_z(X_1000, "A") == pytest.approx(200.0, rel=1e-9)

    def test_known_value_category_B_x1000(self):
        # σz = 0.12 * 1000 = 120 m
        assert sigma_z(X_1000, "B") == pytest.approx(120.0, rel=1e-9)

    def test_known_value_category_D_x1000(self):
        # σz = 0.06 * 1000 * (1 + 0.0015*1000)^(-0.5) = 60 / sqrt(2.5)
        expected = 0.06 * 1000.0 * (1.0 + 0.0015 * 1000.0) ** (-0.5)
        assert sigma_z(X_1000, "D") == pytest.approx(expected, rel=1e-9)

    def test_known_value_category_F_x1000(self):
        # σz = 0.016 * 1000 * (1 + 0.0003*1000)^(-1) = 16 / 1.3
        expected = 0.016 * 1000.0 * (1.0 + 0.0003 * 1000.0) ** (-1.0)
        assert sigma_z(X_1000, "F") == pytest.approx(expected, rel=1e-9)

    def test_zero_x_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sigma_z(0.0, "D")

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sigma_z(-1.0, "D")

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown stability category"):
            sigma_z(X_1000, "Z")


# ---------------------------------------------------------------------------
# Stability category metadata
# ---------------------------------------------------------------------------


class TestStabilityMetadata:
    def test_six_categories_defined(self):
        assert len(STABILITY_CATEGORIES) == 6

    def test_all_letters_A_to_F(self):
        assert set(STABILITY_CATEGORIES) == {"A", "B", "C", "D", "E", "F"}

    def test_descriptions_present_for_all_categories(self):
        for cat in STABILITY_CATEGORIES:
            assert cat in CATEGORY_DESCRIPTIONS
            assert isinstance(CATEGORY_DESCRIPTIONS[cat], str)


# ---------------------------------------------------------------------------
# GaussianPlume – construction and validation
# ---------------------------------------------------------------------------


class TestGaussianPlumeConstruction:
    def test_basic_construction(self):
        plume = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=0.0)
        assert plume.stability_category == "D"
        assert plume.wind_speed == 2.0
        assert plume.release_height == 0.0

    def test_multi_nuclide_release_stored(self):
        plume = GaussianPlume(RELEASE_MULTI, wind_speed=3.0, stability_category="B", release_height=10.0)
        assert set(plume.release.keys()) == {"Co60", "Cs137"}
        assert plume.release["Co60"] == 3.7e10

    def test_release_dict_is_copy(self):
        original = {"Cs137": 1.0e6}
        plume = GaussianPlume(original, wind_speed=2.0, stability_category="D", release_height=0.0)
        original["Cs137"] = 0.0
        assert plume.release["Cs137"] == 1.0e6

    def test_zero_release_rate_allowed(self):
        # A nuclide with a zero release rate is valid (not yet released)
        plume = GaussianPlume({"Cs137": 0.0}, wind_speed=2.0, stability_category="D", release_height=0.0)
        assert plume.release["Cs137"] == 0.0

    def test_zero_release_height_allowed(self):
        plume = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=0.0)
        assert plume.release_height == 0.0

    def test_negative_wind_speed_raises(self):
        with pytest.raises(ValueError, match="wind_speed"):
            GaussianPlume(RELEASE_1BQ, wind_speed=-1.0, stability_category="D", release_height=0.0)

    def test_zero_wind_speed_raises(self):
        with pytest.raises(ValueError, match="wind_speed"):
            GaussianPlume(RELEASE_1BQ, wind_speed=0.0, stability_category="D", release_height=0.0)

    def test_negative_release_height_raises(self):
        with pytest.raises(ValueError, match="release_height"):
            GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=-1.0)

    def test_invalid_stability_category_raises(self):
        with pytest.raises(ValueError, match="Unknown stability category"):
            GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="X", release_height=0.0)

    def test_empty_release_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            GaussianPlume({}, wind_speed=2.0, stability_category="D", release_height=0.0)

    def test_negative_release_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            GaussianPlume({"Cs137": -1.0}, wind_speed=2.0, stability_category="D", release_height=0.0)

    def test_repr_contains_key_info(self):
        plume = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=50.0)
        r = repr(plume)
        assert "GaussianPlume" in r
        assert "Cs137" in r
        assert "'D'" in r
        assert "H=50.0" in r


# ---------------------------------------------------------------------------
# GaussianPlume – air_concentration()
# ---------------------------------------------------------------------------


class TestAirConcentration:
    """Tests for air_concentration(x, y, z)."""

    def _plume(self, cat="D", H=0.0, u=2.0, release=None):
        return GaussianPlume(
            release=release or RELEASE_1BQ,
            wind_speed=u,
            stability_category=cat,
            release_height=H,
        )

    def test_returns_dict_with_correct_keys(self):
        plume = GaussianPlume(RELEASE_MULTI, wind_speed=2.0, stability_category="D", release_height=0.0)
        chi = plume.air_concentration(X_1000, 0.0, 0.0)
        assert set(chi.keys()) == {"Co60", "Cs137"}

    def test_concentration_is_positive(self):
        plume = self._plume()
        chi = plume.air_concentration(X_1000, 0.0, 0.0)
        assert chi["Cs137"] > 0

    def test_centreline_equals_air_concentration_at_y0_z0(self):
        plume = self._plume(H=30.0)
        assert plume.centreline_concentration(X_1000) == plume.air_concentration(X_1000, 0.0, 0.0)

    def test_concentration_zero_for_zero_release(self):
        plume = GaussianPlume({"Cs137": 0.0}, wind_speed=2.0, stability_category="D", release_height=0.0)
        chi = plume.air_concentration(X_1000, 0.0, 0.0)
        assert chi["Cs137"] == 0.0

    def test_linearity_in_release_rate(self):
        """Doubling the release rate must double the concentration."""
        plume1 = self._plume(release={"Cs137": 1.0e6})
        plume2 = self._plume(release={"Cs137": 2.0e6})
        chi1 = plume1.centreline_concentration(X_1000)["Cs137"]
        chi2 = plume2.centreline_concentration(X_1000)["Cs137"]
        assert chi2 == pytest.approx(2.0 * chi1, rel=1e-9)

    def test_concentration_decreases_with_downwind_distance_ground_release(self):
        """For H=0, centreline concentration must decrease as x increases."""
        plume = self._plume(H=0.0)
        chi_near = plume.centreline_concentration(X_500)["Cs137"]
        chi_far = plume.centreline_concentration(X_1000)["Cs137"]
        assert chi_near > chi_far

    def test_concentration_decreases_off_centreline(self):
        """Crosswind concentration must be less than centreline."""
        plume = self._plume(H=0.0)
        chi_centre = plume.air_concentration(X_1000, 0.0, 0.0)["Cs137"]
        chi_off = plume.air_concentration(X_1000, 50.0, 0.0)["Cs137"]
        assert chi_centre > chi_off

    def test_concentration_symmetric_about_centreline(self):
        """Concentration at (x, +y, 0) must equal (x, −y, 0)."""
        plume = self._plume(H=20.0)
        chi_pos = plume.air_concentration(X_1000, 100.0, 0.0)["Cs137"]
        chi_neg = plume.air_concentration(X_1000, -100.0, 0.0)["Cs137"]
        assert chi_pos == pytest.approx(chi_neg, rel=1e-12)

    def test_stable_category_higher_concentration_than_unstable_at_ground(self):
        """At H=0 and same x, stable (F) gives higher centreline concentration than unstable (A)."""
        plume_A = self._plume(cat="A", H=0.0)
        plume_F = self._plume(cat="F", H=0.0)
        chi_A = plume_A.centreline_concentration(X_1000)["Cs137"]
        chi_F = plume_F.centreline_concentration(X_1000)["Cs137"]
        assert chi_F > chi_A

    def test_known_value_ground_release_cat_D_x1000(self):
        """Ground-level release, H=0, cat D, x=1km: compare to analytic formula."""
        plume = GaussianPlume({"Cs137": 1.0}, wind_speed=2.0, stability_category="D", release_height=0.0)
        sy = sigma_y(X_1000, "D")
        sz = sigma_z(X_1000, "D")
        expected = 1.0 / (math.pi * 2.0 * sy * sz)
        chi = plume.centreline_concentration(X_1000)["Cs137"]
        assert chi == pytest.approx(expected, rel=1e-9)

    def test_known_value_elevated_release_cat_D_x1000(self):
        """Elevated release H=50m, cat D, x=1km: compare to analytic formula."""
        H = 50.0
        plume = GaussianPlume({"Cs137": 1.0}, wind_speed=2.0, stability_category="D", release_height=H)
        sy = sigma_y(X_1000, "D")
        sz = sigma_z(X_1000, "D")
        expected = (1.0 / (math.pi * 2.0 * sy * sz)) * math.exp(-H ** 2 / (2.0 * sz ** 2))
        chi = plume.centreline_concentration(X_1000)["Cs137"]
        assert chi == pytest.approx(expected, rel=1e-9)

    def test_multi_nuclide_concentrations_independent(self):
        """Each nuclide's concentration scales only with its own release rate."""
        plume = GaussianPlume({"Co60": 2.0, "Cs137": 4.0}, wind_speed=2.0,
                              stability_category="D", release_height=0.0)
        chi = plume.centreline_concentration(X_1000)
        # Co60 rate is half Cs137 rate, so concentration ratio should be 0.5
        assert chi["Co60"] == pytest.approx(0.5 * chi["Cs137"], rel=1e-9)

    def test_negative_x_raises(self):
        plume = self._plume()
        with pytest.raises(ValueError, match="positive"):
            plume.air_concentration(-100.0, 0.0, 0.0)

    def test_zero_x_raises(self):
        plume = self._plume()
        with pytest.raises(ValueError, match="positive"):
            plume.air_concentration(0.0, 0.0, 0.0)

    def test_negative_z_raises(self):
        plume = self._plume()
        with pytest.raises(ValueError, match="non-negative"):
            plume.air_concentration(X_1000, 0.0, -1.0)

    def test_all_stability_categories_run_without_error(self):
        for cat in STABILITY_CATEGORIES:
            plume = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category=cat, release_height=10.0)
            chi = plume.centreline_concentration(X_1000)
            assert chi["Cs137"] > 0


# ---------------------------------------------------------------------------
# GaussianPlume – elevated release integration check
# ---------------------------------------------------------------------------


class TestElevatedRelease:
    def test_elevated_release_lower_ground_concentration_than_ground_release(self):
        """For small x where the plume has not touched down, H>0 gives less ground concentration."""
        plume_ground = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=0.0)
        plume_elev = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=100.0)
        chi_ground = plume_ground.centreline_concentration(X_100)["Cs137"]
        chi_elev = plume_elev.centreline_concentration(X_100)["Cs137"]
        assert chi_ground > chi_elev

    def test_max_concentration_height_equals_release_height_at_x0_approx(self):
        """At the source location the max concentration should be near the release height."""
        H = 50.0
        plume = GaussianPlume(RELEASE_1BQ, wind_speed=2.0, stability_category="D", release_height=H)
        # At x=1m (very close) the plume has barely dispersed; peak should be near z=H
        chi_at_H = plume.air_concentration(1.0, 0.0, H)["Cs137"]
        chi_at_0 = plume.air_concentration(1.0, 0.0, 0.0)["Cs137"]
        assert chi_at_H > chi_at_0
