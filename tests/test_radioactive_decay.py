"""Tests for utilities.radioactive_decay – decay calculation functions."""

from __future__ import annotations

import math

import pytest

from utilities.radioactive_decay import (
    activity_at_time,
    decay_constant,
    decays_in_period,
    time_to_activity,
)


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

# Co-60 half-life in seconds (IAEA: 5.2713 years)
T_HALF_CO60 = 1.66348e8  # s
# One year in seconds (Julian: 365.25 days)
ONE_YEAR_S = 3.15576e7  # s
# One hour in seconds
ONE_HOUR_S = 3600.0


# ---------------------------------------------------------------------------
# decay_constant()
# ---------------------------------------------------------------------------


class TestDecayConstant:
    def test_known_value(self):
        lam = decay_constant(T_HALF_CO60)
        expected = math.log(2) / T_HALF_CO60
        assert abs(lam - expected) / expected < 1e-10

    def test_unit_half_life(self):
        assert abs(decay_constant(1.0) - math.log(2)) < 1e-12

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError, match="positive"):
            decay_constant(0.0)

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="positive"):
            decay_constant(-1.0)


# ---------------------------------------------------------------------------
# activity_at_time()
# ---------------------------------------------------------------------------


class TestActivityAtTime:
    def test_no_decay_at_t0(self):
        A0 = 1.0e9
        assert activity_at_time(A0, T_HALF_CO60, 0.0) == pytest.approx(A0)

    def test_half_activity_at_one_half_life(self):
        A0 = 1.0e9
        A = activity_at_time(A0, T_HALF_CO60, T_HALF_CO60)
        assert A == pytest.approx(A0 / 2.0, rel=1e-9)

    def test_quarter_activity_at_two_half_lives(self):
        A0 = 1.0e9
        A = activity_at_time(A0, T_HALF_CO60, 2 * T_HALF_CO60)
        assert A == pytest.approx(A0 / 4.0, rel=1e-9)

    def test_co60_activity_after_one_year(self):
        """After ~1 year Co-60 should retain ≈87.7% of initial activity."""
        A0 = 3.7e10  # 1 Ci in Bq
        A = activity_at_time(A0, T_HALF_CO60, ONE_YEAR_S)
        fraction = A / A0
        assert 0.870 < fraction < 0.885

    def test_large_t_approaches_zero(self):
        A0 = 1.0e9
        A = activity_at_time(A0, T_HALF_CO60, 1000 * T_HALF_CO60)
        assert A < 1e-9 * A0

    def test_zero_initial_activity(self):
        assert activity_at_time(0.0, T_HALF_CO60, ONE_YEAR_S) == 0.0

    def test_negative_A0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            activity_at_time(-1.0, T_HALF_CO60, ONE_YEAR_S)

    def test_negative_t_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            activity_at_time(1.0e9, T_HALF_CO60, -1.0)

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError):
            activity_at_time(1.0e9, 0.0, ONE_YEAR_S)


# ---------------------------------------------------------------------------
# decays_in_period()
# ---------------------------------------------------------------------------


class TestDecaysInPeriod:
    def test_zero_duration_returns_zero(self):
        assert decays_in_period(1.0e9, T_HALF_CO60, 0.0, 0.0) == 0.0

    def test_returns_positive_count(self):
        N = decays_in_period(3.7e10, T_HALF_CO60, 0.0, ONE_HOUR_S)
        assert N > 0

    def test_short_time_approx_A0_times_t(self):
        """For t << T½ the number of decays ≈ A0 * duration."""
        A0 = 1.0e6
        duration = 1.0  # 1 second, tiny compared to any typical half-life
        N = decays_in_period(A0, T_HALF_CO60, 0.0, duration)
        # Within 0.01% of the constant-rate approximation
        assert N == pytest.approx(A0 * duration, rel=1e-4)

    def test_consistency_with_activity_at_time(self):
        """Integral over [0, T½] is exactly (A0/λ) * (1 − ½) = A0 * T½ / (2·ln2)."""
        A0 = 1.0e9
        N = decays_in_period(A0, T_HALF_CO60, 0.0, T_HALF_CO60)
        # Analytical: ∫₀^T½ A0·exp(−λt)dt = (A0/λ)(1 − exp(−λT½))
        #            = (A0/λ)(1 − 1/2) = A0·T½ / (2·ln2)
        expected = A0 * T_HALF_CO60 / (2 * math.log(2))
        assert N == pytest.approx(expected, rel=1e-9)

    def test_start_time_reduces_count(self):
        """Starting later (activity lower) → fewer decays in same duration."""
        N_early = decays_in_period(1.0e9, T_HALF_CO60, 0.0, ONE_HOUR_S)
        N_late = decays_in_period(1.0e9, T_HALF_CO60, 10 * T_HALF_CO60, ONE_HOUR_S)
        assert N_early > N_late

    def test_additivity(self):
        """Counts over two adjacent intervals should sum to the total count."""
        A0 = 1.0e9
        T1 = ONE_YEAR_S
        T2 = ONE_YEAR_S
        N_total = decays_in_period(A0, T_HALF_CO60, 0.0, T1 + T2)
        N_first = decays_in_period(A0, T_HALF_CO60, 0.0, T1)
        N_second = decays_in_period(A0, T_HALF_CO60, T1, T2)
        assert N_total == pytest.approx(N_first + N_second, rel=1e-9)

    def test_negative_A0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            decays_in_period(-1.0, T_HALF_CO60, 0.0, ONE_HOUR_S)

    def test_negative_t_start_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            decays_in_period(1.0e9, T_HALF_CO60, -1.0, ONE_HOUR_S)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            decays_in_period(1.0e9, T_HALF_CO60, 0.0, -1.0)


# ---------------------------------------------------------------------------
# time_to_activity()
# ---------------------------------------------------------------------------


class TestTimeToActivity:
    def test_one_half_life_to_half_activity(self):
        A0 = 1.0e9
        t = time_to_activity(A0, A0 / 2, T_HALF_CO60)
        assert t == pytest.approx(T_HALF_CO60, rel=1e-9)

    def test_two_half_lives_to_quarter_activity(self):
        A0 = 1.0e9
        t = time_to_activity(A0, A0 / 4, T_HALF_CO60)
        assert t == pytest.approx(2 * T_HALF_CO60, rel=1e-9)

    def test_same_activity_returns_zero(self):
        A0 = 1.0e9
        assert time_to_activity(A0, A0, T_HALF_CO60) == pytest.approx(0.0, abs=1e-10)

    def test_roundtrip_with_activity_at_time(self):
        """activity_at_time(A0, T½, time_to_activity(...)) should recover A_target."""
        A0 = 3.7e10
        A_target = 1.0e9
        t = time_to_activity(A0, A_target, T_HALF_CO60)
        recovered = activity_at_time(A0, T_HALF_CO60, t)
        assert recovered == pytest.approx(A_target, rel=1e-9)

    def test_cs137_years_to_100_bq(self, nuclides):
        """Sanity check using Cs-137 data from the nuclide database."""
        cs137 = nuclides["Cs137"]
        A0 = 1.0e6  # 1 MBq
        A_target = 100.0  # 100 Bq
        t_s = time_to_activity(A0, A_target, cs137.half_life_seconds)
        t_years = t_s / ONE_YEAR_S
        # ln(1e4) / ln(2) * 30.17 ≈ 13.29 * 30.17 ≈ 401 years
        assert 350 < t_years < 450

    def test_zero_A0_raises(self):
        with pytest.raises(ValueError, match="positive"):
            time_to_activity(0.0, 1.0, T_HALF_CO60)

    def test_negative_A0_raises(self):
        with pytest.raises(ValueError, match="positive"):
            time_to_activity(-1.0, 1.0, T_HALF_CO60)

    def test_zero_A_target_raises(self):
        with pytest.raises(ValueError, match="positive"):
            time_to_activity(1.0e9, 0.0, T_HALF_CO60)

    def test_A_target_exceeds_A0_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            time_to_activity(1.0e6, 2.0e6, T_HALF_CO60)
