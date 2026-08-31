"""Tests for utilities.radioactive_decay – decay calculation functions."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from utilities.nuclide import Nuclide
from utilities.radioactive_decay import (
    activity_at_time,
    alpha_decay,
    beta_minus_decay,
    beta_plus_decay,
    decay_chain,
    decay_constant,
    decays_in_period,
    electron_capture,
    plot_decay_chain,
    time_to_activity,
)


# ---------------------------------------------------------------------------
# Helpers – minimal in-memory Nuclide database for decay-mode tests
# ---------------------------------------------------------------------------

def _make_nuclide(name: str, Z: int, A: int, stable: bool, modes: list | None = None) -> Nuclide:
    """Build a minimal :class:`Nuclide` suitable for unit-testing decay helpers."""
    data: dict = {
        "name": name,
        "long_name": name,
        "symbol": name[:2].strip("0123456789") or "X",
        "A": A,
        "Z": Z,
        "stable": stable,
        "decay_modes": modes or [],
    }
    if not stable:
        # Provide a nominal half-life so the constructor is happy.
        data["half_life_seconds"] = 1.0
        data["half_life_years"] = 3.17e-8
    return Nuclide(data)


# A tiny three-nuclide chain:  A(alpha) → B(beta-) → C(stable)
#   A: Z=84, A=212  →  B: Z=82, A=208  →  C: Z=83, A=208 (stable)
_A = _make_nuclide("A212", Z=84, A=212, stable=False,
                   modes=[{"mode": "alpha", "branching_fraction": 1.0}])
_B = _make_nuclide("B208", Z=82, A=208, stable=False,
                   modes=[{"mode": "beta-", "branching_fraction": 1.0}])
_C = _make_nuclide("C208", Z=83, A=208, stable=True)

_MINI_DB: dict[str, Nuclide] = {"A212": _A, "B208": _B, "C208": _C}

# Nuclide with a named daughter that IS in the mini-db.
_NAMED_PARENT = _make_nuclide(
    "NP60", Z=27, A=60, stable=False,
    modes=[{"mode": "beta-", "branching_fraction": 1.0, "daughter": "B208"}],
)
_NAMED_DB: dict[str, Nuclide] = {"NP60": _NAMED_PARENT, "B208": _B, "C208": _C}

# Nuclide with a named daughter that is NOT in any db (simulates stable end-product).
_NAMED_PARENT_MISSING = _make_nuclide(
    "NM60", Z=27, A=60, stable=False,
    modes=[{"mode": "beta-", "branching_fraction": 1.0, "daughter": "Ni60"}],
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


# ---------------------------------------------------------------------------
# alpha_decay()
# ---------------------------------------------------------------------------


class TestAlphaDecay:
    def test_z_decreases_by_2(self):
        daughter = alpha_decay(_A, _MINI_DB)
        assert daughter.Z == _A.Z - 2

    def test_a_decreases_by_4(self):
        daughter = alpha_decay(_A, _MINI_DB)
        assert daughter.A == _A.A - 4

    def test_returns_correct_nuclide(self):
        daughter = alpha_decay(_A, _MINI_DB)
        assert daughter is _B

    def test_stable_raises(self):
        with pytest.raises(ValueError, match="stable"):
            alpha_decay(_C, _MINI_DB)

    def test_daughter_not_in_db_raises(self):
        with pytest.raises(KeyError):
            alpha_decay(_B, {})  # empty db


# ---------------------------------------------------------------------------
# beta_minus_decay()
# ---------------------------------------------------------------------------


class TestBetaMinusDecay:
    def test_z_increases_by_1(self):
        daughter = beta_minus_decay(_B, _MINI_DB)
        assert daughter.Z == _B.Z + 1

    def test_a_unchanged(self):
        daughter = beta_minus_decay(_B, _MINI_DB)
        assert daughter.A == _B.A

    def test_returns_correct_nuclide(self):
        daughter = beta_minus_decay(_B, _MINI_DB)
        assert daughter is _C

    def test_stable_raises(self):
        with pytest.raises(ValueError, match="stable"):
            beta_minus_decay(_C, _MINI_DB)

    def test_daughter_not_in_db_raises(self):
        with pytest.raises(KeyError):
            beta_minus_decay(_A, {})  # Z+1,A not in empty db


# ---------------------------------------------------------------------------
# beta_plus_decay()
# ---------------------------------------------------------------------------


class TestBetaPlusDecay:
    def test_z_decreases_by_1(self):
        # Use _C (Z=83, A=208) as parent (mark it unstable via a fresh nuclide).
        parent = _make_nuclide("Pplus", Z=83, A=208, stable=False,
                               modes=[{"mode": "beta+", "branching_fraction": 1.0}])
        db = {"Pplus": parent, "B208": _B}
        daughter = beta_plus_decay(parent, db)
        assert daughter.Z == parent.Z - 1

    def test_a_unchanged(self):
        parent = _make_nuclide("Pplus", Z=83, A=208, stable=False,
                               modes=[{"mode": "beta+", "branching_fraction": 1.0}])
        db = {"Pplus": parent, "B208": _B}
        daughter = beta_plus_decay(parent, db)
        assert daughter.A == parent.A

    def test_stable_raises(self):
        with pytest.raises(ValueError, match="stable"):
            beta_plus_decay(_C, _MINI_DB)


# ---------------------------------------------------------------------------
# electron_capture()
# ---------------------------------------------------------------------------


class TestElectronCapture:
    def test_z_decreases_by_1(self):
        parent = _make_nuclide("Pec", Z=83, A=208, stable=False,
                               modes=[{"mode": "electron_capture", "branching_fraction": 1.0}])
        db = {"Pec": parent, "B208": _B}
        daughter = electron_capture(parent, db)
        assert daughter.Z == parent.Z - 1

    def test_a_unchanged(self):
        parent = _make_nuclide("Pec", Z=83, A=208, stable=False,
                               modes=[{"mode": "electron_capture", "branching_fraction": 1.0}])
        db = {"Pec": parent, "B208": _B}
        daughter = electron_capture(parent, db)
        assert daughter.A == parent.A

    def test_stable_raises(self):
        with pytest.raises(ValueError, match="stable"):
            electron_capture(_C, _MINI_DB)


# ---------------------------------------------------------------------------
# decay_chain()
# ---------------------------------------------------------------------------


class TestDecayChain:
    def test_stable_nuclide_returns_single_element(self):
        chain = decay_chain(_C, _MINI_DB)
        assert chain == [_C]

    def test_chain_starts_with_parent(self):
        chain = decay_chain(_A, _MINI_DB)
        assert chain[0] is _A

    def test_chain_ends_with_stable(self):
        chain = decay_chain(_A, _MINI_DB)
        assert chain[-1].stable

    def test_full_chain_length(self):
        # A212 (alpha) → B208 (beta-) → C208 (stable)
        chain = decay_chain(_A, _MINI_DB)
        assert len(chain) == 3

    def test_full_chain_order(self):
        chain = decay_chain(_A, _MINI_DB)
        assert [n.name for n in chain] == ["A212", "B208", "C208"]

    def test_named_daughter_in_db(self):
        """When decay_modes carries an explicit 'daughter' name that is in the db,
        it should be resolved and the chain should continue."""
        chain = decay_chain(_NAMED_PARENT, _NAMED_DB)
        # NP60 → B208 (named) → C208 (stable)
        assert len(chain) == 3
        assert chain[0].name == "NP60"
        assert chain[-1].stable

    def test_named_daughter_not_in_db_stops_chain(self):
        """When the named daughter is absent from the db (e.g. a stable end-product
        not stored there), the chain should stop at the last tracked nuclide."""
        db = {"NM60": _NAMED_PARENT_MISSING}
        chain = decay_chain(_NAMED_PARENT_MISSING, db)
        assert chain == [_NAMED_PARENT_MISSING]

    def test_single_step_chain(self):
        chain = decay_chain(_B, _MINI_DB)
        assert len(chain) == 2
        assert chain[0] is _B
        assert chain[1] is _C

    def test_co60_chain_bundled_db(self, nuclides):
        """Co-60 → Ni-60 (stable, not in bundled db).  Chain is [Co60] only."""
        chain = decay_chain(nuclides["Co60"], nuclides)
        assert chain[0] == nuclides["Co60"]

    def test_max_steps_raises(self):
        """A cycle-free but very long chain should raise if max_steps is hit."""
        # Build a 5-nuclide chain but cap max_steps at 2.
        alpha_mode = [{"mode": "alpha", "branching_fraction": 1.0}]
        beta_mode = [{"mode": "beta-", "branching_fraction": 1.0}]
        n1 = _make_nuclide("N1", Z=84, A=212, stable=False, modes=alpha_mode)
        n2 = _make_nuclide("N2", Z=82, A=208, stable=False, modes=beta_mode)
        n3 = _make_nuclide("N3", Z=83, A=208, stable=False, modes=beta_mode)
        n4 = _make_nuclide("N4", Z=84, A=208, stable=True)
        db = {"N1": n1, "N2": n2, "N3": n3, "N4": n4}
        with pytest.raises(ValueError, match="steps"):
            decay_chain(n1, db, max_steps=2)


# ---------------------------------------------------------------------------
# plot_decay_chain()
# ---------------------------------------------------------------------------


class TestPlotDecayChain:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "chain.png"
        result = plot_decay_chain([_A, _B, _C], out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_nuclide_stable(self, tmp_path):
        out = tmp_path / "stable.png"
        plot_decay_chain([_C], out)
        assert out.exists()

    def test_returns_path_object(self, tmp_path):
        out = tmp_path / "chain.png"
        result = plot_decay_chain([_A, _B, _C], out)
        assert isinstance(result, Path)

    def test_string_path_accepted(self, tmp_path):
        out = str(tmp_path / "chain.png")
        result = plot_decay_chain([_A, _B, _C], out)
        assert Path(out).exists()
        assert result == Path(out)

    def test_pdf_format(self, tmp_path):
        out = tmp_path / "chain.pdf"
        plot_decay_chain([_A, _B, _C], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_chain_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            plot_decay_chain([], tmp_path / "out.png")

    def test_bundled_nuclides_chain(self, nuclides, tmp_path):
        """Smoke-test with real nuclide data from the bundled database."""
        chain = decay_chain(nuclides["Cs137"], nuclides)
        out = tmp_path / "cs137_chain.png"
        plot_decay_chain(chain, out)
        assert out.exists()
