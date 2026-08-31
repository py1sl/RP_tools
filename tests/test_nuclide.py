"""Tests for utilities.nuclide – Nuclide class and load_nuclides() loader."""

from __future__ import annotations

import json

import pytest

from utilities.nuclide import (
    Nuclide,
    is_in_chain,
    load_nuclides,
    normalize_nuclide_name,
    nuclides_of_element,
    stable_daughters,
)


# ---------------------------------------------------------------------------
# load_nuclides() – loading and structure
# ---------------------------------------------------------------------------


class TestLoadNuclides:
    def test_returns_dict(self, nuclides):
        assert isinstance(nuclides, dict)

    def test_expected_keys_present(self, nuclides):
        for key in ("Co60", "Fe55", "Cs137", "I131", "H3", "Sr90", "Fe56", "Co59"):
            assert key in nuclides, f"Expected nuclide '{key}' not found in loaded data"

    def test_values_are_nuclide_instances(self, nuclides):
        for name, nuc in nuclides.items():
            assert isinstance(nuc, Nuclide), f"Entry '{name}' is not a Nuclide instance"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_nuclides("/nonexistent/path/nuclides.json")

    def test_missing_nuclides_key_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(json.dumps({"wrong_key": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'nuclides' key"):
            load_nuclides(bad_json)

    def test_custom_data_file(self, tmp_path):
        """load_nuclides accepts a custom file path."""
        custom = {
            "nuclides": {
                "Fe56": {
                    "name": "Fe56",
                    "long_name": "Iron-56",
                    "symbol": "Fe",
                    "A": 56,
                    "Z": 26,
                    "stable": True,
                }
            }
        }
        custom_file = tmp_path / "custom.json"
        custom_file.write_text(json.dumps(custom), encoding="utf-8")
        result = load_nuclides(custom_file)
        assert "Fe56" in result
        assert result["Fe56"].stable is True


# ---------------------------------------------------------------------------
# Nuclide – stable nuclides
# ---------------------------------------------------------------------------


class TestStableNuclide:
    def test_fe56_is_stable(self, nuclides):
        assert nuclides["Fe56"].stable is True

    def test_stable_half_life_is_none(self, nuclides):
        assert nuclides["Fe56"].half_life_seconds is None
        assert nuclides["Fe56"].half_life_years is None

    def test_stable_decay_modes_empty(self, nuclides):
        assert nuclides["Fe56"].decay_modes == []

    def test_stable_gamma_lines_empty(self, nuclides):
        assert nuclides["Fe56"].gamma_lines == []

    def test_co59_properties(self, nuclides):
        co59 = nuclides["Co59"]
        assert co59.symbol == "Co"
        assert co59.A == 59
        assert co59.Z == 27
        assert co59.N == 32


# ---------------------------------------------------------------------------
# Nuclide – unstable nuclides (Co-60)
# ---------------------------------------------------------------------------


class TestCo60:
    def test_not_stable(self, nuclides):
        assert nuclides["Co60"].stable is False

    def test_atomic_numbers(self, nuclides):
        co60 = nuclides["Co60"]
        assert co60.Z == 27
        assert co60.A == 60
        assert co60.N == 33

    def test_half_life_seconds(self, nuclides):
        # IAEA: 5.2713 years = 1.66348e8 s (within 0.1 %)
        assert abs(nuclides["Co60"].half_life_seconds - 1.66348e8) / 1.66348e8 < 0.001

    def test_half_life_years(self, nuclides):
        assert abs(nuclides["Co60"].half_life_years - 5.2713) < 0.01

    def test_decay_modes_count(self, nuclides):
        assert len(nuclides["Co60"].decay_modes) == 2

    def test_decay_mode_fields(self, nuclides):
        mode = nuclides["Co60"].decay_modes[0]
        assert "mode" in mode
        assert "branching_fraction" in mode
        assert "daughter" in mode

    def test_dominant_decay_mode(self, nuclides):
        modes = nuclides["Co60"].decay_modes
        dominant = max(modes, key=lambda m: m["branching_fraction"])
        assert dominant["mode"] == "beta-"
        assert dominant["branching_fraction"] > 0.99

    def test_gamma_lines(self, nuclides):
        gamma = nuclides["Co60"].gamma_lines
        assert len(gamma) == 2

    def test_gamma_energies(self, nuclides):
        energies = {round(g["energy_MeV"], 3) for g in nuclides["Co60"].gamma_lines}
        assert 1.173 in energies
        assert 1.333 in energies

    def test_gamma_intensities_near_100_percent(self, nuclides):
        for line in nuclides["Co60"].gamma_lines:
            assert line["intensity_percent"] > 99.0

    def test_beta_lines(self, nuclides):
        assert len(nuclides["Co60"].beta_lines) > 0


# ---------------------------------------------------------------------------
# Nuclide – Fe-55 (electron capture, X-rays)
# ---------------------------------------------------------------------------


class TestFe55:
    def test_not_stable(self, nuclides):
        assert nuclides["Fe55"].stable is False

    def test_decay_mode_is_ec(self, nuclides):
        modes = nuclides["Fe55"].decay_modes
        assert len(modes) == 1
        assert modes[0]["mode"] == "electron_capture"

    def test_no_gamma_lines(self, nuclides):
        assert nuclides["Fe55"].gamma_lines == []

    def test_x_ray_lines_present(self, nuclides):
        xrays = nuclides["Fe55"].x_ray_lines
        assert len(xrays) >= 2

    def test_x_ray_energies_in_keV_range(self, nuclides):
        for line in nuclides["Fe55"].x_ray_lines:
            # Mn K-shell X-rays are in the 5–7 keV range
            assert 0.004 < line["energy_MeV"] < 0.008


# ---------------------------------------------------------------------------
# Nuclide – Cs-137
# ---------------------------------------------------------------------------


class TestCs137:
    def test_not_stable(self, nuclides):
        assert nuclides["Cs137"].stable is False

    def test_half_life_about_30_years(self, nuclides):
        hl_years = nuclides["Cs137"].half_life_years
        assert 29.0 < hl_years < 32.0

    def test_gamma_line_662_keV(self, nuclides):
        energies = [g["energy_MeV"] for g in nuclides["Cs137"].gamma_lines]
        assert any(abs(e - 0.6617) < 0.001 for e in energies)

    def test_beta_lines(self, nuclides):
        assert len(nuclides["Cs137"].beta_lines) == 2


# ---------------------------------------------------------------------------
# Nuclide – I-131
# ---------------------------------------------------------------------------


class TestI131:
    def test_not_stable(self, nuclides):
        assert nuclides["I131"].stable is False

    def test_half_life_about_8_days(self, nuclides):
        hl_years = nuclides["I131"].half_life_years
        assert 0.021 < hl_years < 0.023

    def test_gamma_line_364_keV(self, nuclides):
        energies = [g["energy_MeV"] for g in nuclides["I131"].gamma_lines]
        assert any(abs(e - 0.3645) < 0.001 for e in energies)


# ---------------------------------------------------------------------------
# Nuclide – validation / edge cases
# ---------------------------------------------------------------------------


class TestNuclideValidation:
    _base = {
        "name": "X1",
        "long_name": "Element-1",
        "symbol": "X",
        "A": 1,
        "Z": 0,
        "stable": True,
    }

    def test_invalid_A_raises(self):
        data = {**self._base, "A": 0}
        with pytest.raises(ValueError, match="Mass number A"):
            Nuclide(data)

    def test_negative_Z_raises(self):
        data = {**self._base, "Z": -1}
        with pytest.raises(ValueError, match="Atomic number Z"):
            Nuclide(data)

    def test_zero_half_life_raises(self):
        data = {
            **self._base,
            "stable": False,
            "half_life_seconds": 0.0,
            "half_life_years": 0.0,
            "decay_modes": [],
        }
        with pytest.raises(ValueError, match="half_life_seconds"):
            Nuclide(data)

    def test_negative_half_life_raises(self):
        data = {
            **self._base,
            "stable": False,
            "half_life_seconds": -1.0,
            "half_life_years": -1.0,
            "decay_modes": [],
        }
        with pytest.raises(ValueError, match="half_life_seconds"):
            Nuclide(data)

    def test_repr_stable(self, nuclides):
        assert "stable" in repr(nuclides["Fe56"])

    def test_repr_unstable(self, nuclides):
        assert "T½=" in repr(nuclides["Co60"])

    def test_equality(self, nuclides):
        assert nuclides["Co60"] == nuclides["Co60"]

    def test_inequality(self, nuclides):
        assert nuclides["Co60"] != nuclides["Fe56"]

    def test_hashable(self, nuclides):
        s = {nuclides["Co60"], nuclides["Fe56"]}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Nuclide – element_name and zaid properties
# ---------------------------------------------------------------------------


class TestElementNameAndZAID:
    def test_element_name_co60(self, nuclides):
        assert nuclides["Co60"].element_name == "Cobalt"

    def test_element_name_cs137(self, nuclides):
        assert nuclides["Cs137"].element_name == "Caesium"

    def test_element_name_i131(self, nuclides):
        assert nuclides["I131"].element_name == "Iodine"

    def test_element_name_fe56(self, nuclides):
        assert nuclides["Fe56"].element_name == "Iron"

    def test_zaid_co60(self, nuclides):
        # Z=27, A=60 → 27*1000 + 60 = 27060
        assert nuclides["Co60"].zaid == 27060

    def test_zaid_cs137(self, nuclides):
        # Z=55, A=137 → 55137
        assert nuclides["Cs137"].zaid == 55137

    def test_zaid_i131(self, nuclides):
        # Z=53, A=131 → 53131
        assert nuclides["I131"].zaid == 53131

    def test_zaid_fe56(self, nuclides):
        # Z=26, A=56 → 26056
        assert nuclides["Fe56"].zaid == 26056

    def test_zaid_formula(self, nuclides):
        for nuc in nuclides.values():
            assert nuc.zaid == nuc.Z * 1000 + nuc.A


class TestNormalizeNuclideName:
    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("U-235", "U235"),
            ("U235", "U235"),
            ("235U", "U235"),
            ("  co-60  ", "Co60"),
            ("137-cs", "Cs137"),
            ("001H", "H1"),
        ],
    )
    def test_normalizes_supported_formats(self, input_name, expected):
        assert normalize_nuclide_name(input_name) == expected

    @pytest.mark.parametrize("bad_name", ["", "U", "235", "U-0", "abc", "U--235"])
    def test_invalid_names_raise(self, bad_name):
        with pytest.raises(ValueError):
            normalize_nuclide_name(bad_name)


# ---------------------------------------------------------------------------
# nuclides_of_element
# ---------------------------------------------------------------------------


class TestNuclidesOfElement:
    def test_returns_list(self, nuclides):
        result = nuclides_of_element("Co", nuclides)
        assert isinstance(result, list)

    def test_all_have_correct_symbol(self, nuclides):
        for nuc in nuclides_of_element("Co", nuclides):
            assert nuc.symbol == "Co"

    def test_sorted_by_mass_number(self, nuclides):
        masses = [n.A for n in nuclides_of_element("Co", nuclides)]
        assert masses == sorted(masses)

    def test_contains_co59_and_co60(self, nuclides):
        names = {n.name for n in nuclides_of_element("Co", nuclides)}
        assert "Co59" in names
        assert "Co60" in names

    def test_case_insensitive(self, nuclides):
        lower = nuclides_of_element("co", nuclides)
        upper = nuclides_of_element("CO", nuclides)
        mixed = nuclides_of_element("Co", nuclides)
        assert lower == upper == mixed

    def test_unknown_element_returns_empty(self, nuclides):
        assert nuclides_of_element("Xx", nuclides) == []

    def test_empty_symbol_raises(self, nuclides):
        with pytest.raises(ValueError):
            nuclides_of_element("", nuclides)

    def test_two_letter_symbol(self, nuclides):
        cs_nuclides = nuclides_of_element("Cs", nuclides)
        assert all(n.symbol == "Cs" for n in cs_nuclides)
        assert any(n.name == "Cs137" for n in cs_nuclides)


# ---------------------------------------------------------------------------
# Helpers – minimal nuclide databases for stable_daughters / is_in_chain
# ---------------------------------------------------------------------------

def _make_nuclide(
    name: str,
    symbol: str,
    Z: int,
    A: int,
    stable: bool,
    decay_modes: list | None = None,
    half_life_seconds: float = 1e6,
) -> Nuclide:
    """Build a minimal :class:`Nuclide` for testing."""
    data: dict = {
        "name": name,
        "long_name": f"{symbol}-{A}",
        "symbol": symbol,
        "A": A,
        "Z": Z,
        "stable": stable,
    }
    if not stable:
        data["half_life_seconds"] = half_life_seconds
        data["half_life_years"] = half_life_seconds / 3.156e7
        data["decay_modes"] = decay_modes or []
    return Nuclide(data)


# A–B–C linear chain: A (unstable) → B (unstable, via explicit daughter key)
# → C (stable, via explicit daughter key).
_NUC_C = _make_nuclide("C1", "Xx", 10, 30, stable=True)
_NUC_B = _make_nuclide(
    "B1", "Xy", 11, 31, stable=False,
    decay_modes=[{"mode": "beta-", "branching_fraction": 1.0, "daughter": "C1"}],
)
_NUC_A = _make_nuclide(
    "A1", "Xz", 12, 32, stable=False,
    decay_modes=[{"mode": "beta-", "branching_fraction": 1.0, "daughter": "B1"}],
)
_LINEAR_DB: dict[str, Nuclide] = {"A1": _NUC_A, "B1": _NUC_B, "C1": _NUC_C}

# Branching chain: D → (E stable, F unstable → G stable)
_NUC_G = _make_nuclide("G1", "Ga", 30, 70, stable=True)
_NUC_E = _make_nuclide("E1", "Ea", 20, 50, stable=True)
_NUC_F = _make_nuclide(
    "F1", "Fa", 21, 51, stable=False,
    decay_modes=[{"mode": "beta-", "branching_fraction": 1.0, "daughter": "G1"}],
)
_NUC_D = _make_nuclide(
    "D1", "Da", 22, 52, stable=False,
    decay_modes=[
        {"mode": "beta-", "branching_fraction": 0.6, "daughter": "E1"},
        {"mode": "beta+", "branching_fraction": 0.4, "daughter": "F1"},
    ],
)
_BRANCH_DB: dict[str, Nuclide] = {
    "D1": _NUC_D, "E1": _NUC_E, "F1": _NUC_F, "G1": _NUC_G,
}


# ---------------------------------------------------------------------------
# stable_daughters
# ---------------------------------------------------------------------------


class TestStableDaughters:
    # --- stable input ---

    def test_stable_nuclide_returns_itself(self, nuclides):
        result = stable_daughters(nuclides["Fe56"], nuclides)
        assert result == {nuclides["Fe56"]}

    # --- linear chain (A → B → C stable) ---

    def test_linear_chain_finds_end(self):
        result = stable_daughters(_NUC_A, _LINEAR_DB)
        assert result == {_NUC_C}

    def test_linear_chain_result_is_stable(self):
        for nuc in stable_daughters(_NUC_A, _LINEAR_DB):
            assert nuc.stable

    def test_direct_daughter_stable(self):
        # B → C (stable): should find C directly.
        result = stable_daughters(_NUC_B, _LINEAR_DB)
        assert result == {_NUC_C}

    def test_returns_set_of_nuclides(self):
        result = stable_daughters(_NUC_A, _LINEAR_DB)
        assert isinstance(result, set)
        assert all(isinstance(n, Nuclide) for n in result)

    # --- branching chain ---

    def test_branching_chain_finds_both_ends(self):
        result = stable_daughters(_NUC_D, _BRANCH_DB)
        # E (direct stable branch) and G (via F) should both appear.
        assert _NUC_E in result
        assert _NUC_G in result

    def test_branching_chain_only_stable(self):
        result = stable_daughters(_NUC_D, _BRANCH_DB)
        assert all(n.stable for n in result)

    # --- daughter absent from DB (graceful stop) ---

    def test_daughter_absent_from_db_returns_empty(self, nuclides):
        # Co-60's daughter Ni-60 is not in the bundled DB; graceful stop → empty set.
        result = stable_daughters(nuclides["Co60"], nuclides)
        assert isinstance(result, set)
        assert len(result) == 0

    # --- bundled DB: Fe-56 ---

    def test_fe56_stable_returns_itself(self, nuclides):
        assert stable_daughters(nuclides["Fe56"], nuclides) == {nuclides["Fe56"]}


# ---------------------------------------------------------------------------
# is_in_chain
# ---------------------------------------------------------------------------


class TestIsInChain:
    # --- linear chain ---

    def test_direct_daughter_is_in_chain(self):
        assert is_in_chain(_NUC_A, _NUC_B, _LINEAR_DB) is True

    def test_indirect_daughter_is_in_chain(self):
        assert is_in_chain(_NUC_A, _NUC_C, _LINEAR_DB) is True

    def test_unrelated_nuclide_not_in_chain(self, nuclides):
        assert is_in_chain(nuclides["Co60"], nuclides["Cs137"], nuclides) is False

    def test_parent_not_in_own_chain(self):
        assert is_in_chain(_NUC_A, _NUC_A, _LINEAR_DB) is False

    def test_stable_parent_returns_false(self, nuclides):
        assert is_in_chain(nuclides["Fe56"], nuclides["Co60"], nuclides) is False

    def test_returns_bool(self):
        result = is_in_chain(_NUC_A, _NUC_B, _LINEAR_DB)
        assert isinstance(result, bool)

    # --- branching chain ---

    def test_branching_direct_stable_branch(self):
        assert is_in_chain(_NUC_D, _NUC_E, _BRANCH_DB) is True

    def test_branching_indirect_via_unstable(self):
        # D → F → G; G should be found.
        assert is_in_chain(_NUC_D, _NUC_G, _BRANCH_DB) is True

    def test_branching_intermediate_found(self):
        assert is_in_chain(_NUC_D, _NUC_F, _BRANCH_DB) is True

    # --- daughter absent from DB (graceful stop → returns False) ---

    def test_daughter_absent_from_db_returns_false(self, nuclides):
        # Co-60's daughter Ni-60 is not in the bundled DB.
        # is_in_chain should return False gracefully (cannot find it).
        result = is_in_chain(nuclides["Co60"], nuclides["Fe56"], nuclides)
        assert result is False
