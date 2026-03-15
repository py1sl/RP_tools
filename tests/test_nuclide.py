"""Tests for utilities.nuclide – Nuclide class and load_nuclides() loader."""

from __future__ import annotations

import json

import pytest

from utilities.nuclide import Nuclide, load_nuclides


# ---------------------------------------------------------------------------
# load_nuclides() – loading and structure
# ---------------------------------------------------------------------------


class TestLoadNuclides:
    def test_returns_dict(self, nuclides):
        assert isinstance(nuclides, dict)

    def test_expected_keys_present(self, nuclides):
        for key in ("Co60", "Fe55", "Cs137", "H3", "Sr90", "Fe56", "Co59"):
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
