"""Tests for utilities.unit_conversion."""

from __future__ import annotations

import pytest

from utilities.unit_conversion import (
    SI_PREFIX_FACTORS,
    convert_absorbed_dose,
    convert_activity,
    convert_dose_equivalent,
    convert_radiation_unit,
    prefix_factor,
)


class TestConvertRadiationUnit:
    def test_sievert_to_rem(self):
        assert convert_radiation_unit(1.0, "Sv", "rem") == pytest.approx(100.0)

    def test_rem_to_sievert(self):
        assert convert_radiation_unit(250.0, "rem", "Sv") == pytest.approx(2.5)

    def test_becquerel_to_curie(self):
        assert convert_radiation_unit(3.7e10, "Bq", "Ci") == pytest.approx(1.0)

    def test_curie_to_becquerel(self):
        assert convert_radiation_unit(2.0, "Ci", "Bq") == pytest.approx(7.4e10)

    def test_gray_to_rad(self):
        assert convert_radiation_unit(0.5, "Gy", "rad") == pytest.approx(50.0)

    def test_prefixed_activity_units(self):
        assert convert_radiation_unit(1.0, "mCi", "MBq") == pytest.approx(37.0)

    def test_prefixed_dose_equivalent_units(self):
        assert convert_radiation_unit(250.0, "uSv", "mrem") == pytest.approx(25.0)

    def test_micro_symbol_is_supported(self):
        assert convert_radiation_unit(1.0, "µCi", "kBq") == pytest.approx(37.0)

    def test_spelled_out_names_are_supported(self):
        assert convert_radiation_unit(1.0, "curie", "becquerels") == pytest.approx(3.7e10)

    def test_incompatible_families_raise(self):
        with pytest.raises(ValueError, match="cannot convert"):
            convert_radiation_unit(1.0, "Sv", "Bq")

    def test_unsupported_unit_raises(self):
        with pytest.raises(ValueError, match="unsupported unit"):
            convert_radiation_unit(1.0, "R", "Sv")


class TestTypedConversions:
    def test_convert_activity(self):
        assert convert_activity(1000.0, "kBq", "MBq") == pytest.approx(1.0)

    def test_convert_dose_equivalent(self):
        assert convert_dose_equivalent(1.0, "mSv", "uSv") == pytest.approx(1000.0)

    def test_convert_absorbed_dose(self):
        assert convert_absorbed_dose(2.0, "rad", "mGy") == pytest.approx(20.0)

    def test_typed_conversion_rejects_wrong_family(self):
        with pytest.raises(ValueError, match="expected activity"):
            convert_activity(1.0, "Sv", "rem")


class TestPrefixFactor:
    def test_known_prefixes(self):
        assert prefix_factor("m") == pytest.approx(1e-3)
        assert prefix_factor("u") == pytest.approx(1e-6)
        assert prefix_factor("µ") == pytest.approx(1e-6)
        assert prefix_factor("M") == pytest.approx(1e6)

    def test_supported_prefix_table_contains_common_entries(self):
        assert SI_PREFIX_FACTORS[""] == pytest.approx(1.0)
        assert SI_PREFIX_FACTORS["k"] == pytest.approx(1e3)
        assert SI_PREFIX_FACTORS["n"] == pytest.approx(1e-9)

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="unsupported SI prefix"):
            prefix_factor("x")
