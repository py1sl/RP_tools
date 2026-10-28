"""Tests for utilities.compartment_model."""

import math

import pytest

from utilities.compartment_model import (
    CompartmentModel,
    effective_half_life,
    iodine_thyroid_model,
    single_compartment_retention,
)


def test_effective_half_life():
    assert effective_half_life(8.0, 8.0) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        effective_half_life(0, 1)


def test_single_compartment_retention():
    assert single_compartment_retention(80, 80) == pytest.approx(0.5)
    assert single_compartment_retention(80, 0) == 1.0
    assert single_compartment_retention(10, 10, 10) == pytest.approx(0.25)


def test_matches_analytic_single_compartment():
    m = CompartmentModel(["a"], {("a", "out"): math.log(2) / 5})
    assert m.amounts({"a": 100}, 10)["a"] == pytest.approx(25.0)


def test_mass_conserved_without_decay():
    m = iodine_thyroid_model(half_life_physical=None)
    q = m.amounts({"blood": 1.0}, 3.0)
    assert 0 < q["thyroid"] < 0.3
    assert q["blood"] + q["thyroid"] < 1.0


def test_iodine_thyroid_peak_and_clearance():
    m = iodine_thyroid_model(half_life_physical=None)
    # long after uptake, thyroid decays with 80 d half-life
    a = m.amounts({"blood": 1.0}, 100.0)["thyroid"]
    b = m.amounts({"blood": 1.0}, 180.0)["thyroid"]
    assert b / a == pytest.approx(0.5, rel=1e-3)


def test_time_series_and_integral():
    m = CompartmentModel(["a"], {("a", "out"): 0.5})
    ts = m.time_series({"a": 1.0}, [0, 1, 2])
    assert ts["a"][0] == 1.0
    assert ts["a"][2] == pytest.approx(math.exp(-1))
    assert m.total_decays({"a": 1.0}, "a", 4.0) == pytest.approx(2 * (1 - math.exp(-2)))


def test_validation():
    with pytest.raises(ValueError):
        CompartmentModel(["a", "a"], {})
    with pytest.raises(ValueError):
        CompartmentModel(["a"], {("b", "a"): 1})
    with pytest.raises(ValueError):
        iodine_thyroid_model(thyroid_uptake_fraction=2)
    with pytest.raises(ValueError):
        CompartmentModel(["a"], {}).amounts({"a": 1}, -1)
