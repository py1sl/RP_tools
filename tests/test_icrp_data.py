"""Tests for utilities.icrp_data ICRP table loader."""

from __future__ import annotations

import numpy as np
import pytest

from utilities.icrp_data import ICRPDataLibrary, ICRPTable, load_icrp_data


class TestICRPDataLibrary:
    def test_loads_default_publications(self):
        lib = load_icrp_data()
        assert set(lib.publications()) == {"74", "116"}

    def test_particles_for_publication(self):
        lib = load_icrp_data()
        particles_74 = set(lib.particles("74"))
        particles_116 = set(lib.particles("116"))

        assert particles_74 == {"neutrons", "photons"}
        assert {"photons", "electrons", "protons", "photons_kerma"}.issubset(particles_116)

    def test_get_table_returns_table_object(self):
        lib = load_icrp_data()
        table = lib.get_table("116", "photons")

        assert isinstance(table, ICRPTable)
        assert table.publication == "116"
        assert table.particle == "photons"
        assert table.values.ndim == 2
        assert table.values.shape[1] == len(table.columns)

    def test_get_table_normalizes_publication_label(self):
        lib = load_icrp_data()
        t1 = lib.get_table("116", "photons")
        t2 = lib.get_table("ICRP116", "photons")
        np.testing.assert_allclose(t1.values, t2.values)

    def test_missing_table_raises(self):
        lib = load_icrp_data()
        with pytest.raises(KeyError, match="ICRP table not found"):
            lib.get_table("74", "protons")

    def test_tables_for_publication(self):
        lib = load_icrp_data()
        tables = lib.tables_for_publication("74")
        assert set(tables) == {"neutrons", "photons"}


class TestICRPTable:
    def test_energy_column_present(self):
        lib = ICRPDataLibrary()
        table = lib.get_table("74", "photons")

        assert table.columns[0] == "Energy (MeV)"
        energies = table.energies_MeV
        assert energies.ndim == 1
        assert energies.size > 0

    def test_column_lookup(self):
        lib = ICRPDataLibrary()
        table = lib.get_table("74", "photons")

        ap = table.column("AP")
        assert ap.shape == table.energies_MeV.shape
        assert np.all(ap > 0)

    def test_unknown_column_raises(self):
        lib = ICRPDataLibrary()
        table = lib.get_table("74", "photons")

        with pytest.raises(KeyError, match="Column"):
            table.column("NOT_A_COLUMN")

    def test_description_is_read(self):
        lib = ICRPDataLibrary()
        table = lib.get_table("116", "electrons")
        assert "Effective dose per fluence" in table.description
