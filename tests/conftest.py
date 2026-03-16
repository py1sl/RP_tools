"""Shared pytest fixtures for RP_tools tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from utilities.nuclide import load_nuclides


@pytest.fixture(scope="session")
def nuclides():
    """Return the full nuclide dictionary loaded from the bundled data file."""
    return load_nuclides()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the ``data/`` directory."""
    return Path(__file__).resolve().parent.parent / "data"
